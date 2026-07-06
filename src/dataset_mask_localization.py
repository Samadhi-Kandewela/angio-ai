"""
Dataset for mask-input anatomy localization.

Reads only SYNTAX annotations (no stenosis needed).

Returns:
  vessel_mask : float tensor (1, H, W)  — binary vessel mask  (MODEL INPUT)
  anatomy_mask: long  tensor (H, W)     — per-pixel merged segment 0..14 (TARGET)
  file_name   : str
  orig_size   : (H, W)

Raw 1-25 SYNTAX segment ids are collapsed to the 14-class merged scheme
(see localization_labels.SEGMENT_MERGE_MAP) before being returned, since rare
side-branch classes are not learnable from this dataset's support counts.
"""

import json
import os
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from localization_labels import STENOSIS_CATEGORY_ID, RAW_TO_MERGED_ID, MERGED_NUM_ANATOMY_CLASSES

_RAW_TO_MERGED_LUT = np.array(RAW_TO_MERGED_ID, dtype=np.uint8)


class MaskLocalizationDataset(Dataset):
    """
    Provides (vessel_mask, anatomy_mask) pairs from SYNTAX annotations.

    The vessel_mask is derived from anatomy_mask (foreground pixels = 1).
    Augmentations are applied consistently to both masks.

    Optional mask noise (augment=True) dilates/erodes the input mask slightly
    to simulate imperfect predictions from the upstream segmentation model.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=(512, 512),
        augment=False,
        mask_noise=True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.mask_noise = mask_noise and augment  # only during training

        syntax_json_path = os.path.join(
            root_dir, "syntax", split, "annotations", f"{split}.json"
        )
        with open(syntax_json_path, "r") as f:
            syntax_data = json.load(f)

        self.images = sorted(syntax_data["images"], key=lambda z: z["file_name"])
        self.syntax_by_image = self._annotations_by_filename(syntax_data)
        self.transform = self._build_transform(augment)

    @staticmethod
    def _annotations_by_filename(coco_data):
        id_to_name = {img["id"]: img["file_name"] for img in coco_data["images"]}
        out = defaultdict(list)
        for ann in coco_data["annotations"]:
            name = id_to_name.get(ann["image_id"])
            if name is not None:
                out[name].append(ann)
        return out

    def _build_transform(self, augment):
        height, width = self.image_size
        transforms = [A.Resize(height=height, width=width)]
        if augment:
            transforms += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    scale=(0.90, 1.10),
                    rotate=(-12, 12),
                    p=0.5,
                ),
                A.ElasticTransform(alpha=40, sigma=6, p=0.35),
                A.GridDistortion(num_steps=4, distort_limit=0.18, p=0.25),
            ]
        return A.Compose(transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        file_name = info["file_name"]

        anatomy_mask = self._build_anatomy_mask(
            file_name, info["height"], info["width"]
        )

        # albumentations requires an `image` arg; use a dummy so only masks are transformed
        h, w = anatomy_mask.shape
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        augmented = self.transform(image=dummy, masks=[anatomy_mask])
        anatomy_mask = augmented["masks"][0]

        vessel_mask = (anatomy_mask > 0).astype(np.float32)

        if self.mask_noise:
            vessel_mask = self._apply_mask_noise(vessel_mask)

        return {
            "vessel_mask": torch.from_numpy(vessel_mask).unsqueeze(0).float(),
            "anatomy_mask": torch.from_numpy(anatomy_mask.astype(np.int64)).long(),
            "file_name": file_name,
            "orig_size": (info["height"], info["width"]),
        }

    def _build_anatomy_mask(self, file_name, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in self.syntax_by_image.get(file_name, []):
            category_id = int(ann["category_id"])
            if category_id <= 0 or category_id >= STENOSIS_CATEGORY_ID:
                continue
            self._fill_annotation(mask, ann, category_id)
        return _RAW_TO_MERGED_LUT[mask]

    @staticmethod
    def _apply_mask_noise(vessel_mask):
        """Randomly dilate or erode the binary mask to simulate segmentation noise."""
        kernel_size = np.random.choice([1, 2, 3])
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1)
        )
        op = np.random.choice(["dilate", "erode", "none"])
        mask_uint8 = (vessel_mask * 255).astype(np.uint8)
        if op == "dilate":
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
        elif op == "erode":
            mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
        return (mask_uint8 > 127).astype(np.float32)

    @staticmethod
    def _fill_annotation(mask, ann, value):
        for seg in ann.get("segmentation", []):
            if len(seg) < 6:
                continue
            poly = np.array(seg, dtype=np.float32).reshape((-1, 2))
            poly = np.round(poly).astype(np.int32)
            cv2.fillPoly(mask, [poly], value)

    def estimate_anatomy_pixel_counts(self, num_classes=None, max_samples=None):
        if num_classes is None:
            num_classes = MERGED_NUM_ANATOMY_CLASSES
        counts = np.zeros(num_classes, dtype=np.float64)
        images = self.images if max_samples is None else self.images[:max_samples]
        for info in images:
            mask = self._build_anatomy_mask(
                info["file_name"], info["height"], info["width"]
            )
            ids, pix = np.unique(mask, return_counts=True)
            for class_id, count in zip(ids, pix):
                if 0 <= int(class_id) < num_classes:
                    counts[int(class_id)] += float(count)
        return counts


if __name__ == "__main__":
    ds = MaskLocalizationDataset("dataset", split="train", augment=True)
    sample = ds[0]
    print("vessel_mask:", tuple(sample["vessel_mask"].shape), sample["vessel_mask"].max().item())
    print("anatomy_mask:", tuple(sample["anatomy_mask"].shape), sample["anatomy_mask"].unique().tolist())
    print(f"Dataset size: {len(ds)}")
