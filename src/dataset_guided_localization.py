"""
Dataset for guided single-model localization.

Reads only SYNTAX annotations + raw angiogram images.
No stenosis dependency.

Returns:
  image       : float tensor (3, H, W)  ImageNet-normalized RGB
  vessel_mask : float tensor (1, H, W)  binary vessel mask
  anatomy_mask: long  tensor (H, W)     per-pixel SYNTAX segment 0..25
"""

import json
import os
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from localization_labels import STENOSIS_CATEGORY_ID


class GuidedLocalizationDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=(512, 512),
        augment=False,
        use_clahe=True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.use_clahe = use_clahe

        self.images_dir = os.path.join(root_dir, "syntax", split, "images")
        syntax_json = os.path.join(
            root_dir, "syntax", split, "annotations", f"{split}.json"
        )
        with open(syntax_json, "r") as f:
            syntax_data = json.load(f)

        self.images = sorted(syntax_data["images"], key=lambda z: z["file_name"])
        self.syntax_by_image = self._annotations_by_filename(syntax_data)
        self.transform = self._build_transform(augment)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
        h, w = self.image_size
        transforms = [A.Resize(height=h, width=w)]
        if augment:
            transforms += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(
                    translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                    scale=(0.92, 1.08),
                    rotate=(-8, 8),
                    p=0.45,
                ),
                A.ElasticTransform(alpha=30, sigma=5, p=0.30),
                A.GridDistortion(num_steps=4, distort_limit=0.15, p=0.20),
                A.RandomBrightnessContrast(
                    brightness_limit=0.18, contrast_limit=0.18, p=0.45
                ),
                A.GaussNoise(std_range=(0.02, 0.10), p=0.20),
            ]
        return A.Compose(transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        file_name = info["file_name"]

        img_bgr = cv2.imread(os.path.join(self.images_dir, file_name))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {file_name}")
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        anatomy_mask = self._build_anatomy_mask(
            file_name, info["height"], info["width"]
        )

        if self.use_clahe:
            image_rgb = self._apply_clahe(image_rgb)

        augmented = self.transform(image=image_rgb, masks=[anatomy_mask])
        image_rgb  = augmented["image"]
        anatomy_mask = augmented["masks"][0]

        vessel_mask = (anatomy_mask > 0).astype(np.float32)

        image = image_rgb.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))

        return {
            "image":        torch.from_numpy(image).float(),
            "vessel_mask":  torch.from_numpy(vessel_mask).unsqueeze(0).float(),
            "anatomy_mask": torch.from_numpy(anatomy_mask.astype(np.int64)).long(),
            "file_name":    file_name,
            "orig_size":    (info["height"], info["width"]),
        }

    def _build_anatomy_mask(self, file_name, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in self.syntax_by_image.get(file_name, []):
            cat = int(ann["category_id"])
            if cat <= 0 or cat >= STENOSIS_CATEGORY_ID:
                continue
            self._fill_annotation(mask, ann, cat)
        return mask

    def estimate_anatomy_pixel_counts(self, num_classes=26, max_samples=None):
        counts = np.zeros(num_classes, dtype=np.float64)
        images = self.images if max_samples is None else self.images[:max_samples]
        for info in images:
            mask = self._build_anatomy_mask(
                info["file_name"], info["height"], info["width"]
            )
            ids, pix = np.unique(mask, return_counts=True)
            for cid, cnt in zip(ids, pix):
                if 0 <= int(cid) < num_classes:
                    counts[int(cid)] += float(cnt)
        return counts

    @staticmethod
    def _fill_annotation(mask, ann, value):
        for seg in ann.get("segmentation", []):
            if len(seg) < 6:
                continue
            poly = np.array(seg, dtype=np.float32).reshape((-1, 2))
            cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], value)

    @staticmethod
    def _apply_clahe(image_rgb):
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


if __name__ == "__main__":
    ds = GuidedLocalizationDataset("dataset", split="train", augment=True)
    s = ds[0]
    print("image:       ", tuple(s["image"].shape))
    print("vessel_mask: ", tuple(s["vessel_mask"].shape))
    print("anatomy_mask:", tuple(s["anatomy_mask"].shape), s["anatomy_mask"].unique().tolist())
    print(f"Dataset size: {len(ds)}")
