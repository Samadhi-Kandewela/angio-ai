import json
import os
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from localization_labels import STENOSIS_CATEGORY_ID


class CoronaryMultiTaskDataset(Dataset):
    """
    Aligned SYNTAX + stenosis dataset.

    Returns:
      image: float tensor (3, H, W), ImageNet-normalized
      vessel_mask: float tensor (1, H, W), all anatomical segments merged
      anatomy_mask: long tensor (H, W), 0=background, 1..25=SYNTAX segment
      stenosis_mask: float tensor (1, H, W), stenosis polygons
      meta: filename and original size
    """

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

        self.syntax_images_dir = os.path.join(root_dir, "syntax", split, "images")
        self.syntax_json_path = os.path.join(root_dir, "syntax", split, "annotations", f"{split}.json")
        self.stenosis_json_path = os.path.join(root_dir, "stenosis", split, "annotations", f"{split}.json")

        with open(self.syntax_json_path, "r") as f:
            syntax_data = json.load(f)
        with open(self.stenosis_json_path, "r") as f:
            stenosis_data = json.load(f)

        self.images = sorted(syntax_data["images"], key=lambda z: z["file_name"])
        self.syntax_by_image = self._annotations_by_filename(syntax_data)
        self.stenosis_by_image = self._annotations_by_filename(stenosis_data)

        stenosis_names = {img["file_name"] for img in stenosis_data["images"]}
        self.images = [img for img in self.images if img["file_name"] in stenosis_names]

        self.transform = self._build_transform(augment)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
                A.ShiftScaleRotate(
                    shift_limit=0.04,
                    scale_limit=0.08,
                    rotate_limit=8,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.45,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.18,
                    contrast_limit=0.18,
                    p=0.45,
                ),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.20),
            ]
        return A.Compose(transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        file_name = info["file_name"]
        img_path = os.path.join(self.syntax_images_dir, file_name)

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        anatomy_mask = self._build_anatomy_mask(file_name, info["height"], info["width"])
        stenosis_mask = self._build_stenosis_mask(file_name, info["height"], info["width"])

        if self.use_clahe:
            image_rgb = self._apply_clahe_rgb(image_rgb)

        augmented = self.transform(
            image=image_rgb,
            masks=[anatomy_mask, stenosis_mask],
        )
        image_rgb = augmented["image"]
        anatomy_mask, stenosis_mask = augmented["masks"]

        vessel_mask = (anatomy_mask > 0).astype(np.float32)
        stenosis_mask = (stenosis_mask > 0).astype(np.float32)

        image = image_rgb.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))

        return {
            "image": torch.from_numpy(image).float(),
            "vessel_mask": torch.from_numpy(vessel_mask).unsqueeze(0).float(),
            "anatomy_mask": torch.from_numpy(anatomy_mask.astype(np.int64)).long(),
            "stenosis_mask": torch.from_numpy(stenosis_mask).unsqueeze(0).float(),
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
        return mask

    def _build_stenosis_mask(self, file_name, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in self.stenosis_by_image.get(file_name, []):
            if int(ann["category_id"]) == STENOSIS_CATEGORY_ID:
                self._fill_annotation(mask, ann, 1)
        return mask

    def estimate_anatomy_pixel_counts(self, num_classes=26, max_samples=None):
        counts = np.zeros(num_classes, dtype=np.float64)
        images = self.images if max_samples is None else self.images[:max_samples]
        for info in images:
            mask = self._build_anatomy_mask(info["file_name"], info["height"], info["width"])
            ids, pix = np.unique(mask, return_counts=True)
            for class_id, count in zip(ids, pix):
                if 0 <= int(class_id) < num_classes:
                    counts[int(class_id)] += float(count)
        return counts

    def estimate_stenosis_pixel_counts(self, max_samples=None):
        positives = 0.0
        total = 0.0
        images = self.images if max_samples is None else self.images[:max_samples]
        for info in images:
            mask = self._build_stenosis_mask(info["file_name"], info["height"], info["width"])
            positives += float(np.sum(mask > 0))
            total += float(mask.size)
        negatives = max(total - positives, 0.0)
        return positives, negatives

    @staticmethod
    def _fill_annotation(mask, ann, value):
        for seg in ann.get("segmentation", []):
            if len(seg) < 6:
                continue
            poly = np.array(seg, dtype=np.float32).reshape((-1, 2))
            poly = np.round(poly).astype(np.int32)
            cv2.fillPoly(mask, [poly], value)

    @staticmethod
    def _apply_clahe_rgb(image_rgb):
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_chan)
        enhanced_lab = cv2.merge((enhanced_l, a_chan, b_chan))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


if __name__ == "__main__":
    dataset = CoronaryMultiTaskDataset("dataset", split="train", augment=True)
    sample = dataset[0]
    print("image:", tuple(sample["image"].shape))
    print("vessel:", tuple(sample["vessel_mask"].shape), sample["vessel_mask"].max().item())
    print("anatomy:", tuple(sample["anatomy_mask"].shape), sample["anatomy_mask"].unique().tolist())
    print("stenosis:", tuple(sample["stenosis_mask"].shape), sample["stenosis_mask"].max().item())
