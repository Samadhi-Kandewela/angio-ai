"""
Evaluate the MobileUNetv3 vessel segmentation model alone (no localization).

Ground truth is the binary vessel mask derived from the SYNTAX annotations
(same source used by eval_pipeline.py), so results are directly comparable
to the segmentation half of the end-to-end pipeline eval.

Usage:
    python src/eval_segmentation.py \
        --checkpoint checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth \
        --data-dir dataset --split val --device cuda
"""

import argparse
import os
import sys

import albumentations as A
import cv2
import json
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model_lightweight import MobileUNetv3


# ─── Dataset ──────────────────────────────────────────────────────────────────

class VesselSegDataset(Dataset):
    """Loads syntax-split images + binary vessel masks (all SYNTAX segment
    polygons merged into one foreground class)."""

    def __init__(self, data_dir, split, image_size):
        self.images_dir = os.path.join(data_dir, "syntax", split, "images")
        json_path = os.path.join(data_dir, "syntax", split, "annotations", f"{split}.json")
        with open(json_path, "r") as f:
            coco = json.load(f)

        self.images = sorted(coco["images"], key=lambda z: z["file_name"])
        id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
        self.anns_by_name = {}
        for ann in coco["annotations"]:
            name = id_to_name.get(ann["image_id"])
            if name is not None:
                self.anns_by_name.setdefault(name, []).append(ann)

        h, w = image_size
        self.image_size = image_size
        self.image_transform = A.Compose([
            A.Resize(height=h, width=w),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.mask_transform = A.Resize(height=h, width=w, interpolation=cv2.INTER_NEAREST)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        file_name = info["file_name"]

        img_path = os.path.join(self.images_dir, file_name)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image_rgb)["image"]

        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for ann in self.anns_by_name.get(file_name, []):
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                poly = np.round(np.array(seg, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
        mask = self.mask_transform(image=mask)["image"]

        return {
            "image": image,
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "file_name": file_name,
        }


# ─── Metrics ──────────────────────────────────────────────────────────────────

def segmentation_metrics(pred, target, smooth=1.0):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return dice.item(), iou.item(), precision.item(), recall.item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    n = 0

    for batch in tqdm(loader, desc="Evaluating segmentation", unit="img"):
        image = batch["image"].to(device, dtype=torch.float32)
        mask_true = batch["mask"].to(device)

        out = model(image)
        logits = out["out"] if isinstance(out, dict) else out
        pred = (torch.sigmoid(logits).squeeze(1) > 0.5).float()

        for i in range(pred.size(0)):
            dice, iou, precision, recall = segmentation_metrics(pred[i], mask_true[i])
            totals["dice"] += dice
            totals["iou"] += iou
            totals["precision"] += precision
            totals["recall"] += recall
            n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}, n


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Evaluate the MobileUNetv3 segmentation model alone")
    p.add_argument("--checkpoint", default="checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    dataset = VesselSegDataset(args.data_dir, args.split, (args.image_size, args.image_size))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    print(f"Images: {len(dataset)} ({args.split} split)")

    model = MobileUNetv3(n_classes=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Checkpoint: {args.checkpoint}\n")

    metrics, n = evaluate(model, loader, device)

    print("\n" + "=" * 45)
    print("MOBILEUNET-V3 SEGMENTATION EVALUATION")
    print("=" * 45)
    print(f"  Images evaluated: {n}")
    print(f"  Dice:             {metrics['dice']:.4f}")
    print(f"  IoU:              {metrics['iou']:.4f}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print("=" * 45)
