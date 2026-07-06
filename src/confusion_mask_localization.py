"""
Per-segment confusion analysis for the mask-input anatomy localization model.

Runs MaskLocalizationNet on ground-truth vessel masks (same setup as
eval_mask_localization.py) and reports:
  - per-segment precision/recall/F1, sorted worst-first
  - the most frequently confused segment pairs

Usage:
    python src/confusion_mask_localization.py \
        --checkpoint checkpoints/mask_localization/best.pth \
        --data-dir dataset --split val --device cpu
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_mask_localization import MaskLocalizationDataset
from localization_labels import (
    MERGED_NUM_ANATOMY_CLASSES as NUM_ANATOMY_CLASSES,
    merged_segment_label as segment_label,
)
from model_mask_localization import MaskLocalizationNet


@torch.no_grad()
def build_confusion_matrix(model, loader, device, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in tqdm(loader, desc="Running inference", unit="batch"):
        vessel_mask = batch["vessel_mask"].to(device, dtype=torch.float32)
        anatomy_mask = batch["anatomy_mask"].to(device, dtype=torch.long)

        pred = torch.argmax(model(vessel_mask)["anatomy"], dim=1)

        valid = anatomy_mask > 0
        t = anatomy_mask[valid].cpu().numpy()
        p = pred[valid].cpu().numpy()
        np.add.at(cm, (t, p), 1)
    return cm


def report(cm, num_classes, top_n=15):
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    pred_totals = cm.sum(axis=0)

    print("\n" + "=" * 70)
    print("PER-SEGMENT ACCURACY (recall) — worst first")
    print("=" * 70)
    print(f"  {'segment':<32} {'support':>8} {'recall':>8} {'precision':>10}")
    rows = []
    for c in range(1, num_classes):
        if support[c] == 0:
            continue
        recall = tp[c] / support[c]
        precision = tp[c] / pred_totals[c] if pred_totals[c] > 0 else 0.0
        rows.append((c, support[c], recall, precision))
    rows.sort(key=lambda r: r[2])
    for c, sup, recall, precision in rows:
        print(f"  {segment_label(c):<32} {sup:>8d} {recall:>8.3f} {precision:>10.3f}")

    print("\n" + "=" * 70)
    print(f"TOP {top_n} MOST CONFUSED SEGMENT PAIRS (true -> predicted)")
    print("=" * 70)
    confusions = []
    for t in range(1, num_classes):
        for p in range(1, num_classes):
            if t != p and cm[t, p] > 0:
                confusions.append((cm[t, p], t, p))
    confusions.sort(reverse=True)
    for count, t, p in confusions[:top_n]:
        pct_of_true = count / support[t] * 100 if support[t] > 0 else 0.0
        print(f"  {segment_label(t):<28} -> {segment_label(p):<28} {count:>7d} px ({pct_of_true:5.1f}% of true class)")
    print("=" * 70)


def get_args():
    p = argparse.ArgumentParser(description="Per-segment confusion analysis for mask localization model")
    p.add_argument("--checkpoint", default="checkpoints/mask_localization/best.pth")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--top-n", type=int, default=15)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    dataset = MaskLocalizationDataset(
        args.data_dir, split=args.split,
        image_size=(args.image_size, args.image_size), augment=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    print(f"Images: {len(dataset)} ({args.split} split)")

    model = MaskLocalizationNet(n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Checkpoint: {args.checkpoint}\n")

    cm = build_confusion_matrix(model, loader, device, NUM_ANATOMY_CLASSES)
    report(cm, NUM_ANATOMY_CLASSES, top_n=args.top_n)
