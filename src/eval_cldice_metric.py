"""
Measure the actual clDice (topology/connectivity) metric on hard binary
predictions, using a real morphological skeleton (skimage.morphology.skeletonize)
rather than the differentiable soft-skeleton used during training.

This exists to answer one question: did fine-tuning with SoftClDiceLoss
(train_mobileunet_cldice.py) actually improve vessel-tree connectivity, even
though it left pixel-overlap Dice/IoU unchanged on the test split? If clDice
also comes back flat, the "fragmented centerlines" hypothesis is not what's
capping accuracy either, and this metric proves it either way.

Read-only: only loads checkpoints, does not modify or retrain anything.

Usage:
    python src/eval_cldice_metric.py --checkpoint checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth
    python src/eval_cldice_metric.py --checkpoint checkpoints/mobileunetv3_cldice/best.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from eval_segmentation import VesselSegDataset
from model_lightweight import MobileUNetv3


def cl_dice_hard(pred, target, smooth=1e-6):
    """pred, target: (H, W) uint8/bool numpy binary masks."""
    if pred.sum() == 0 or target.sum() == 0:
        return None  # skip empty frames, not meaningful for connectivity

    skel_pred = skeletonize(pred > 0)
    skel_true = skeletonize(target > 0)

    tprec = (skel_pred & (target > 0)).sum() / (skel_pred.sum() + smooth)
    tsens = (skel_true & (pred > 0)).sum() / (skel_true.sum() + smooth)
    if tprec + tsens == 0:
        return 0.0
    return 2 * tprec * tsens / (tprec + tsens)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores = []
    for batch in tqdm(loader, desc="Computing hard clDice", unit="img"):
        image = batch["image"].to(device, dtype=torch.float32)
        mask_true = batch["mask"].numpy()

        out = model(image)
        logits = out["out"] if isinstance(out, dict) else out
        pred = (torch.sigmoid(logits).squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)

        for i in range(pred.shape[0]):
            score = cl_dice_hard(pred[i], mask_true[i].astype(np.uint8))
            if score is not None:
                scores.append(score)
    return float(np.mean(scores)), len(scores)


def get_args():
    p = argparse.ArgumentParser(description="Measure hard clDice (connectivity) on a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = VesselSegDataset(args.data_dir, args.split, (args.image_size, args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                         pin_memory=device.type == "cuda")

    model = MobileUNetv3(n_classes=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Checkpoint: {args.checkpoint} (read-only)")

    mean_cldice, n = evaluate(model, loader, device)
    print(f"\nImages scored (non-empty only): {n}")
    print(f"Hard clDice (connectivity): {mean_cldice:.4f}")
