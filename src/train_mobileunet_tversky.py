"""
Fine-tune MobileUNetv3 with a recall-weighted Focal Tversky loss.

Baseline (checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth) trained
with BCE + plain Dice scores Dice 0.778 / IoU 0.648 / Precision 0.820 /
Recall 0.755 on the syntax test split (see eval_segmentation.py). Recall
trailing precision means the model under-detects vessel pixels (thin/faint
branches), which plain Dice does not correct for since it weighs false
positives and false negatives equally.

This script starts from that checkpoint and continues training with
BCE + FocalTverskyLoss(alpha=0.3, beta=0.7), which penalizes false negatives
more than false positives, and tracks Dice/IoU/Precision/Recall on the val
split every epoch so the trade-off against precision is visible.

Usage:
    python src/train_mobileunet_tversky.py --epochs 20 --device cuda
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import SegmentationDataset
from losses import FocalTverskyLoss
from model_lightweight import MobileUNetv3


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
def evaluate(net, loader, device):
    net.eval()
    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    n = 0
    for images, masks in loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32).unsqueeze(1)

        out = net(images)
        logits = out["out"] if isinstance(out, dict) else out
        pred = (torch.sigmoid(logits) > 0.5).float()

        for i in range(pred.size(0)):
            dice, iou, precision, recall = segmentation_metrics(pred[i], masks[i])
            totals["dice"] += dice
            totals["iou"] += iou
            totals["precision"] += precision
            totals["recall"] += recall
            n += 1

    net.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def train(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Device: {device}")

    train_set = SegmentationDataset(args.data_dir, split="train", image_size=(args.image_size, args.image_size))
    val_set = SegmentationDataset(args.data_dir, split="val", image_size=(args.image_size, args.image_size))
    logging.info(f"Train: {len(train_set)} images, Val: {len(val_set)} images")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    net = MobileUNetv3(n_classes=1, pretrained=not args.init_checkpoint).to(device)
    if args.init_checkpoint:
        net.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        logging.info(f"Initialized weights from {args.init_checkpoint}")

    bce = nn.BCEWithLogitsLoss()
    focal_tversky = FocalTverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta, gamma=args.tversky_gamma)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(args.epochs):
        net.train()
        epoch_loss = 0.0
        with tqdm(total=len(train_set), desc=f"Epoch {epoch + 1}/{args.epochs}", unit="img") as pbar:
            for images, masks in train_loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32).unsqueeze(1)

                out = net(images)
                logits = out["out"] if isinstance(out, dict) else out
                probs = torch.sigmoid(logits)

                loss = (args.bce_weight * bce(logits, masks)
                        + args.tversky_weight * focal_tversky(probs, masks))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{"loss (batch)": loss.item()})
                pbar.update(images.shape[0])

        metrics = evaluate(net, val_loader, device)
        scheduler.step(metrics["dice"])
        logging.info(
            f"Epoch {epoch + 1}: val Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f} "
            f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f}"
        )

        torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "last.pth"))
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "best.pth"))
            logging.info(f"New best val Dice {best_dice:.4f} -> saved {args.checkpoint_dir}/best.pth")

    logging.info(f"Training complete. Best val Dice: {best_dice:.4f}")


def get_args():
    p = argparse.ArgumentParser(description="Fine-tune MobileUNetv3 with a recall-weighted Focal Tversky loss")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--init-checkpoint", default="checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth",
                   help="Warm-start weights; pass '' to train from an ImageNet-pretrained backbone instead")
    p.add_argument("--checkpoint-dir", default="checkpoints/mobileunetv3_tversky")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--tversky-weight", type=float, default=0.5)
    p.add_argument("--tversky-alpha", type=float, default=0.3, help="False-positive weight")
    p.add_argument("--tversky-beta", type=float, default=0.7, help="False-negative weight (> alpha biases toward recall)")
    p.add_argument("--tversky-gamma", type=float, default=0.75, help="<1 up-weights hard/low-overlap samples")
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    try:
        train(args)
    except KeyboardInterrupt:
        logging.info("Interrupted")
        sys.exit(0)
