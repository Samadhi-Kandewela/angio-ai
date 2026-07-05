"""
Fine-tune MobileUNetv3 with BCE + Dice + clDice (topology-aware) loss.

Three prior interventions (Focal Tversky loss reweighting, a from-scratch
retrain with fixed training methodology, and flip-TTA at inference) all
landed at the same ~0.777-0.782 test Dice ceiling -- see
train_mobileunet_tversky.py and train_mobileunet_v2.py. That plateau across
different training recipes suggests the remaining error isn't a matter of
threshold/recall calibration on raw pixel overlap; it may instead be
topological -- predicted vessel segments that are fragmented/disconnected
along thin branches score fine on Dice/Tversky (same pixel count either
way) but would score worse on centerline connectivity specifically.

clDice (Shit et al., CVPR 2021, see losses.SoftClDiceLoss) adds a loss term
computed on a differentiable soft-skeleton of the prediction and target, so
it penalizes broken vessel centerlines in a way pixel-overlap losses can't
express. This script adds it on top of BCE + Dice (not replacing it) and
fine-tunes from the existing best checkpoint, using the same corrected
methodology (real val split, AdamW, warmup+cosine, norm-based grad clip) as
train_mobileunet_v2.py, so the only new variable is the loss.

Usage:
    python src/train_mobileunet_cldice.py --epochs 20 --device cuda
"""

import argparse
import logging
import math
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
from losses import SoftClDiceLoss
from model_lightweight import MobileUNetv3


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) /
                (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()


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


def build_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


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
    cldice = SoftClDiceLoss(iterations=args.cldice_iterations)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(warmup_steps, total_steps))

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

                loss = (bce(logits, masks)
                        + dice_loss(probs, masks)
                        + args.cldice_weight * cldice(probs, masks))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.grad_clip_norm)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{"loss (batch)": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
                pbar.update(images.shape[0])

        metrics = evaluate(net, val_loader, device)
        logging.info(
            f"Epoch {epoch + 1}: val Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f} "
            f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "last.pth"))
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "best.pth"))
            logging.info(f"New best val Dice {best_dice:.4f} -> saved {args.checkpoint_dir}/best.pth")

    logging.info(f"Training complete. Best val Dice: {best_dice:.4f}")


def get_args():
    p = argparse.ArgumentParser(description="Fine-tune MobileUNetv3 with BCE + Dice + clDice loss")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--init-checkpoint", default="checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth",
                   help="Warm-start weights; pass '' to train from an ImageNet-pretrained backbone instead")
    p.add_argument("--checkpoint-dir", default="checkpoints/mobileunetv3_cldice")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--grad-clip-norm", type=float, default=5.0)
    p.add_argument("--cldice-weight", type=float, default=0.5, help="Weight of the clDice term relative to BCE+Dice")
    p.add_argument("--cldice-iterations", type=int, default=10, help="Soft-skeletonization iterations")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
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
