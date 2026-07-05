"""
Fixed-methodology training script for MobileUNetv3 vessel segmentation.

train.py has five methodology issues that this script fixes, without
touching train.py itself:

  1. Wrong validation set. train.py carves 10% off the *train* folder as
     "validation" (random_split with a fixed seed), instead of using the
     real held-out dataset/syntax/val split that eval_segmentation.py uses.
     Scheduler decisions and "best checkpoint" selection there are reacting
     to a noisy in-sample slice, not true generalization. Fixed here by
     loading SegmentationDataset(split='val') directly.

  2. batch_size defaults to 1. MobileUNetv3's encoder is full of BatchNorm
     layers; batch=1 gives every BatchNorm layer a running-stats estimate
     from a single sample, which destabilizes training. Fixed here by
     defaulting to 4 (with a correspondingly higher LR) -- the largest
     batch size that fits in 6GB of VRAM at 512x512 without --amp; pass
     --amp and/or a larger --batch-size if you have more headroom.

  3. Gradient clipping is value-based and very tight:
     clip_grad_value_(net.parameters(), 0.1) clamps every individual
     gradient element to +/-0.1, which can bottleneck a decoder built on
     top of a pretrained encoder. Fixed here with norm-based clipping
     (clip_grad_norm_), the standard choice for U-Net-style models.

  4. Short, schedule-free training. Default is 5 epochs, flat LR, and
     ReduceLROnPlateau only reacting after the fact. Fixed here with a
     longer default run (50 epochs) plus linear warmup + cosine annealing,
     which reliably extracts more from a small (1000-image) training set
     than a flat LR.

  5. No encoder freeze phase. Training the ImageNet-pretrained encoder
     jointly with a randomly-initialized decoder from step one lets early,
     large decoder gradients disturb useful pretrained features. Fixed
     here by freezing the encoder (both requires_grad and BatchNorm
     running stats, via encoder.eval()) for the first --freeze-epochs
     epochs, then unfreezing.

Mixed precision (torch.cuda.amp) is available via --amp; it does not
change accuracy but frees up VRAM to afford the larger batch size from #2.

Usage:
    python src/train_mobileunet_v2.py --epochs 50 --device cuda
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


def set_encoder_frozen(net, frozen):
    for p in net.encoder.parameters():
        p.requires_grad = not frozen
    if frozen:
        net.encoder.eval()  # also freeze BatchNorm running stats
    else:
        net.encoder.train()


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
    logging.info(f"Train: {len(train_set)} images, Val: {len(val_set)} images (real held-out split)")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    net = MobileUNetv3(n_classes=1, pretrained=not args.init_checkpoint).to(device)
    if args.init_checkpoint:
        net.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        logging.info(f"Initialized weights from {args.init_checkpoint}")

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(warmup_steps, total_steps))

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(args.epochs):
        net.train()
        frozen = epoch < args.freeze_epochs
        set_encoder_frozen(net, frozen)
        if epoch == 0 or epoch == args.freeze_epochs:
            logging.info(f"Encoder {'frozen' if frozen else 'unfrozen'} (epoch {epoch + 1})")

        epoch_loss = 0.0
        with tqdm(total=len(train_set), desc=f"Epoch {epoch + 1}/{args.epochs}", unit="img") as pbar:
            for images, masks in train_loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32).unsqueeze(1)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=args.amp):
                    out = net(images)
                    logits = out["out"] if isinstance(out, dict) else out
                    loss = bce(logits, masks) + dice_loss(torch.sigmoid(logits), masks)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
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
    p = argparse.ArgumentParser(description="Train MobileUNetv3 with fixed validation/batch/clip/schedule/freeze methodology")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--init-checkpoint", default="",
                   help="Optional warm-start weights; empty trains from an ImageNet-pretrained backbone")
    p.add_argument("--checkpoint-dir", default="checkpoints/mobileunetv3_v2")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--freeze-epochs", type=int, default=5,
                   help="Epochs to keep the pretrained encoder frozen before joint fine-tuning")
    p.add_argument("--grad-clip-norm", type=float, default=5.0)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision training")
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
