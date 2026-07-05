"""
Train MobileUNetv3ASPP (model_mobileunet_aspp.py) -- the ASPP-bottleneck
architecture change -- so its test-split Dice/IoU can be compared directly
against the plain MobileUNetv3 baseline and against
train_mobileunet_v2.py's from-scratch run (same training methodology,
different architecture).

Uses the same fixed methodology as train_mobileunet_v2.py (real
dataset/syntax/val split for validation/checkpoint selection, batch size 4,
AdamW, linear warmup + cosine annealing, norm-based grad clipping, encoder
frozen for the first few epochs then unfrozen) so architecture is the only
variable that differs from that run. Trains from scratch -- the ASPP
bottleneck changes the first decoder conv's input channel count, so the
existing mobileunetv3_augmented_best.pth checkpoint is not compatible as a
warm start.

Does not modify train.py, train_mobileunet_v2.py, model_lightweight.py, or
any existing checkpoint.

Usage:
    python src/train_mobileunet_aspp.py --epochs 50 --device cuda
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
from model_mobileunet_aspp import MobileUNetv3ASPP


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

    net = MobileUNetv3ASPP(n_classes=1, pretrained=not args.init_checkpoint,
                            aspp_out_channels=args.aspp_channels).to(device)
    if args.init_checkpoint:
        net.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        logging.info(f"Initialized weights from {args.init_checkpoint}")

    n_params = sum(p.numel() for p in net.parameters())
    logging.info(f"Model params: {n_params / 1e6:.2f}M")

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(warmup_steps, total_steps))

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

                out = net(images)
                logits = out["out"] if isinstance(out, dict) else out
                loss = bce(logits, masks) + dice_loss(torch.sigmoid(logits), masks)

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
    p = argparse.ArgumentParser(description="Train MobileUNetv3ASPP with the fixed training methodology")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--init-checkpoint", default="",
                   help="Optional warm-start weights (must match aspp_out_channels); empty trains from scratch")
    p.add_argument("--checkpoint-dir", default="checkpoints/mobileunetv3_aspp")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--freeze-epochs", type=int, default=5,
                   help="Epochs to keep the pretrained encoder frozen before joint fine-tuning")
    p.add_argument("--grad-clip-norm", type=float, default=5.0)
    p.add_argument("--aspp-channels", type=int, default=256, help="ASPP branch/output channel count")
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
