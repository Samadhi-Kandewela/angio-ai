"""
Train the guided single-model localization (vessel segmentation + anatomy).

The vessel prediction masks the decoder features before the anatomy head,
so anatomy classification is guided by where vessels are detected.

Usage:
    python src/train_guided_localization.py `
        --device cuda --amp `
        --data-dir dataset `
        --pretrained `
        --num-workers 0 `
        --output-dir checkpoints/guided_localization

Evaluate:
    python src/eval_guided_localization.py `
        --checkpoint checkpoints/guided_localization/best.pth `
        --data-dir dataset --device cuda
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_guided_localization import GuidedLocalizationDataset
from localization_labels import (
    NUM_ANATOMY_CLASSES,
    SEGMENT_TO_ARTERY_ID,
    SEGMENT_TO_GROUP_ID,
)
from model_guided_localization import GuidedLocalizationNet


# ─── Loss functions ───────────────────────────────────────────────────────────

def dice_loss(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits).contiguous()
    target = target.contiguous()
    dims = (2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + smooth) / (union + smooth)).mean()


def vessel_only_cross_entropy(logits, target, class_weights=None):
    valid = target > 0
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits, target, weight=class_weights, reduction="none"
    )[valid].mean()


def compute_loss(outputs, batch, weights, class_weights=None):
    vessel_bce   = F.binary_cross_entropy_with_logits(
        outputs["vessel"], batch["vessel_mask"]
    )
    vessel_dice  = dice_loss(outputs["vessel"], batch["vessel_mask"])
    anatomy_ce   = vessel_only_cross_entropy(
        outputs["anatomy"], batch["anatomy_mask"], class_weights=class_weights
    )
    total = (
        weights["vessel"]  * (vessel_bce + vessel_dice)
        + weights["anatomy"] * anatomy_ce
    )
    return total, {
        "vessel":  (vessel_bce + vessel_dice).item(),
        "anatomy": anatomy_ce.item(),
    }


# ─── Metrics ──────────────────────────────────────────────────────────────────

def dice_score(logits, target, threshold=0.5, smooth=1.0):
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + smooth) / (union + smooth)).mean().item()


def anatomy_accuracy(logits, target):
    pred = torch.argmax(logits, dim=1)
    valid = target > 0
    if valid.sum() == 0:
        return 0.0
    return (pred[valid] == target[valid]).float().mean().item()


def mapped_accuracy(logits, target, mapping, device):
    pred = torch.argmax(logits, dim=1)
    valid = target > 0
    if valid.sum() == 0:
        return 0.0
    map_tensor = torch.as_tensor(mapping, device=device, dtype=torch.long)
    pred_mapped   = map_tensor[pred.clamp(0, len(mapping) - 1)]
    target_mapped = map_tensor[target.clamp(0, len(mapping) - 1)]
    return (pred_mapped[valid] == target_mapped[valid]).float().mean().item()


# ─── Class weights ────────────────────────────────────────────────────────────

def make_anatomy_class_weights(dataset, num_classes, device, max_weight=8.0):
    counts = dataset.estimate_anatomy_pixel_counts(num_classes=num_classes)
    vessel_counts = counts.copy()
    vessel_counts[0] = 0.0
    positive = vessel_counts[vessel_counts > 0]
    weights = torch.ones(num_classes, dtype=torch.float32)
    weights[0] = 0.0
    if len(positive) > 0:
        median = float(torch.tensor(positive).median().item())
        for c in range(1, num_classes):
            if vessel_counts[c] > 0:
                weights[c] = min(
                    (median / float(vessel_counts[c])) ** 0.5, max_weight
                )
    logging.info(
        "Class weights: %s",
        ", ".join(f"{i}:{weights[i]:.2f}" for i in range(num_classes)),
    )
    return weights.to(device)


# ─── Batch helper ─────────────────────────────────────────────────────────────

def move_batch(batch, device):
    return {
        "image":        batch["image"].to(device, dtype=torch.float32),
        "vessel_mask":  batch["vessel_mask"].to(device, dtype=torch.float32),
        "anatomy_mask": batch["anatomy_mask"].to(device, dtype=torch.long),
    }


# ─── Train / eval loops ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, weights, class_weights, scaler=None):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc="Train", unit="batch")
    for raw_batch in pbar:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(batch["image"])
                loss, parts = compute_loss(outputs, batch, weights, class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch["image"])
            loss, parts = compute_loss(outputs, batch, weights, class_weights)
            loss.backward()
            optimizer.step()
        running += loss.item()
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            vessel=f"{parts['vessel']:.3f}",
            anatomy=f"{parts['anatomy']:.3f}",
        )
    return running / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, weights, class_weights=None):
    model.eval()
    total_loss = vessel_dice_total = anat_acc = anat_grp = anat_art = 0.0
    for raw_batch in tqdm(loader, desc="Val", unit="batch", leave=True):
        batch = move_batch(raw_batch, device)
        outputs = model(batch["image"])
        loss, _ = compute_loss(outputs, batch, weights, class_weights)
        total_loss      += loss.item()
        vessel_dice_total += dice_score(outputs["vessel"], batch["vessel_mask"])
        anat_acc        += anatomy_accuracy(outputs["anatomy"], batch["anatomy_mask"])
        anat_grp        += mapped_accuracy(outputs["anatomy"], batch["anatomy_mask"], SEGMENT_TO_GROUP_ID, device)
        anat_art        += mapped_accuracy(outputs["anatomy"], batch["anatomy_mask"], SEGMENT_TO_ARTERY_ID, device)
    n = max(len(loader), 1)
    return {
        "loss":               total_loss / n,
        "vessel_dice":        vessel_dice_total / n,
        "anatomy_acc":        anat_acc / n,
        "anatomy_group_acc":  anat_grp / n,
        "anatomy_artery_acc": anat_art / n,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Device: %s", device)

    val_dataset = GuidedLocalizationDataset(
        args.data_dir, split="val",
        image_size=(args.image_size, args.image_size), augment=False,
    )
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)

    model = GuidedLocalizationNet(
        n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=args.pretrained
    ).to(device)

    if args.resume_checkpoint:
        model.load_state_dict(
            torch.load(args.resume_checkpoint, map_location=device)
        )
        logging.info("Resumed from: %s", args.resume_checkpoint)

    weights = {"vessel": args.vessel_weight, "anatomy": args.anatomy_weight}

    if args.eval_only:
        metrics = evaluate(model, val_loader, device, weights)
        _print_metrics(metrics)
        return

    train_dataset = GuidedLocalizationDataset(
        args.data_dir, split="train",
        image_size=(args.image_size, args.image_size), augment=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_args)

    class_weights = make_anatomy_class_weights(
        train_dataset, NUM_ANATOMY_CLASSES, device, max_weight=args.max_class_weight
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    os.makedirs(args.output_dir, exist_ok=True)
    best_score = -1.0
    logging.info(
        "Train: %d | Val: %d | Output: %s",
        len(train_dataset), len(val_dataset), args.output_dir,
    )

    for epoch in range(1, args.epochs + 1):
        logging.info("Epoch %d/%d", epoch, args.epochs)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, weights, class_weights, scaler
        )
        metrics = evaluate(model, val_loader, device, weights, class_weights)
        scheduler.step()

        score = (
            0.50 * metrics["anatomy_acc"]
            + 0.30 * metrics["anatomy_group_acc"]
            + 0.20 * metrics["vessel_dice"]
        )

        logging.info(
            "train_loss=%.4f val_loss=%.4f score=%.4f",
            train_loss, metrics["loss"], score,
        )
        print(
            f"  vessel_dice={metrics['vessel_dice']:.4f}  "
            f"anatomy_acc={metrics['anatomy_acc']:.4f}  "
            f"group={metrics['anatomy_group_acc']:.4f}  "
            f"artery={metrics['anatomy_artery_acc']:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(args.output_dir, "latest.pth"))
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pth"))
            logging.info("New best score=%.4f — saved.", best_score)


def _print_metrics(metrics):
    print("\n" + "=" * 50)
    print("GUIDED LOCALIZATION EVALUATION")
    print("=" * 50)
    print(f"  Val Loss:               {metrics['loss']:.4f}")
    print(f"  Vessel Dice:            {metrics['vessel_dice']:.4f}")
    print(f"  Anatomy Acc (segment):  {metrics['anatomy_acc']:.4f}  <- 25 SYNTAX segments")
    print(f"  Anatomy Acc (group):    {metrics['anatomy_group_acc']:.4f}  <- e.g. LAD proximal")
    print(f"  Anatomy Acc (artery):   {metrics['anatomy_artery_acc']:.4f}  <- RCA / LAD / LCX / LM")
    print("=" * 50)


def get_args():
    p = argparse.ArgumentParser(description="Train guided localization model")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--output-dir", default="checkpoints/guided_localization")
    p.add_argument("--resume-checkpoint", default="")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--vessel-weight", type=float, default=1.0)
    p.add_argument("--anatomy-weight", type=float, default=3.0)
    p.add_argument("--max-class-weight", type=float, default=8.0)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    try:
        train(args)
    except KeyboardInterrupt:
        logging.info("Interrupted.")
        sys.exit(0)
