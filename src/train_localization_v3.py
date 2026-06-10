"""
Improved multitask localization training script (v3).

Trains vessel segmentation + anatomy localization only.
Stenosis is handled by QCA so it is completely excluded from training.

Usage:
    python src/train_localization_v3.py ^
        --device cuda --amp ^
        --data-dir dataset ^
        --init-segmentation-checkpoint checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth ^
        --output-dir checkpoints/multitask_v3

Evaluate:
    python src/eval_localization.py ^
        --checkpoint checkpoints/multitask_v3/multitask_best.pth ^
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

from dataset_multitask import CoronaryMultiTaskDataset
from localization_labels import (
    NUM_ANATOMY_CLASSES,
    SEGMENT_TO_ARTERY_ID,
    SEGMENT_TO_GROUP_ID,
)
from model_multitask import MultiTaskMobileUNetv3


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
    per_pixel = F.cross_entropy(logits, target, weight=class_weights, reduction="none")
    return per_pixel[valid].mean()


def compute_loss(outputs, batch, weights, loss_state):
    vessel_bce = F.binary_cross_entropy_with_logits(outputs["vessel"], batch["vessel_mask"])
    vessel_dice_l = dice_loss(outputs["vessel"], batch["vessel_mask"])
    anatomy_ce = vessel_only_cross_entropy(
        outputs["anatomy"], batch["anatomy_mask"],
        class_weights=loss_state.get("anatomy_class_weights"),
    )
    total = (
        weights["vessel"] * (vessel_bce + vessel_dice_l)
        + weights["anatomy"] * anatomy_ce
    )
    return total, {
        "vessel": (vessel_bce + vessel_dice_l).item(),
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
    pred_mapped = map_tensor[pred.clamp(0, len(mapping) - 1)]
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
        for class_id in range(1, num_classes):
            if vessel_counts[class_id] > 0:
                weights[class_id] = min((median / float(vessel_counts[class_id])) ** 0.5, max_weight)
    logging.info(
        "Anatomy class weights: %s",
        ", ".join(f"{i}:{weights[i]:.2f}" for i in range(num_classes)),
    )
    return weights.to(device)


# ─── Batch helpers ────────────────────────────────────────────────────────────

def move_batch_to_device(batch, device):
    return {
        "image": batch["image"].to(device, dtype=torch.float32),
        "vessel_mask": batch["vessel_mask"].to(device, dtype=torch.float32),
        "anatomy_mask": batch["anatomy_mask"].to(device, dtype=torch.long),
    }


# ─── Train / eval loops ───────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, device, weights, loss_state, scaler=None):
    model.train()
    running = 0.0
    pbar = tqdm(dataloader, desc="Train", unit="batch")
    for raw_batch in pbar:
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(batch["image"])
                loss, parts = compute_loss(outputs, batch, weights, loss_state)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch["image"])
            loss, parts = compute_loss(outputs, batch, weights, loss_state)
            loss.backward()
            optimizer.step()
        running += loss.item()
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            vessel=f"{parts['vessel']:.3f}",
            anatomy=f"{parts['anatomy']:.3f}",
        )
    return running / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device, weights, loss_state):
    model.eval()
    total_loss = vessel_dice_total = anat_acc = anat_grp = anat_art = 0.0

    for raw_batch in tqdm(dataloader, desc="Val", unit="batch", leave=True):
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch["image"])
        loss, _ = compute_loss(outputs, batch, weights, loss_state)

        total_loss += loss.item()
        vessel_dice_total += dice_score(outputs["vessel"], batch["vessel_mask"])
        anat_acc += anatomy_accuracy(outputs["anatomy"], batch["anatomy_mask"])
        anat_grp += mapped_accuracy(outputs["anatomy"], batch["anatomy_mask"], SEGMENT_TO_GROUP_ID, device)
        anat_art += mapped_accuracy(outputs["anatomy"], batch["anatomy_mask"], SEGMENT_TO_ARTERY_ID, device)

    n = max(len(dataloader), 1)
    return {
        "loss": total_loss / n,
        "vessel_dice": vessel_dice_total / n,
        "anatomy_acc": anat_acc / n,
        "anatomy_group_acc": anat_grp / n,
        "anatomy_artery_acc": anat_art / n,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Using device: %s", device)

    val_dataset = CoronaryMultiTaskDataset(
        args.data_dir, split=args.val_split,
        image_size=(args.image_size, args.image_size), augment=False,
    )
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)

    model = MultiTaskMobileUNetv3(n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=args.pretrained).to(device)
    if args.resume_checkpoint:
        state = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(state)
        logging.info("Loaded multitask checkpoint: %s", args.resume_checkpoint)
    elif args.init_segmentation_checkpoint:
        state = torch.load(args.init_segmentation_checkpoint, map_location=device)
        model_state = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
        skipped = [k for k in state if k not in compatible]
        model_state.update(compatible)
        model.load_state_dict(model_state)
        logging.info("Warm-start: loaded %d tensors from %s", len(compatible), args.init_segmentation_checkpoint)
        if skipped:
            logging.info("Skipped %d incompatible layers (final heads).", len(skipped))

    weights = {"vessel": args.vessel_weight, "anatomy": args.anatomy_weight}
    loss_state = {"anatomy_class_weights": None}

    if args.eval_only:
        logging.info("Val images: %d", len(val_dataset))
        metrics = evaluate(model, val_loader, device, weights, loss_state)
        print("\n" + "=" * 50)
        print("LOCALIZATION MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"  Val Loss:               {metrics['loss']:.4f}")
        print(f"  Vessel Dice:            {metrics['vessel_dice']:.4f}")
        print(f"  Anatomy Acc (segment):  {metrics['anatomy_acc']:.4f}  <- 25 SYNTAX segments")
        print(f"  Anatomy Acc (group):    {metrics['anatomy_group_acc']:.4f}  <- e.g. LAD proximal")
        print(f"  Anatomy Acc (artery):   {metrics['anatomy_artery_acc']:.4f}  <- RCA / LAD / LCX / LM")
        print("=" * 50)
        return

    # ── Training ──
    train_dataset = CoronaryMultiTaskDataset(
        args.data_dir, split="train",
        image_size=(args.image_size, args.image_size), augment=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_args)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    loss_state["anatomy_class_weights"] = make_anatomy_class_weights(
        train_dataset, NUM_ANATOMY_CLASSES, device, max_weight=args.max_anatomy_class_weight,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_score = -1.0

    logging.info("Train: %d | Val: %d | Output: %s", len(train_dataset), len(val_dataset), args.output_dir)

    for epoch in range(1, args.epochs + 1):
        logging.info("Epoch %d/%d", epoch, args.epochs)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, weights, loss_state, scaler)
        metrics = evaluate(model, val_loader, device, weights, loss_state)
        scheduler.step()

        score = (
            0.50 * metrics["anatomy_acc"]
            + 0.30 * metrics["anatomy_group_acc"]
            + 0.20 * metrics["vessel_dice"]
        )

        logging.info("train_loss=%.4f val_loss=%.4f score=%.4f", train_loss, metrics["loss"], score)
        print(f"  vessel_dice={metrics['vessel_dice']:.4f} "
              f"anatomy_acc={metrics['anatomy_acc']:.4f} "
              f"anatomy_group={metrics['anatomy_group_acc']:.4f} "
              f"anatomy_artery={metrics['anatomy_artery_acc']:.4f}")

        torch.save(model.state_dict(), os.path.join(args.output_dir, "multitask_latest.pth"))
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.output_dir, "multitask_best.pth"))
            logging.info("New best score=%.4f — checkpoint saved.", best_score)


def get_args():
    parser = argparse.ArgumentParser(description="Train localization model — vessel + anatomy only")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="checkpoints/multitask_v3")
    parser.add_argument("--resume-checkpoint", default="", help="Continue from this .pth checkpoint")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained MobileNetV3 encoder")
    parser.add_argument("--init-segmentation-checkpoint", default="", help="Warm-start from a segmentation model")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate checkpoint without training")
    parser.add_argument("--val-split", default="val", choices=["val", "test"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action="store_true", help="Mixed precision (CUDA only)")
    parser.add_argument("--vessel-weight", type=float, default=1.0)
    parser.add_argument("--anatomy-weight", type=float, default=3.0)
    parser.add_argument("--max-anatomy-class-weight", type=float, default=8.0)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    try:
        train(args)
    except KeyboardInterrupt:
        logging.info("Training interrupted.")
        sys.exit(0)
