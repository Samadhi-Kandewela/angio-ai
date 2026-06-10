"""
Improved multitask localization training script (v3).

Changes vs original:
  - Higher anatomy loss weight (2.0) to improve segment accuracy
  - Lower stenosis loss weight (0.8) to reduce false positives
  - Fixed augmentations in dataset_multitask.py (Affine + GaussNoise)
  - Added HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion
  - Default image size 640 for better thin-vessel resolution
  - Default lr 5e-5 (fine-tuning friendly)
  - --eval-only does not load training dataset

Usage (fine-tune from existing checkpoint):
    python src/train_localization_v3.py ^
        --device cuda --amp ^
        --data-dir dataset/arcade ^
        --resume-checkpoint checkpoints/multitask_localization_v2/multitask_best.pth ^
        --output-dir checkpoints/multitask_v3

Evaluate:
    python src/eval_localization.py ^
        --checkpoint checkpoints/multitask_v3/multitask_best.pth ^
        --data-dir dataset/arcade --device cuda
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


def focal_tversky_loss(logits, target, alpha=0.25, beta=0.75, gamma=0.75, smooth=1.0):
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    tp = (probs * target).sum(dim=dims)
    fp = (probs * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - probs) * target).sum(dim=dims)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return torch.pow(1.0 - tversky, gamma).mean()


def vessel_only_cross_entropy(logits, target, class_weights=None):
    valid = target > 0
    if valid.sum() == 0:
        return logits.sum() * 0.0
    per_pixel = F.cross_entropy(logits, target, weight=class_weights, reduction="none")
    return per_pixel[valid].mean()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def dice_score(logits, target, threshold=0.5, smooth=1.0):
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + smooth) / (union + smooth)).mean().item()


def soft_dice_score(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
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


def stenosis_precision_recall(logits, target, threshold=0.3):
    pred = torch.sigmoid(logits) > threshold
    truth = target > 0.5
    tp = (pred & truth).sum().float()
    fp = (pred & ~truth).sum().float()
    fn = (~pred & truth).sum().float()
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    return precision.item(), recall.item()


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


def make_stenosis_pos_weight(dataset, device, max_pos_weight=50.0):
    positives, negatives = dataset.estimate_stenosis_pixel_counts()
    if positives <= 0:
        logging.warning("No stenosis pixels found. Using pos_weight=1.")
        value = 1.0
    else:
        value = min(negatives / positives, max_pos_weight)
    logging.info("Stenosis pos_weight=%.2f (pos=%.0f neg=%.0f)", value, positives, negatives)
    return torch.tensor([value], dtype=torch.float32, device=device)


# ─── Loss computation ─────────────────────────────────────────────────────────

def compute_loss(outputs, batch, weights, loss_state):
    vessel_target = batch["vessel_mask"]
    anatomy_target = batch["anatomy_mask"]
    stenosis_target = batch["stenosis_mask"]

    vessel_bce = F.binary_cross_entropy_with_logits(outputs["vessel"], vessel_target)
    vessel_dice_l = dice_loss(outputs["vessel"], vessel_target)
    anatomy_ce = vessel_only_cross_entropy(
        outputs["anatomy"], anatomy_target,
        class_weights=loss_state.get("anatomy_class_weights"),
    )
    stenosis_bce = F.binary_cross_entropy_with_logits(
        outputs["stenosis"], stenosis_target,
        pos_weight=loss_state.get("stenosis_pos_weight"),
    )
    stenosis_tversky = focal_tversky_loss(
        outputs["stenosis"], stenosis_target,
        alpha=loss_state["tversky_alpha"],
        beta=loss_state["tversky_beta"],
        gamma=loss_state["tversky_gamma"],
    )

    total = (
        weights["vessel"] * (vessel_bce + vessel_dice_l)
        + weights["anatomy"] * anatomy_ce
        + weights["stenosis"] * (stenosis_bce + stenosis_tversky)
    )
    return total, {
        "vessel": (vessel_bce + vessel_dice_l).item(),
        "anatomy": anatomy_ce.item(),
        "stenosis": (stenosis_bce + stenosis_tversky).item(),
    }


def move_batch_to_device(batch, device):
    return {
        "image": batch["image"].to(device, dtype=torch.float32),
        "vessel_mask": batch["vessel_mask"].to(device, dtype=torch.float32),
        "anatomy_mask": batch["anatomy_mask"].to(device, dtype=torch.long),
        "stenosis_mask": batch["stenosis_mask"].to(device, dtype=torch.float32),
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
            with torch.cuda.amp.autocast():
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
            anatomy=f"{parts['anatomy']:.3f}",
            stenosis=f"{parts['stenosis']:.3f}",
        )
    return running / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device, weights, loss_state):
    model.eval()
    total_loss = vessel_dice = stenosis_dice = stenosis_soft = 0.0
    stenosis_prec = stenosis_rec = anat_acc = anat_grp = anat_art = 0.0

    for raw_batch in tqdm(dataloader, desc="Val", unit="batch", leave=True):
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch["image"])
        loss, _ = compute_loss(outputs, batch, weights, loss_state)
        prec, rec = stenosis_precision_recall(outputs["stenosis"], batch["stenosis_mask"])

        total_loss += loss.item()
        vessel_dice += dice_score(outputs["vessel"], batch["vessel_mask"])
        stenosis_dice += dice_score(outputs["stenosis"], batch["stenosis_mask"])
        stenosis_soft += soft_dice_score(outputs["stenosis"], batch["stenosis_mask"])
        stenosis_prec += prec
        stenosis_rec += rec
        anat_acc += anatomy_accuracy(outputs["anatomy"], batch["anatomy_mask"])
        anat_grp += mapped_accuracy(outputs["anatomy"], batch["anatomy_mask"], SEGMENT_TO_GROUP_ID, device)
        anat_art += mapped_accuracy(outputs["anatomy"], batch["anatomy_mask"], SEGMENT_TO_ARTERY_ID, device)

    n = max(len(dataloader), 1)
    return {
        "loss": total_loss / n,
        "vessel_dice": vessel_dice / n,
        "stenosis_dice": stenosis_dice / n,
        "stenosis_soft_dice": stenosis_soft / n,
        "stenosis_precision": stenosis_prec / n,
        "stenosis_recall": stenosis_rec / n,
        "anatomy_acc": anat_acc / n,
        "anatomy_group_acc": anat_grp / n,
        "anatomy_artery_acc": anat_art / n,
    }


def print_metrics(metrics):
    print(f"  vessel_dice={metrics['vessel_dice']:.4f} "
          f"anatomy_acc={metrics['anatomy_acc']:.4f} "
          f"anatomy_group={metrics['anatomy_group_acc']:.4f} "
          f"anatomy_artery={metrics['anatomy_artery_acc']:.4f} "
          f"stenosis_dice={metrics['stenosis_dice']:.4f} "
          f"precision={metrics['stenosis_precision']:.4f} "
          f"recall={metrics['stenosis_recall']:.4f}")


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

    weights = {"vessel": args.vessel_weight, "anatomy": args.anatomy_weight, "stenosis": args.stenosis_weight}
    loss_state = {
        "anatomy_class_weights": None,
        "stenosis_pos_weight": None,
        "tversky_alpha": args.tversky_alpha,
        "tversky_beta": args.tversky_beta,
        "tversky_gamma": args.tversky_gamma,
    }

    if args.eval_only:
        logging.info("Val images: %d", len(val_dataset))
        metrics = evaluate(model, val_loader, device, weights, loss_state)
        print("\n" + "=" * 50)
        print("LOCALIZATION MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"  Vessel Dice:            {metrics['vessel_dice']:.4f}")
        print(f"  Anatomy Acc (segment):  {metrics['anatomy_acc']:.4f}  <- 25 SYNTAX segments")
        print(f"  Anatomy Acc (group):    {metrics['anatomy_group_acc']:.4f}  <- e.g. LAD proximal")
        print(f"  Anatomy Acc (artery):   {metrics['anatomy_artery_acc']:.4f}  <- RCA / LAD / LCX / LM")
        print(f"  Stenosis Dice:          {metrics['stenosis_dice']:.4f}")
        print(f"  Stenosis Soft Dice:     {metrics['stenosis_soft_dice']:.4f}")
        print(f"  Stenosis Precision:     {metrics['stenosis_precision']:.4f}")
        print(f"  Stenosis Recall:        {metrics['stenosis_recall']:.4f}")
        print(f"  Val Loss:               {metrics['loss']:.4f}")
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
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    loss_state["anatomy_class_weights"] = make_anatomy_class_weights(
        train_dataset, NUM_ANATOMY_CLASSES, device, max_weight=args.max_anatomy_class_weight,
    )
    loss_state["stenosis_pos_weight"] = make_stenosis_pos_weight(
        train_dataset, device, max_pos_weight=args.max_stenosis_pos_weight,
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
            0.35 * metrics["anatomy_acc"]
            + 0.25 * metrics["anatomy_group_acc"]
            + 0.25 * metrics["vessel_dice"]
            + 0.10 * metrics["stenosis_recall"]
            + 0.05 * metrics["stenosis_soft_dice"]
        )

        logging.info("train_loss=%.4f val_loss=%.4f score=%.4f", train_loss, metrics["loss"], score)
        print_metrics(metrics)

        torch.save(model.state_dict(), os.path.join(args.output_dir, "multitask_latest.pth"))
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.output_dir, "multitask_best.pth"))
            logging.info("New best score=%.4f — checkpoint saved.", best_score)


def get_args():
    parser = argparse.ArgumentParser(description="Train improved multitask localization model (v3)")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="checkpoints/multitask_v3")
    parser.add_argument("--resume-checkpoint", default="", help="Continue from this .pth checkpoint")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained MobileNetV3 encoder")
    parser.add_argument("--init-segmentation-checkpoint", default="", help="Warm-start encoder+decoder from a segmentation model")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate checkpoint without training")
    parser.add_argument("--val-split", default="val", choices=["val", "test"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action="store_true", help="Mixed precision (CUDA only)")
    parser.add_argument("--vessel-weight", type=float, default=1.0)
    parser.add_argument("--anatomy-weight", type=float, default=2.0)
    parser.add_argument("--stenosis-weight", type=float, default=0.8)
    parser.add_argument("--max-anatomy-class-weight", type=float, default=8.0)
    parser.add_argument("--max-stenosis-pos-weight", type=float, default=50.0)
    parser.add_argument("--tversky-alpha", type=float, default=0.25)
    parser.add_argument("--tversky-beta", type=float, default=0.75)
    parser.add_argument("--tversky-gamma", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    try:
        train(args)
    except KeyboardInterrupt:
        logging.info("Training interrupted.")
        sys.exit(0)
