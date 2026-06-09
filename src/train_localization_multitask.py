import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_multitask import CoronaryMultiTaskDataset
from localization_labels import (
    NUM_ANATOMY_CLASSES,
    SEGMENT_TO_ARTERY_ID,
    SEGMENT_TO_GROUP_ID,
)
from model_multitask import MultiTaskMobileUNetv3


def dice_loss_from_logits(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.contiguous()
    target = target.contiguous()
    dims = (2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - ((2.0 * intersection + smooth) / (union + smooth))).mean()


def focal_tversky_loss_from_logits(logits, target, alpha=0.25, beta=0.75, gamma=0.75, smooth=1.0):
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    true_pos = (probs * target).sum(dim=dims)
    false_pos = (probs * (1.0 - target)).sum(dim=dims)
    false_neg = ((1.0 - probs) * target).sum(dim=dims)
    tversky = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    return torch.pow(1.0 - tversky, gamma).mean()


def dice_score_from_logits(logits, target, threshold=0.5, smooth=1.0):
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + smooth) / (union + smooth)).mean().item()


def soft_dice_score_from_logits(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + smooth) / (union + smooth)).mean().item()


def anatomy_accuracy(logits, target, vessel_only=True):
    pred = torch.argmax(logits, dim=1)
    valid = target > 0 if vessel_only else torch.ones_like(target, dtype=torch.bool)
    if valid.sum() == 0:
        return 0.0
    return (pred[valid] == target[valid]).float().mean().item()


def mapped_anatomy_accuracy(logits, target, mapping, device):
    pred = torch.argmax(logits, dim=1)
    valid = target > 0
    if valid.sum() == 0:
        return 0.0
    map_tensor = torch.as_tensor(mapping, device=device, dtype=torch.long)
    pred_mapped = map_tensor[pred.clamp(min=0, max=len(mapping) - 1)]
    target_mapped = map_tensor[target.clamp(min=0, max=len(mapping) - 1)]
    return (pred_mapped[valid] == target_mapped[valid]).float().mean().item()


def stenosis_detection_stats(logits, target, threshold=0.3):
    pred = torch.sigmoid(logits) > threshold
    truth = target > 0.5
    true_pos = (pred & truth).sum().float()
    false_pos = (pred & ~truth).sum().float()
    false_neg = (~pred & truth).sum().float()
    precision = true_pos / torch.clamp(true_pos + false_pos, min=1.0)
    recall = true_pos / torch.clamp(true_pos + false_neg, min=1.0)
    return precision.item(), recall.item()


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
                # Square-root inverse frequency is gentler than full inverse frequency.
                weights[class_id] = min((median / float(vessel_counts[class_id])) ** 0.5, max_weight)
    logging.info(
        "Anatomy class weights: %s",
        ", ".join(f"{idx}:{weights[idx]:.2f}" for idx in range(num_classes)),
    )
    return weights.to(device)


def make_stenosis_pos_weight(dataset, device, max_pos_weight=50.0):
    positives, negatives = dataset.estimate_stenosis_pixel_counts()
    if positives <= 0:
        logging.warning("No stenosis pixels found in training set. Using pos_weight=1.")
        value = 1.0
    else:
        value = min(negatives / positives, max_pos_weight)
    logging.info(
        "Stenosis pixels: positive=%.0f negative=%.0f pos_weight=%.2f",
        positives,
        negatives,
        value,
    )
    return torch.tensor([value], dtype=torch.float32, device=device)


def vessel_only_cross_entropy(logits, target, class_weights=None):
    valid = target > 0
    if valid.sum() == 0:
        return logits.sum() * 0.0
    per_pixel = F.cross_entropy(logits, target, weight=class_weights, reduction="none")
    return per_pixel[valid].mean()


def load_segmentation_warm_start(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    model_state.update(compatible)
    model.load_state_dict(model_state)
    logging.info("Warm-start loaded %d tensors from %s", len(compatible), checkpoint_path)
    if skipped:
        logging.info("Skipped %d incompatible tensors, usually the old final head.", len(skipped))


def load_multitask_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Multitask checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    logging.info("Loaded multitask checkpoint: %s", checkpoint_path)


def compute_loss(outputs, batch, weights, loss_state):
    vessel_target = batch["vessel_mask"]
    anatomy_target = batch["anatomy_mask"]
    stenosis_target = batch["stenosis_mask"]

    vessel_bce = F.binary_cross_entropy_with_logits(outputs["vessel"], vessel_target)
    vessel_dice = dice_loss_from_logits(outputs["vessel"], vessel_target)
    anatomy_ce = vessel_only_cross_entropy(
        outputs["anatomy"],
        anatomy_target,
        class_weights=loss_state.get("anatomy_class_weights"),
    )
    stenosis_bce = F.binary_cross_entropy_with_logits(
        outputs["stenosis"],
        stenosis_target,
        pos_weight=loss_state.get("stenosis_pos_weight"),
    )
    stenosis_tversky = focal_tversky_loss_from_logits(
        outputs["stenosis"],
        stenosis_target,
        alpha=loss_state["tversky_alpha"],
        beta=loss_state["tversky_beta"],
        gamma=loss_state["tversky_gamma"],
    )

    total = (
        weights["vessel"] * (vessel_bce + vessel_dice)
        + weights["anatomy"] * anatomy_ce
        + weights["stenosis"] * (stenosis_bce + stenosis_tversky)
    )

    return total, {
        "vessel": (vessel_bce + vessel_dice).item(),
        "anatomy": anatomy_ce.item(),
        "stenosis": (stenosis_bce + stenosis_tversky).item(),
    }


def move_batch_to_device(batch, device):
    return {
        "image": batch["image"].to(device=device, dtype=torch.float32),
        "vessel_mask": batch["vessel_mask"].to(device=device, dtype=torch.float32),
        "anatomy_mask": batch["anatomy_mask"].to(device=device, dtype=torch.long),
        "stenosis_mask": batch["stenosis_mask"].to(device=device, dtype=torch.float32),
    }


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
    total_loss = 0.0
    vessel_dice = 0.0
    stenosis_dice = 0.0
    stenosis_soft_dice = 0.0
    stenosis_precision = 0.0
    stenosis_recall = 0.0
    anatomy_acc = 0.0
    anatomy_group_acc = 0.0
    anatomy_artery_acc = 0.0

    for raw_batch in tqdm(dataloader, desc="Val", unit="batch", leave=False):
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch["image"])
        loss, _ = compute_loss(outputs, batch, weights, loss_state)
        precision, recall = stenosis_detection_stats(outputs["stenosis"], batch["stenosis_mask"])

        total_loss += loss.item()
        vessel_dice += dice_score_from_logits(outputs["vessel"], batch["vessel_mask"])
        stenosis_dice += dice_score_from_logits(outputs["stenosis"], batch["stenosis_mask"])
        stenosis_soft_dice += soft_dice_score_from_logits(outputs["stenosis"], batch["stenosis_mask"])
        stenosis_precision += precision
        stenosis_recall += recall
        anatomy_acc += anatomy_accuracy(outputs["anatomy"], batch["anatomy_mask"], vessel_only=True)
        anatomy_group_acc += mapped_anatomy_accuracy(outputs["anatomy"], batch["anatomy_mask"], SEGMENT_TO_GROUP_ID, device)
        anatomy_artery_acc += mapped_anatomy_accuracy(outputs["anatomy"], batch["anatomy_mask"], SEGMENT_TO_ARTERY_ID, device)

    n = max(len(dataloader), 1)
    return {
        "loss": total_loss / n,
        "vessel_dice": vessel_dice / n,
        "stenosis_dice": stenosis_dice / n,
        "stenosis_soft_dice": stenosis_soft_dice / n,
        "stenosis_precision": stenosis_precision / n,
        "stenosis_recall": stenosis_recall / n,
        "anatomy_acc": anatomy_acc / n,
        "anatomy_group_acc": anatomy_group_acc / n,
        "anatomy_artery_acc": anatomy_artery_acc / n,
    }


def train(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Using device: %s", device)

    train_dataset = CoronaryMultiTaskDataset(
        args.data_dir,
        split="train",
        image_size=(args.image_size, args.image_size),
        augment=True,
    )
    val_dataset = CoronaryMultiTaskDataset(
        args.data_dir,
        split=args.val_split,
        image_size=(args.image_size, args.image_size),
        augment=False,
    )

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)

    model = MultiTaskMobileUNetv3(
        n_anatomy_classes=NUM_ANATOMY_CLASSES,
        pretrained=args.pretrained,
    ).to(device)
    if args.resume_checkpoint:
        load_multitask_checkpoint(model, args.resume_checkpoint, device)
    else:
        load_segmentation_warm_start(model, args.init_segmentation_checkpoint, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    weights = {
        "vessel": args.vessel_weight,
        "anatomy": args.anatomy_weight,
        "stenosis": args.stenosis_weight,
    }
    loss_state = {
        "anatomy_class_weights": make_anatomy_class_weights(
            train_dataset,
            NUM_ANATOMY_CLASSES,
            device,
            max_weight=args.max_anatomy_class_weight,
        ) if args.class_balanced_anatomy else None,
        "stenosis_pos_weight": make_stenosis_pos_weight(
            train_dataset,
            device,
            max_pos_weight=args.max_stenosis_pos_weight,
        ) if args.weighted_stenosis_bce else None,
        "tversky_alpha": args.tversky_alpha,
        "tversky_beta": args.tversky_beta,
        "tversky_gamma": args.tversky_gamma,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    best_score = -1.0

    logging.info("Train images: %d | Val images: %d", len(train_dataset), len(val_dataset))
    logging.info("Saving checkpoints to: %s", args.output_dir)

    if args.eval_only:
        metrics = evaluate(model, val_loader, device, weights, loss_state)
        logging.info(
            (
                "eval_only val_loss=%.4f vessel_dice=%.4f "
                "anatomy_acc=%.4f anatomy_group_acc=%.4f anatomy_artery_acc=%.4f "
                "stenosis_dice=%.4f stenosis_soft_dice=%.4f "
                "stenosis_precision=%.4f stenosis_recall=%.4f"
            ),
            metrics["loss"],
            metrics["vessel_dice"],
            metrics["anatomy_acc"],
            metrics["anatomy_group_acc"],
            metrics["anatomy_artery_acc"],
            metrics["stenosis_dice"],
            metrics["stenosis_soft_dice"],
            metrics["stenosis_precision"],
            metrics["stenosis_recall"],
        )
        return

    for epoch in range(1, args.epochs + 1):
        logging.info("Epoch %d/%d", epoch, args.epochs)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, weights, loss_state, scaler)
        metrics = evaluate(model, val_loader, device, weights, loss_state)
        scheduler.step()

        score = (
            0.25 * metrics["vessel_dice"]
            + 0.25 * metrics["anatomy_acc"]
            + 0.25 * metrics["anatomy_group_acc"]
            + 0.15 * metrics["stenosis_recall"]
            + 0.10 * metrics["stenosis_soft_dice"]
        )

        logging.info(
            (
                "train_loss=%.4f val_loss=%.4f vessel_dice=%.4f "
                "anatomy_acc=%.4f anatomy_group_acc=%.4f anatomy_artery_acc=%.4f "
                "stenosis_dice=%.4f stenosis_soft_dice=%.4f "
                "stenosis_precision=%.4f stenosis_recall=%.4f score=%.4f"
            ),
            train_loss,
            metrics["loss"],
            metrics["vessel_dice"],
            metrics["anatomy_acc"],
            metrics["anatomy_group_acc"],
            metrics["anatomy_artery_acc"],
            metrics["stenosis_dice"],
            metrics["stenosis_soft_dice"],
            metrics["stenosis_precision"],
            metrics["stenosis_recall"],
            score,
        )

        latest_path = os.path.join(args.output_dir, "multitask_latest.pth")
        torch.save(model.state_dict(), latest_path)

        if score > best_score:
            best_score = score
            best_path = os.path.join(args.output_dir, "multitask_best.pth")
            torch.save(model.state_dict(), best_path)
            logging.info("New best checkpoint saved: %s", best_path)


def get_args():
    parser = argparse.ArgumentParser(description="Train real-time coronary segmentation + localization model")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints/multitask_localization")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--val-split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained MobileNetV3 encoder")
    parser.add_argument("--init-segmentation-checkpoint", type=str, default="")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Resume/evaluate a multitask checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the loaded checkpoint")
    parser.add_argument("--vessel-weight", type=float, default=0.7)
    parser.add_argument("--anatomy-weight", type=float, default=1.4)
    parser.add_argument("--stenosis-weight", type=float, default=1.6)
    parser.add_argument("--class-balanced-anatomy", action="store_true", default=True)
    parser.add_argument("--no-class-balanced-anatomy", action="store_false", dest="class_balanced_anatomy")
    parser.add_argument("--max-anatomy-class-weight", type=float, default=8.0)
    parser.add_argument("--weighted-stenosis-bce", action="store_true", default=True)
    parser.add_argument("--no-weighted-stenosis-bce", action="store_false", dest="weighted_stenosis_bce")
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