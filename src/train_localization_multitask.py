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
from localization_labels import NUM_ANATOMY_CLASSES
from model_multitask import MultiTaskMobileUNetv3


def dice_loss_from_logits(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.contiguous()
    target = target.contiguous()
    dims = (2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - ((2.0 * intersection + smooth) / (union + smooth))).mean()


def dice_score_from_logits(logits, target, threshold=0.5, smooth=1.0):
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + smooth) / (union + smooth)).mean().item()


def anatomy_accuracy(logits, target, vessel_only=True):
    pred = torch.argmax(logits, dim=1)
    valid = target > 0 if vessel_only else torch.ones_like(target, dtype=torch.bool)
    if valid.sum() == 0:
        return 0.0
    return (pred[valid] == target[valid]).float().mean().item()


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


def compute_loss(outputs, batch, weights):
    vessel_target = batch["vessel_mask"]
    anatomy_target = batch["anatomy_mask"]
    stenosis_target = batch["stenosis_mask"]

    vessel_bce = F.binary_cross_entropy_with_logits(outputs["vessel"], vessel_target)
    vessel_dice = dice_loss_from_logits(outputs["vessel"], vessel_target)
    anatomy_ce = F.cross_entropy(outputs["anatomy"], anatomy_target)
    stenosis_bce = F.binary_cross_entropy_with_logits(outputs["stenosis"], stenosis_target)
    stenosis_dice = dice_loss_from_logits(outputs["stenosis"], stenosis_target)

    total = (
        weights["vessel"] * (vessel_bce + vessel_dice)
        + weights["anatomy"] * anatomy_ce
        + weights["stenosis"] * (stenosis_bce + stenosis_dice)
    )

    return total, {
        "vessel": (vessel_bce + vessel_dice).item(),
        "anatomy": anatomy_ce.item(),
        "stenosis": (stenosis_bce + stenosis_dice).item(),
    }


def move_batch_to_device(batch, device):
    return {
        "image": batch["image"].to(device=device, dtype=torch.float32),
        "vessel_mask": batch["vessel_mask"].to(device=device, dtype=torch.float32),
        "anatomy_mask": batch["anatomy_mask"].to(device=device, dtype=torch.long),
        "stenosis_mask": batch["stenosis_mask"].to(device=device, dtype=torch.float32),
    }


def train_one_epoch(model, dataloader, optimizer, device, weights, scaler=None):
    model.train()
    running = 0.0

    pbar = tqdm(dataloader, desc="Train", unit="batch")
    for raw_batch in pbar:
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch["image"])
                loss, parts = compute_loss(outputs, batch, weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch["image"])
            loss, parts = compute_loss(outputs, batch, weights)
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
def evaluate(model, dataloader, device, weights):
    model.eval()
    total_loss = 0.0
    vessel_dice = 0.0
    stenosis_dice = 0.0
    anatomy_acc = 0.0

    for raw_batch in tqdm(dataloader, desc="Val", unit="batch", leave=False):
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch["image"])
        loss, _ = compute_loss(outputs, batch, weights)

        total_loss += loss.item()
        vessel_dice += dice_score_from_logits(outputs["vessel"], batch["vessel_mask"])
        stenosis_dice += dice_score_from_logits(outputs["stenosis"], batch["stenosis_mask"])
        anatomy_acc += anatomy_accuracy(outputs["anatomy"], batch["anatomy_mask"], vessel_only=True)

    n = max(len(dataloader), 1)
    return {
        "loss": total_loss / n,
        "vessel_dice": vessel_dice / n,
        "stenosis_dice": stenosis_dice / n,
        "anatomy_acc": anatomy_acc / n,
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
    load_segmentation_warm_start(model, args.init_segmentation_checkpoint, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    weights = {
        "vessel": args.vessel_weight,
        "anatomy": args.anatomy_weight,
        "stenosis": args.stenosis_weight,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    best_score = -1.0

    logging.info("Train images: %d | Val images: %d", len(train_dataset), len(val_dataset))
    logging.info("Saving checkpoints to: %s", args.output_dir)

    for epoch in range(1, args.epochs + 1):
        logging.info("Epoch %d/%d", epoch, args.epochs)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, weights, scaler)
        metrics = evaluate(model, val_loader, device, weights)
        scheduler.step()

        score = (
            0.35 * metrics["vessel_dice"]
            + 0.45 * metrics["anatomy_acc"]
            + 0.20 * metrics["stenosis_dice"]
        )

        logging.info(
            "train_loss=%.4f val_loss=%.4f vessel_dice=%.4f anatomy_acc=%.4f stenosis_dice=%.4f score=%.4f",
            train_loss,
            metrics["loss"],
            metrics["vessel_dice"],
            metrics["anatomy_acc"],
            metrics["stenosis_dice"],
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
    parser.add_argument("--vessel-weight", type=float, default=0.7)
    parser.add_argument("--anatomy-weight", type=float, default=1.0)
    parser.add_argument("--stenosis-weight", type=float, default=1.2)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    try:
        train(args)
    except KeyboardInterrupt:
        logging.info("Training interrupted.")
        sys.exit(0)
