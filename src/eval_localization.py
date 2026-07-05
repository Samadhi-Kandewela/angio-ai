"""
Evaluate the multitask localization model on the val or test split.

Usage:
    python src/eval_localization.py --checkpoint checkpoints/multitask_localization_v2/multitask_best.pth --data-dir dataset/arcade
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def dice_loss_from_logits(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits).contiguous()
    target = target.contiguous()
    dims = (2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + smooth) / (union + smooth)).mean()


def vessel_only_cross_entropy(logits, target):
    valid = target > 0
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits, target, reduction="none")[valid].mean()


def compute_loss(outputs, batch):
    vessel_bce = F.binary_cross_entropy_with_logits(outputs["vessel"], batch["vessel_mask"])
    vessel_dice = dice_loss_from_logits(outputs["vessel"], batch["vessel_mask"])
    anatomy_ce = vessel_only_cross_entropy(outputs["anatomy"], batch["anatomy_mask"])
    total = (
        0.7 * (vessel_bce + vessel_dice)
        + 1.4 * anatomy_ce
    )
    return total.item()

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_multitask import CoronaryMultiTaskDataset
from localization_labels import NUM_ANATOMY_CLASSES, SEGMENT_TO_ARTERY_ID, SEGMENT_TO_GROUP_ID
from model_multitask import MultiTaskMobileUNetv3


# ─── Metrics ──────────────────────────────────────────────────────────────────

def dice_score(logits, target, threshold=0.5, smooth=1.0):
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


def mapped_accuracy(logits, target, mapping, device):
    pred = torch.argmax(logits, dim=1)
    valid = target > 0
    if valid.sum() == 0:
        return 0.0
    map_tensor = torch.as_tensor(mapping, device=device, dtype=torch.long)
    pred_mapped = map_tensor[pred.clamp(0, len(mapping) - 1)]
    target_mapped = map_tensor[target.clamp(0, len(mapping) - 1)]
    return (pred_mapped[valid] == target_mapped[valid]).float().mean().item()


# ─── Evaluation loop ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    vessel_dice_total = 0.0
    anatomy_acc_total = 0.0
    anatomy_group_acc_total = 0.0
    anatomy_artery_acc_total = 0.0

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        image = batch["image"].to(device, dtype=torch.float32)
        vessel_mask = batch["vessel_mask"].to(device, dtype=torch.float32)
        anatomy_mask = batch["anatomy_mask"].to(device, dtype=torch.long)

        dev_batch = {
            "vessel_mask": vessel_mask,
            "anatomy_mask": anatomy_mask,
        }
        outputs = model(image)

        total_loss += compute_loss(outputs, dev_batch)
        vessel_dice_total += dice_score(outputs["vessel"], vessel_mask)
        anatomy_acc_total += anatomy_accuracy(outputs["anatomy"], anatomy_mask)
        anatomy_group_acc_total += mapped_accuracy(outputs["anatomy"], anatomy_mask, SEGMENT_TO_GROUP_ID, device)
        anatomy_artery_acc_total += mapped_accuracy(outputs["anatomy"], anatomy_mask, SEGMENT_TO_ARTERY_ID, device)

    n = max(len(dataloader), 1)
    return {
        "loss": total_loss / n,
        "vessel_dice": vessel_dice_total / n,
        "anatomy_acc": anatomy_acc_total / n,
        "anatomy_group_acc": anatomy_group_acc_total / n,
        "anatomy_artery_acc": anatomy_artery_acc_total / n,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate multitask localization model")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data-dir", default="dataset", help="Dataset root directory")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Split to evaluate on")
    parser.add_argument("--image-size", type=int, default=512, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="", help="cuda or cpu (auto-detects if empty)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    dataset = CoronaryMultiTaskDataset(
        args.data_dir,
        split=args.split,
        image_size=(args.image_size, args.image_size),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    print(f"Images: {len(dataset)} ({args.split} split)")

    model = MultiTaskMobileUNetv3(n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    # strict=False: older checkpoints may still contain a now-removed stenosis_head
    model.load_state_dict(state, strict=False)
    print(f"Checkpoint: {args.checkpoint}\n")

    metrics = evaluate(model, loader, device)

    print("\n" + "=" * 50)
    print("LOCALIZATION MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Val Loss:               {metrics['loss']:.4f}")
    print(f"  Vessel Dice:            {metrics['vessel_dice']:.4f}")
    print(f"  Anatomy Acc (segment):  {metrics['anatomy_acc']:.4f}  <- 25 SYNTAX segments")
    print(f"  Anatomy Acc (group):    {metrics['anatomy_group_acc']:.4f}  <- e.g. LAD proximal, RCA mid")
    print(f"  Anatomy Acc (artery):   {metrics['anatomy_artery_acc']:.4f}  <- RCA / LAD / LCX / LM")
    print("=" * 50)
