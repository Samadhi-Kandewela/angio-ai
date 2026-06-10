"""
Evaluate the mask-input anatomy localization model.

Usage:
    python src/eval_mask_localization.py ^
        --checkpoint checkpoints/mask_localization/best.pth ^
        --data-dir dataset ^
        --device cuda
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_mask_localization import MaskLocalizationDataset
from localization_labels import (
    NUM_ANATOMY_CLASSES,
    SEGMENT_TO_ARTERY_ID,
    SEGMENT_TO_GROUP_ID,
)
from model_mask_localization import MaskLocalizationNet


# ─── Metrics ─────────────────────────────────────────────────────────────────

def vessel_only_cross_entropy(logits, target):
    valid = target > 0
    if valid.sum() == 0:
        return 0.0
    return F.cross_entropy(logits, target, reduction="none")[valid].mean().item()


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


# ─── Evaluation loop ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = anat_acc = anat_grp = anat_art = 0.0

    for batch in tqdm(loader, desc="Evaluating", unit="batch"):
        vessel_mask = batch["vessel_mask"].to(device, dtype=torch.float32)
        anatomy_mask = batch["anatomy_mask"].to(device, dtype=torch.long)

        outputs = model(vessel_mask)

        total_loss += vessel_only_cross_entropy(outputs["anatomy"], anatomy_mask)
        anat_acc += anatomy_accuracy(outputs["anatomy"], anatomy_mask)
        anat_grp += mapped_accuracy(outputs["anatomy"], anatomy_mask, SEGMENT_TO_GROUP_ID, device)
        anat_art += mapped_accuracy(outputs["anatomy"], anatomy_mask, SEGMENT_TO_ARTERY_ID, device)

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "anatomy_acc": anat_acc / n,
        "anatomy_group_acc": anat_grp / n,
        "anatomy_artery_acc": anat_art / n,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Evaluate mask-input localization model")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    dataset = MaskLocalizationDataset(
        args.data_dir,
        split=args.split,
        image_size=(args.image_size, args.image_size),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Images: {len(dataset)} ({args.split} split)")

    model = MaskLocalizationNet(n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Checkpoint: {args.checkpoint}\n")

    metrics = evaluate(model, loader, device)

    print("\n" + "=" * 50)
    print("MASK LOCALIZATION EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Val Loss:               {metrics['loss']:.4f}")
    print(f"  Anatomy Acc (segment):  {metrics['anatomy_acc']:.4f}  <- 25 SYNTAX segments")
    print(f"  Anatomy Acc (group):    {metrics['anatomy_group_acc']:.4f}  <- e.g. LAD proximal")
    print(f"  Anatomy Acc (artery):   {metrics['anatomy_artery_acc']:.4f}  <- RCA / LAD / LCX / LM")
    print("=" * 50)
