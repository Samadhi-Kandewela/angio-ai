"""
End-to-end evaluation of the two-stage segmentation + localization pipeline.

Pipeline:
  Angiogram -> [MobileUNetv3 vessel segmentation] -> predicted binary mask
            -> [MaskLocalizationNet] -> per-pixel anatomy map

Unlike eval_mask_localization.py (which feeds the localization model
ground-truth vessel masks), this script measures the FULL pipeline: the
localization model sees the segmentation model's actual predictions, so
segmentation errors propagate into the anatomy accuracy numbers.

Usage:
    python src/eval_pipeline.py \
        --seg-checkpoint checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth \
        --loc-checkpoint checkpoints/mask_localization/best.pth \
        --data-dir dataset --split val --device cuda
"""

import argparse
import os
import sys

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_mask_localization import MaskLocalizationDataset
from localization_labels import (
    MERGED_NUM_ANATOMY_CLASSES as NUM_ANATOMY_CLASSES,
    MERGED_SEGMENT_TO_ARTERY_ID as SEGMENT_TO_ARTERY_ID,
    MERGED_SEGMENT_TO_GROUP_ID as SEGMENT_TO_GROUP_ID,
)
from model_lightweight import MobileUNetv3
from model_mask_localization import MaskLocalizationNet


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PipelineDataset(Dataset):
    """
    Pairs each syntax-split image with a seg-model-ready tensor and its
    ground-truth anatomy mask, indexed identically to MaskLocalizationDataset
    (same file, same resize) so segmentation output lines up pixel-for-pixel
    with the anatomy target.
    """

    def __init__(self, data_dir, split, image_size):
        self.anatomy_ds = MaskLocalizationDataset(
            data_dir, split=split, image_size=image_size, augment=False, mask_noise=False
        )
        self.images_dir = os.path.join(data_dir, "syntax", split, "images")
        h, w = image_size
        self.image_transform = A.Compose([
            A.Resize(height=h, width=w),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.anatomy_ds)

    def __getitem__(self, idx):
        sample = self.anatomy_ds[idx]
        img_path = os.path.join(self.images_dir, sample["file_name"])
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image_rgb)["image"]

        return {
            "image": image,
            "anatomy_mask": sample["anatomy_mask"],
            "file_name": sample["file_name"],
        }


# ─── Segmentation metrics ─────────────────────────────────────────────────────

def dice_iou(pred, target, smooth=1.0):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    return dice.item(), iou.item()


# ─── Localization metrics (same definitions as eval_mask_localization.py) ────

def anatomy_accuracy(logits, target):
    pred = torch.argmax(logits, dim=1)
    valid = target > 0
    if valid.sum() == 0:
        return None
    return (pred[valid] == target[valid]).float().mean().item()


def mapped_accuracy(logits, target, mapping, device):
    pred = torch.argmax(logits, dim=1)
    valid = target > 0
    if valid.sum() == 0:
        return None
    map_tensor = torch.as_tensor(mapping, device=device, dtype=torch.long)
    pred_mapped = map_tensor[pred.clamp(0, len(mapping) - 1)]
    target_mapped = map_tensor[target.clamp(0, len(mapping) - 1)]
    return (pred_mapped[valid] == target_mapped[valid]).float().mean().item()


# ─── Evaluation loop ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(seg_model, loc_model, loader, device):
    seg_model.eval()
    loc_model.eval()

    dice_sum = iou_sum = 0.0
    acc_sum = grp_sum = art_sum = 0.0
    n_seg = 0
    n_loc = 0

    for batch in tqdm(loader, desc="Evaluating pipeline", unit="batch"):
        image = batch["image"].to(device, dtype=torch.float32)
        anatomy_mask = batch["anatomy_mask"].to(device, dtype=torch.long)
        vessel_gt = (anatomy_mask > 0).float().unsqueeze(1)

        seg_out = seg_model(image)
        seg_logits = seg_out["out"] if isinstance(seg_out, dict) else seg_out
        vessel_pred = (torch.sigmoid(seg_logits) > 0.5).float()

        for i in range(vessel_pred.size(0)):
            dice, iou = dice_iou(vessel_pred[i], vessel_gt[i])
            dice_sum += dice
            iou_sum += iou
            n_seg += 1

        loc_out = loc_model(vessel_pred)
        acc = anatomy_accuracy(loc_out["anatomy"], anatomy_mask)
        if acc is not None:
            acc_sum += acc
            grp_sum += mapped_accuracy(loc_out["anatomy"], anatomy_mask, SEGMENT_TO_GROUP_ID, device)
            art_sum += mapped_accuracy(loc_out["anatomy"], anatomy_mask, SEGMENT_TO_ARTERY_ID, device)
            n_loc += 1

    return {
        "dice": dice_sum / max(n_seg, 1),
        "iou": iou_sum / max(n_seg, 1),
        "anatomy_acc": acc_sum / max(n_loc, 1),
        "anatomy_group_acc": grp_sum / max(n_loc, 1),
        "anatomy_artery_acc": art_sum / max(n_loc, 1),
        "n_images": n_seg,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Evaluate the segmentation + localization pipeline end-to-end")
    p.add_argument("--seg-checkpoint", default="checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth")
    p.add_argument("--loc-checkpoint", default="checkpoints/mask_localization/best.pth")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    dataset = PipelineDataset(args.data_dir, args.split, (args.image_size, args.image_size))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    print(f"Images: {len(dataset)} ({args.split} split)")

    seg_model = MobileUNetv3(n_classes=1, pretrained=False).to(device)
    seg_model.load_state_dict(torch.load(args.seg_checkpoint, map_location=device))
    print(f"Segmentation checkpoint: {args.seg_checkpoint}")

    loc_model = MaskLocalizationNet(n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=False).to(device)
    loc_model.load_state_dict(torch.load(args.loc_checkpoint, map_location=device))
    print(f"Localization checkpoint: {args.loc_checkpoint}\n")

    metrics = evaluate(seg_model, loc_model, loader, device)

    print("\n" + "=" * 55)
    print("END-TO-END PIPELINE EVALUATION (predicted masks, not GT)")
    print("=" * 55)
    print(f"  Images evaluated:       {metrics['n_images']}")
    print(f"  Segmentation Dice:      {metrics['dice']:.4f}")
    print(f"  Segmentation IoU:       {metrics['iou']:.4f}")
    print(f"  Anatomy Acc (segment):  {metrics['anatomy_acc']:.4f}  <- 25 SYNTAX segments")
    print(f"  Anatomy Acc (group):    {metrics['anatomy_group_acc']:.4f}  <- e.g. LAD proximal")
    print(f"  Anatomy Acc (artery):   {metrics['anatomy_artery_acc']:.4f}  <- RCA / LAD / LCX / LM")
    print("=" * 55)
