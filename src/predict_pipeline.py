"""
Run the full two-stage pipeline (MobileUNetv3 segmentation -> MaskLocalizationNet
anatomy localization) on a single input image, and save visualizations.

Usage:
    python src/predict_pipeline.py --image path/to/frame.png
    python src/predict_pipeline.py --image path/to/frame.png --point 240,180

--point takes "y,x" pixel coordinates in the ORIGINAL image and prints which
merged anatomy segment the pipeline assigns to that point (majority vote in a
small neighborhood), the same idea as localization.py's localize_point but
using the new 15-class merged label scheme.
"""

import argparse
import os
import sys

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from localization_labels import MERGED_NUM_ANATOMY_CLASSES, MERGED_SEGMENT_LABELS, merged_segment_label
from model_lightweight import MobileUNetv3
from model_mask_localization import MaskLocalizationNet

# Fixed BGR color per merged class id (1-14); index 0 (background) unused in overlay.
_PALETTE = [
    (0, 0, 0),
    (60, 60, 220), (30, 170, 250), (0, 200, 200), (50, 200, 50),
    (200, 130, 0), (220, 60, 60), (200, 0, 200), (128, 128, 0),
    (0, 128, 255), (180, 105, 255), (0, 215, 255), (255, 191, 0),
    (147, 20, 255), (139, 139, 0),
]


def build_transform(image_size):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def localize_point_merged(class_map, point_yx, radius=7):
    y, x = int(point_yx[0]), int(point_yx[1])
    h, w = class_map.shape
    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
    x1, x2 = max(0, x - radius), min(w, x + radius + 1)
    patch = class_map[y1:y2, x1:x2]
    valid = patch > 0
    if not np.any(valid):
        return None
    ids, counts = np.unique(patch[valid], return_counts=True)
    return int(ids[np.argmax(counts)])


def get_args():
    p = argparse.ArgumentParser(description="Run the full segmentation+localization pipeline on one image")
    p.add_argument("--image", required=True)
    p.add_argument("--seg-checkpoint", default="checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth")
    p.add_argument("--loc-checkpoint", default="checkpoints/mask_localization_v2/best.pth")
    p.add_argument("--output-dir", default="")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--point", default="", help="'y,x' pixel coords in the original image to query")
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = build_transform(args.image_size)
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    seg_model = MobileUNetv3(n_classes=1, pretrained=False).to(device).eval()
    seg_model.load_state_dict(torch.load(args.seg_checkpoint, map_location=device))
    print(f"Segmentation checkpoint: {args.seg_checkpoint}")

    loc_model = MaskLocalizationNet(n_anatomy_classes=MERGED_NUM_ANATOMY_CLASSES, pretrained=False).to(device).eval()
    loc_model.load_state_dict(torch.load(args.loc_checkpoint, map_location=device))
    print(f"Localization checkpoint: {args.loc_checkpoint}\n")

    with torch.no_grad():
        seg_out = seg_model(tensor)
        seg_logits = seg_out["out"] if isinstance(seg_out, dict) else seg_out
        vessel_pred = (torch.sigmoid(seg_logits) > args.threshold).float()

        loc_out = loc_model(vessel_pred)
        anatomy_pred = torch.argmax(loc_out["anatomy"], dim=1)[0].cpu().numpy().astype(np.uint8)

    # The localization model is only ever supervised on ground-truth vessel
    # pixels (vessel_only_cross_entropy masks out background during training),
    # so its raw argmax is meaningless outside the vessel mask. Constrain the
    # anatomy prediction to the segmentation model's predicted vessel pixels
    # before using it for anything.
    anatomy_pred[vessel_pred[0, 0].cpu().numpy() == 0] = 0

    vessel_mask_small = (vessel_pred[0, 0].cpu().numpy() > 0).astype(np.uint8) * 255
    vessel_mask_full = cv2.resize(vessel_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    anatomy_full = cv2.resize(anatomy_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # ── Stats ──────────────────────────────────────────────────────────────
    vessel_pixels = int((vessel_mask_full > 0).sum())
    total_pixels = orig_h * orig_w
    print(f"Image: {args.image} ({orig_w}x{orig_h})")
    print(f"Vessel coverage: {vessel_pixels}/{total_pixels} px ({100.0 * vessel_pixels / total_pixels:.2f}%)\n")

    print("Anatomy segments detected:")
    ids, counts = np.unique(anatomy_full, return_counts=True)
    rows = [(cid, cnt) for cid, cnt in zip(ids, counts) if cid > 0]
    rows.sort(key=lambda r: -r[1])
    for cid, cnt in rows:
        pct = 100.0 * cnt / max(vessel_pixels, 1)
        print(f"  {merged_segment_label(cid):<45} {cnt:>8d} px  ({pct:5.1f}% of vessel)")

    # ── Point query ────────────────────────────────────────────────────────
    if args.point:
        y_str, x_str = args.point.split(",")
        result = localize_point_merged(anatomy_full, (int(y_str), int(x_str)))
        print(f"\nPoint ({y_str}, {x_str}): ", end="")
        print("no vessel nearby" if result is None else merged_segment_label(result))

    # ── Save overlays ──────────────────────────────────────────────────────
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.image))
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    vessel_overlay = image_bgr.copy()
    vessel_overlay[vessel_mask_full > 0] = (0, 0, 255)
    vessel_blend = cv2.addWeighted(image_bgr, 0.6, vessel_overlay, 0.4, 0)

    anatomy_color = np.zeros_like(image_bgr)
    for cid in range(1, MERGED_NUM_ANATOMY_CLASSES):
        anatomy_color[anatomy_full == cid] = _PALETTE[cid % len(_PALETTE)]
    anatomy_blend = image_bgr.copy()
    fg = anatomy_full > 0
    anatomy_blend[fg] = cv2.addWeighted(image_bgr, 0.4, anatomy_color, 0.6, 0)[fg]

    mask_path = os.path.join(output_dir, f"{stem}_vessel_mask.png")
    vessel_overlay_path = os.path.join(output_dir, f"{stem}_vessel_overlay.png")
    anatomy_overlay_path = os.path.join(output_dir, f"{stem}_anatomy_overlay.png")
    cv2.imwrite(mask_path, vessel_mask_full)
    cv2.imwrite(vessel_overlay_path, vessel_blend)
    cv2.imwrite(anatomy_overlay_path, anatomy_blend)

    print(f"\nSaved mask:            {mask_path}")
    print(f"Saved vessel overlay:  {vessel_overlay_path}")
    print(f"Saved anatomy overlay: {anatomy_overlay_path}")
