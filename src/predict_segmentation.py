"""
Run vessel segmentation on a single image using a MultiTaskMobileUNetv3 checkpoint.

Usage:
    python src/predict_segmentation.py --checkpoint checkpoints/multitask_v5/multitask_best.pth --image path/to/frame.png
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from localization_labels import NUM_ANATOMY_CLASSES
from model_multitask import MultiTaskMobileUNetv3

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_clahe_rgb(image_rgb):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_chan)
    enhanced_lab = cv2.merge((enhanced_l, a_chan, b_chan))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess(image_bgr, image_size):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = apply_clahe_rgb(image_rgb)
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - MEAN) / STD
    tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1))).unsqueeze(0).float()
    return tensor, image_rgb


def get_args():
    p = argparse.ArgumentParser(description="Run vessel segmentation on a single image")
    p.add_argument("--checkpoint", required=True, help="Path to multitask .pth checkpoint")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--output-dir", default="", help="Where to save mask/overlay (default: alongside input image)")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    orig_h, orig_w = image_bgr.shape[:2]

    model = MultiTaskMobileUNetv3(n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    # strict=False: older checkpoints may still contain a now-removed stenosis_head
    model.load_state_dict(state, strict=False)
    model.eval()

    tensor, preproc_rgb = preprocess(image_bgr, args.image_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        vessel_prob = torch.sigmoid(outputs["vessel"])[0, 0].cpu().numpy()

    vessel_mask = (vessel_prob > args.threshold).astype(np.uint8) * 255
    vessel_mask_full = cv2.resize(vessel_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    overlay = image_bgr.copy()
    overlay[vessel_mask_full > 0] = (0, 0, 255)  # red highlight, BGR
    blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)

    vessel_pixels = int((vessel_mask_full > 0).sum())
    total_pixels = orig_h * orig_w
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image:      {args.image} ({orig_w}x{orig_h})")
    print(f"Vessel coverage: {vessel_pixels}/{total_pixels} px ({100.0 * vessel_pixels / total_pixels:.2f}%)")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.image))
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    mask_path = os.path.join(output_dir, f"{stem}_vessel_mask.png")
    overlay_path = os.path.join(output_dir, f"{stem}_vessel_overlay.png")
    cv2.imwrite(mask_path, vessel_mask_full)
    cv2.imwrite(overlay_path, blended)

    print(f"Saved mask:    {mask_path}")
    print(f"Saved overlay: {overlay_path}")
