"""
Compare inference latency between:
  A) Old approach  — single MultiTaskMobileUNetv3 (raw image → vessel + anatomy)
  B) New approach  — MobileUNetv3 + MaskLocalizationNet (two-stage pipeline)

Usage:
    python src/benchmark_latency.py `
        --multitask-checkpoint checkpoints/multitask_localization_v2/multitask_best.pth `
        --seg-checkpoint checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth `
        --loc-checkpoint checkpoints/mask_localization/best.pth `
        --device cuda
"""

import argparse
import os
import sys
import time

import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model_lightweight import MobileUNetv3
from model_mask_localization import MaskLocalizationNet
from model_multitask import MultiTaskMobileUNetv3
from localization_labels import NUM_ANATOMY_CLASSES, MERGED_NUM_ANATOMY_CLASSES


def measure(fn, warmup=10, runs=100):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / runs) * 1000
    fps = runs / elapsed
    return avg_ms, fps


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--multitask-checkpoint",
                   default="checkpoints/multitask_localization_v2/multitask_best.pth")
    p.add_argument("--seg-checkpoint",
                   default="checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth")
    p.add_argument("--loc-checkpoint",
                   default="checkpoints/mask_localization/best.pth")
    p.add_argument("--device", default="")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    H = W = args.image_size

    print(f"Device : {device}")
    print(f"Image  : {H}x{W}")
    print(f"Runs   : {args.runs}  (warmup={args.warmup})\n")

    # ── Load models ──────────────────────────────────────────────────────────

    # Old: single multitask model
    multitask = MultiTaskMobileUNetv3(
        n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=False
    ).to(device)
    multitask.load_state_dict(
        torch.load(args.multitask_checkpoint, map_location=device), strict=False
    )
    multitask.eval()

    # New: two-stage pipeline
    seg_model = MobileUNetv3(n_classes=1, pretrained=False).to(device)
    seg_model.load_state_dict(
        torch.load(args.seg_checkpoint, map_location=device), strict=False
    )
    seg_model.eval()

    loc_model = MaskLocalizationNet(
        n_anatomy_classes=MERGED_NUM_ANATOMY_CLASSES, pretrained=False
    ).to(device)
    loc_model.load_state_dict(
        torch.load(args.loc_checkpoint, map_location=device)
    )
    loc_model.eval()

    # ── Dummy inputs ─────────────────────────────────────────────────────────
    image = torch.randn(1, 3, H, W, device=device)
    mask  = torch.zeros(1, 1, H, W, device=device)
    mask[0, 0, 50:400, 100:450] = 1.0

    # ── Benchmark ────────────────────────────────────────────────────────────
    with torch.no_grad():

        def run_multitask():
            return multitask(image)

        def run_seg():
            return seg_model(image)

        def run_loc():
            return loc_model(mask)

        def run_new_pipeline():
            out = seg_model(image)
            vessel = (torch.sigmoid(out["out"]) > 0.5).float()
            return loc_model(vessel)

        mt_ms,  mt_fps  = measure(run_multitask,   args.warmup, args.runs)
        seg_ms, seg_fps = measure(run_seg,          args.warmup, args.runs)
        loc_ms, loc_fps = measure(run_loc,          args.warmup, args.runs)
        new_ms, new_fps = measure(run_new_pipeline, args.warmup, args.runs)

    # ── Report ────────────────────────────────────────────────────────────────
    W2 = 52
    print("=" * W2)
    print("  APPROACH A — Old: single MultiTaskMobileUNetv3")
    print("-" * W2)
    print(f"  {'MultiTaskMobileUNetv3':<32} {mt_ms:>7.2f} ms  {mt_fps:>6.1f} FPS")
    print()
    print("  APPROACH B — New: MobileUNetv3 + MaskLocalizationNet")
    print("-" * W2)
    print(f"  {'MobileUNetv3 (vessel seg)':<32} {seg_ms:>7.2f} ms  {seg_fps:>6.1f} FPS")
    print(f"  {'MaskLocalizationNet (anatomy)':<32} {loc_ms:>7.2f} ms  {loc_fps:>6.1f} FPS")
    print(f"  {'Combined pipeline':<32} {new_ms:>7.2f} ms  {new_fps:>6.1f} FPS")
    print("=" * W2)
    diff = new_ms - mt_ms
    sign = "+" if diff > 0 else ""
    print(f"\n  Latency difference: {sign}{diff:.2f} ms  ({sign}{diff/mt_ms*100:.1f}%)")
    print()
    for label, ms, fps in [("Old", mt_ms, mt_fps), ("New", new_ms, new_fps)]:
        status = "real-time capable" if ms < 33.3 else "below 30 FPS"
        print(f"  {label}: {fps:.1f} FPS — {status}")
