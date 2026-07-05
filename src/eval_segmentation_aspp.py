"""
Evaluate MobileUNetv3ASPP (model_mobileunet_aspp.py) checkpoints.

Identical metrics/dataset handling to eval_segmentation.py, just pointed at
the ASPP architecture instead of the plain MobileUNetv3 -- eval_segmentation.py
is hardcoded to the plain model class, so ASPP checkpoints don't load there
(different bottleneck channel count). Kept as a separate file so
eval_segmentation.py stays untouched.

Usage:
    python src/eval_segmentation_aspp.py \
        --checkpoint checkpoints/mobileunetv3_aspp/best.pth \
        --data-dir dataset --split test --device cuda
"""

import argparse
import os
import sys

from torch.utils.data import DataLoader
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from eval_segmentation import VesselSegDataset, evaluate
from model_mobileunet_aspp import MobileUNetv3ASPP


def get_args():
    p = argparse.ArgumentParser(description="Evaluate the MobileUNetv3ASPP segmentation model")
    p.add_argument("--checkpoint", default="checkpoints/mobileunetv3_aspp/best.pth")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--aspp-channels", type=int, default=256)
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    dataset = VesselSegDataset(args.data_dir, args.split, (args.image_size, args.image_size))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    print(f"Images: {len(dataset)} ({args.split} split)")

    model = MobileUNetv3ASPP(n_classes=1, pretrained=False, aspp_out_channels=args.aspp_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Checkpoint: {args.checkpoint}\n")

    metrics, n = evaluate(model, loader, device)

    print("\n" + "=" * 45)
    print("MOBILEUNET-V3-ASPP SEGMENTATION EVALUATION")
    print("=" * 45)
    print(f"  Images evaluated: {n}")
    print(f"  Dice:             {metrics['dice']:.4f}")
    print(f"  IoU:              {metrics['iou']:.4f}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print("=" * 45)
