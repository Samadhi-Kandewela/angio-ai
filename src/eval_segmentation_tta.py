"""
Evaluate MobileUNetv3 vessel segmentation with test-time flip augmentation
(TTA): average the sigmoid probabilities of the original image and its
horizontal/vertical/both-axis flips (flipped back to the original
orientation) before thresholding.

Read-only with respect to checkpoints — this does not retrain or overwrite
any existing model file, it only changes how an already-trained checkpoint
is queried at inference time. Safe to run alongside other in-progress work.

Usage:
    python src/eval_segmentation_tta.py \
        --checkpoint checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth \
        --data-dir dataset --split test --device cuda
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from eval_segmentation import VesselSegDataset, segmentation_metrics
from model_lightweight import MobileUNetv3
from tta import predict_tta


@torch.no_grad()
def evaluate(model, loader, device, use_tta):
    model.eval()
    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    n = 0

    for batch in tqdm(loader, desc=f"Evaluating ({'TTA' if use_tta else 'plain'})", unit="img"):
        image = batch["image"].to(device, dtype=torch.float32)
        mask_true = batch["mask"].to(device)

        if use_tta:
            probs = predict_tta(model, image)
        else:
            out = model(image)
            logits = out["out"] if isinstance(out, dict) else out
            probs = torch.sigmoid(logits)

        pred = (probs.squeeze(1) > 0.5).float()

        for i in range(pred.size(0)):
            dice, iou, precision, recall = segmentation_metrics(pred[i], mask_true[i])
            totals["dice"] += dice
            totals["iou"] += iou
            totals["precision"] += precision
            totals["recall"] += recall
            n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}, n


def get_args():
    p = argparse.ArgumentParser(description="Evaluate MobileUNetv3 with flip test-time augmentation")
    p.add_argument("--checkpoint", default="checkpoints/mobileunetv3/mobileunetv3_augmented_best.pth")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
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

    model = MobileUNetv3(n_classes=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Checkpoint: {args.checkpoint} (read-only, not modified)\n")

    plain_metrics, n = evaluate(model, loader, device, use_tta=False)
    tta_metrics, _ = evaluate(model, loader, device, use_tta=True)

    print("\n" + "=" * 60)
    print("MOBILEUNET-V3 SEGMENTATION: PLAIN vs FLIP-TTA")
    print("=" * 60)
    print(f"  Images evaluated: {n}")
    print(f"  {'Metric':<12}{'Plain':>12}{'TTA':>12}{'Delta':>12}")
    for key in ("dice", "iou", "precision", "recall"):
        delta = tta_metrics[key] - plain_metrics[key]
        print(f"  {key:<12}{plain_metrics[key]:>12.4f}{tta_metrics[key]:>12.4f}{delta:>+12.4f}")
    print("=" * 60)
