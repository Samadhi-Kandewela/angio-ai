import argparse
import glob
import os
import re

import torch
from torch.utils.data import DataLoader, random_split

from dataset import SegmentationDataset
from model_lightweight import MobileUNetv3
from train import evaluate


def main():
    parser = argparse.ArgumentParser(description='Evaluate all saved per-epoch checkpoints and report the best by validation Dice')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory containing checkpoint_epochN.pth files')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--dataset-type', type=str, default='syntax', help='Dataset subfolder name under --data-dir')
    parser.add_argument('--val-percent', type=float, default=10.0, help='Percent of data used as validation (0-100), must match the training run')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset = SegmentationDataset(args.data_dir, dataset_type=args.dataset_type, split='train')
    n_val = int(len(dataset) * (args.val_percent / 100))
    n_train = len(dataset) - n_val
    _, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=(args.device == 'cuda'))

    ckpt_paths = glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint_epoch*.pth'))
    def epoch_num(p):
        m = re.search(r'checkpoint_epoch(\d+)\.pth', os.path.basename(p))
        return int(m.group(1)) if m else -1
    ckpt_paths.sort(key=epoch_num)

    if not ckpt_paths:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    results = []
    for path in ckpt_paths:
        epoch = epoch_num(path)
        net = MobileUNetv3(n_classes=1)
        net.load_state_dict(torch.load(path, map_location=device))
        net.to(device)
        dice = evaluate(net, val_loader, device)
        results.append((epoch, dice, path))
        print(f"Epoch {epoch}: Validation Dice = {dice:.4f}")

    best_epoch, best_dice, best_path = max(results, key=lambda r: r[1])
    print(f"\nBest: epoch {best_epoch} with Dice {best_dice:.4f} -> {best_path}")


if __name__ == '__main__':
    main()
