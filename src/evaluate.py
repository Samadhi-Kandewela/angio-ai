import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import SegmentationDataset
from model import UNet
from model_lightweight import MobileUNet, DSCUNet
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='dataset', help='Path to dataset')
    parser.add_argument('--device', default='', help='Device (cpu/cuda)')
    parser.add_argument('--dataset-type', default='syntax', help='Dataset type (syntax/stenosis)')
    return parser.parse_args()

def evaluate_model(model, dataloader, device):
    model.eval()
    dice_score = 0
    steps = 0
    
    with torch.no_grad():
        for batch in dataloader:
            image, mask_true = batch
            image = image.to(device)
            mask_true = mask_true.to(device).unsqueeze(1)
            
            mask_pred = model(image)
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            
            # Simple Dice
            smooth = 1.
            intersection = (mask_pred * mask_true).sum()
            union = mask_pred.sum() + mask_true.sum()
            dice = (2. * intersection + smooth) / (union + smooth)
            
            dice_score += dice.item()
            steps += 1
            
    if steps == 0:
        return 0
    return dice_score / steps

if __name__ == '__main__':
    args = get_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    print(f"Evaluating on {device}")
    
    # Dataset
    try:
        data_dir = args.data_dir
        # Use same validation split as training
        dataset = SegmentationDataset(data_dir, dataset_type=args.dataset_type, split='train')
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        _, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        print(f"Dataset loaded: {len(val_set)} validation images from {data_dir}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Please specify correct path with --data-dir (default: 'dataset')")
        exit(1)
    
    # Models to evaluate
    models = [
        {'name': 'unet', 'class': UNet, 'args': {'n_channels':3, 'n_classes':1, 'bilinear':False}, 'path': 'checkpoints/unet_baseline_final.pth'},
        {'name': 'mobileunet', 'class': MobileUNet, 'args': {'n_classes':1}, 'path': 'checkpoints/mobileunet_final.pth'},
        {'name': 'dscunet', 'class': DSCUNet, 'args': {'n_channels':3, 'n_classes':1}, 'path': 'checkpoints/dscunet_final.pth'}
    ]
    
    print("\nStarting Evaluation...")
    print(f"{'Model':<20} | {'Dice Score':<10}")
    print("-" * 35)
    
    for m in models:
        path = m['path']
        if not os.path.exists(path):
            print(f"{m['name']:<20} | Not Found (Skipping)")
            continue
            
        try:
            net = m['class'](**m['args'])
            net.load_state_dict(torch.load(path, map_location=device))
            net.to(device)
            score = evaluate_model(net, val_loader, device)
            print(f"{m['name']:<20} | {score:.4f}")
        except Exception as e:
            print(f"{m['name']:<20} | Error: {e}")
            
    print("\nEvaluation Complete.")
