import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import SegmentationDataset
from model import UNet
from model_lightweight import DSCUNet, MobileUNet, MobileUNetv2, MobileUNetv3, DeepLabV3Plus, DeepLabV3ResNet

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def train_net(net, device, args, epochs=5, batch_size=1, lr=1e-4, val_percent=0.1, save_cp=True, img_scale=0.5, data_dir='dataset'):
    # 1. Create dataset
    try:
        dataset = SegmentationDataset(data_dir, dataset_type=args.dataset_type, split='train')
    except FileNotFoundError as e:
        logging.error(f"Dataset not found at {data_dir}. Please specify correct path with --data-dir.")
        raise e
    
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    # 4. Set up optimizer, the loss, the learning rate scheduler
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch
                
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred_raw = net(images)
                masks_pred = masks_pred_raw['out'] if isinstance(masks_pred_raw, dict) else masks_pred_raw
                
                # Squeeze the channel dim of masks_pred (B, 1, H, W) -> (B, H, W)
                # But BCEWithLogitsLoss expects (B, 1, H, W) if target is (B, 1, H, W)
                # My dataset returns (H, W) for mask. DataLoader stacks to (B, H, W).
                # So I need to unsqueeze target:
                true_masks = true_masks.unsqueeze(1)
                
                loss = criterion(masks_pred, true_masks) \
                       + dice_loss(F.sigmoid(masks_pred), true_masks)

                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(images.shape[0])

        # Validation
        val_score = evaluate(net, val_loader, device)
        scheduler.step(val_score)
        logging.info(f'Validation Dice: {val_score}')

        if save_cp:
            checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else os.path.join('checkpoints', args.model)
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved to {checkpoint_dir}!')

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.no_grad():
        for batch in dataloader:
            image, mask_true = batch
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.float32)
            mask_true = mask_true.unsqueeze(1)

            mask_pred_raw = net(image)
            mask_pred = mask_pred_raw['out'] if isinstance(mask_pred_raw, dict) else mask_pred_raw
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            
            # Dice calculation
            # intersection = (mask_pred * mask_true).sum()
            # union = mask_pred.sum() + mask_true.sum()
            # dice = (2. * intersection) / (union + 1e-8)
            # dice_score += dice.item()
            
            # Use the function I defined
            dice_score += (1 - dice_loss(mask_pred, mask_true)).item() # dice_loss returns 1 - dice

    net.train()
    return dice_score / num_val_batches


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--model', '-m', type=str, default='unet', choices=['unet', 'dscunet', 'mobileunet', 'mobileunetv2', 'mobileunetv3', 'deeplabv3', 'deeplabv3_resnet'], help='Model architecture')
    parser.add_argument('--data-dir', type=str, default='dataset', help='Path to dataset root directory')
    parser.add_argument('--dataset-type', type=str, default='syntax', help='Dataset subfolder name under --data-dir, e.g. syntax or combined')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Override checkpoint save directory (default: checkpoints/<model>)')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if args.model == 'unet':
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'dscunet':
        net = DSCUNet(n_channels=3, n_classes=args.classes)
    elif args.model == 'mobileunet':
        net = MobileUNet(n_classes=args.classes)
    elif args.model == 'mobileunetv2':
        net = MobileUNetv2(n_classes=args.classes)
    elif args.model == 'mobileunetv3':
        net = MobileUNetv3(n_classes=args.classes)
    elif args.model == 'deeplabv3':
        net = DeepLabV3Plus(n_classes=args.classes)
    elif args.model == 'deeplabv3_resnet':
        net = DeepLabV3ResNet(n_classes=args.classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    logging.info(f'Model: {args.model}')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    
    try:
        train_net(net=net, 
                  args=args,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  data_dir=args.data_dir)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt checkpoint')
        sys.exit(0)
