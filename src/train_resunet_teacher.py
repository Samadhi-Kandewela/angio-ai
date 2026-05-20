import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import SegmentationDataset

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.eps = 1e-6

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth + self.eps)  
        FocalTversky = (1 - Tversky + self.eps)**self.gamma
                       
        return FocalTversky

def train_net(net, device, data_dir, epochs=50, batch_size=2, lr=1e-5, save_cp=True, dir_checkpoint='checkpoints/resunet_teacher/'):
    train_dataset = SegmentationDataset(data_dir, split='train', image_size=(512, 512))
    val_dataset = SegmentationDataset(data_dir, split='test', image_size=(512, 512)) 
    
    dataset = train_dataset
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    
    logging.info(f'''Starting training ResU-Net:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=2.0)
    
    best_dice = 0.0

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, true_masks = batch

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                true_masks = true_masks.unsqueeze(1)

                optimizer.zero_grad()
                
                # ResUnet from smp returns logits directly
                pred = net(imgs)
                loss = criterion(pred, true_masks)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])

        val_score = evaluate(net, val_loader, device)
        logging.info(f'Validation Dice: {val_score}')
        
        if val_score > best_dice:
            best_dice = val_score
            if save_cp:
                os.makedirs(dir_checkpoint, exist_ok=True)
                torch.save(net.state_dict(), dir_checkpoint + f'resunet_best.pth')
                logging.info(f'New best model saved!')

        if save_cp and (epoch + 1) % 5 == 0:
             os.makedirs(dir_checkpoint, exist_ok=True)
             torch.save(net.state_dict(), dir_checkpoint + f'checkpoint_epoch{epoch + 1}.pth')

def evaluate(net, dataloader, device):
    net.eval()
    mask_type = torch.float32
    n_val = len(dataloader)
    dice_score = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, total=n_val, desc='Validation', unit='batch', leave=False):
            image, mask_true = batch
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=mask_type)
            mask_true = mask_true.unsqueeze(1)

            output = net(image)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
            
            intersection = (output * mask_true).sum()
            union = output.sum() + mask_true.sum()
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_score += dice.item()

    return dice_score / n_val

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--data-dir', type=str, default='dataset', help='Path to dataset root')
    parser.add_argument('--load', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load state-of-the-art ResU-Net using SMP
    logging.info("Loading Teacher: ResU-Net (U-Net with ResNet34 Encoder)...")
    net = smp.Unet(
        encoder_name="resnet34",      # use ResNet34 encoder 
        encoder_weights="imagenet",   # use pre-trained weights for faster convergence
        in_channels=3,                # model input channels
        classes=1                     # model output channels (binary segmentation)
    )
    net.to(device=device)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    try:
        train_net(net=net, device=device, data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED_resunet.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
