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
from model_lightweight import MobileUNetv3
from model_CBAM_UNet import CBAM_UNet

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, temperature=3.0):
        """
        alpha: weight for standard student loss
        beta: weight for feature-based distillation loss
        temperature: softening factor for logit distillation
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()

    def forward(self, student_logits, student_features, teacher_logits, teacher_features, targets):
        # 1. Standard Loss (Student vs Ground Truth)
        student_loss = self.bce(student_logits, targets)

        # 2. Logit Distillation Loss (Student vs Teacher)
        student_2ch = torch.cat([-student_logits, student_logits], dim=1) # (B, 2, H, W)
        teacher_2ch = torch.cat([-teacher_logits, teacher_logits], dim=1)
        
        student_soft = F.log_softmax(student_2ch / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_2ch / self.temperature, dim=1)
        logit_distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # 3. Feature Distillation Loss (MSE between intermediate features)
        # Ensure sizes match via interpolation if spatial dims mismatch
        if student_features.shape[-2:] != teacher_features.shape[-2:]:
            student_features = F.interpolate(student_features, size=teacher_features.shape[-2:], mode='bilinear', align_corners=False)
            
        feature_distill_loss = self.mse(student_features, teacher_features)

        # Total Loss
        total_loss = self.alpha * student_loss + (1 - self.alpha) * logit_distill_loss + self.beta * feature_distill_loss
        return total_loss

class FeatureProjector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.proj(x)

def train_distillation(teacher_path, data_dir, epochs=50, batch_size=4, lr=1e-4, save_cp=True, dir_checkpoint='checkpoints/student_distilled/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = SegmentationDataset(data_dir, split='train', image_size=(768, 768))
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    logging.info(f"Loading Teacher (CBAM-UNet) from {teacher_path}...")
    teacher = CBAM_UNet(n_channels=3, n_classes=1)
    
    # Try to load teacher weights if provided, except if the path doesn't exist yet we continue (for testing pipeline)
    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    else:
        logging.warning("Teacher weights not found, using untrained teacher for pipeline test.")
        
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    logging.info("Initializing Student (MobileUNetv3)...")
    student = MobileUNetv3(n_classes=1)
    student.to(device)
    
    # Projector from student feature dim (960) to teacher feature dim (1024)
    projector = FeatureProjector(960, 1024).to(device)

    optimizer = optim.Adam(list(student.parameters()) + list(projector.parameters()), lr=lr)
    # Give beta=0.1 to feature loss so it doesn't overpower logit loss initially
    criterion = DistillationLoss(alpha=0.5, beta=0.1, temperature=4.0) 

    logging.info(f"Starting Distillation: Epochs={epochs}, Batch={batch_size}, LR={lr}")

    best_dice = 0.0

    for epoch in range(epochs):
        student.train()
        projector.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, masks = batch
                imgs = imgs.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32).unsqueeze(1)

                with torch.no_grad():
                    t_res = teacher(imgs)
                    teacher_logits, teacher_features = t_res['out'], t_res['features']
                
                s_res = student(imgs)
                student_logits, student_features_raw = s_res['out'], s_res['features']
                
                # Project student features to teacher dimension
                student_features_proj = projector(student_features_raw)

                loss = criterion(student_logits, student_features_proj, teacher_logits, teacher_features, masks)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss': f"{loss.item():.4f}"})
                pbar.update(imgs.shape[0])

        val_score = evaluate(student, val_loader, device)
        logging.info(f'Validation Dice: {val_score}')
        
        if val_score > best_dice:
            best_dice = val_score
            if save_cp:
                os.makedirs(dir_checkpoint, exist_ok=True)
                torch.save(student.state_dict(), dir_checkpoint + 'student_best.pth')
                logging.info(f'New best student model saved!')

def evaluate(net, dataloader, device):
    net.eval()
    dice_score = 0
    with torch.no_grad():
        for batch in dataloader:
            image, mask_true = batch
            image = image.to(device, dtype=torch.float32)
            mask_true = mask_true.to(device, dtype=torch.float32).unsqueeze(1)
            
            output = net(image)['out']
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
            
            intersection = (output * mask_true).sum()
            union = output.sum() + mask_true.sum()
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_score += dice.item()
    return dice_score / max(len(dataloader), 1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=str, required=True, help='Path to trained teacher .pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data-dir', type=str, default='dataset', help='Path to dataset root')
    args = parser.parse_args()
    
    try:
        train_distillation(args.teacher, data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    except KeyboardInterrupt:
        sys.exit(0)
