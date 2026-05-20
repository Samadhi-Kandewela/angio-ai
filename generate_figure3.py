import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms

import sys
sys.path.append('E:/Research/src')
from model import UNet
from model_lightweight import MobileUNetv3

# Paths
IMAGE_PATH = "E:/Research/dataset/syntax/test/images/130.png"
UNET_CKPT = "E:/Research/checkpoints/unet_baseline_final.pth"
MOBILE_CKPT = "E:/Research/checkpoints/mobileunetv3_best.pth"
OUTPUT_PATH = "E:/Research/figures/Figure_3_Attention_Proof.png"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Models
print("Loading UNet Baseline...")
unet = UNet(n_channels=3, n_classes=1, bilinear=False)
unet.load_state_dict(torch.load(UNET_CKPT, map_location=device))
unet.to(device)
unet.eval()

print("Loading MobileUNet-v3...")
mobilenet = MobileUNetv3(n_classes=1, pretrained=False)
mobilenet.load_state_dict(torch.load(MOBILE_CKPT, map_location=device))
mobilenet.to(device)
mobilenet.eval()

# 2. Preprocess Image
print("Preprocessing Image...")
img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(img_lab)
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
img_clahe_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
img_clahe_rgb = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB)

# Resize & Normalize
img_resized = cv2.resize(img_clahe_rgb, (512, 512))
img_normalized = img_resized.astype(np.float32) / 255.0

# Image for display (original resized)
img_disp = cv2.resize(img_rgb, (512, 512))

# Tensor conversion
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img_normalized).unsqueeze(0).to(device)

# 3. Inference
with torch.no_grad():
    print("Running UNet Interference...")
    out_u = unet(input_tensor)
    if isinstance(out_u, dict): out_u = out_u['out']
    unet_output = torch.sigmoid(out_u).squeeze().cpu().numpy()
    
    print("Running MobileUNet-v3 Interference...")
    out_m = mobilenet(input_tensor)
    if isinstance(out_m, dict): out_m = out_m['out']
    mobile_output = torch.sigmoid(out_m).squeeze().cpu().numpy()

# Apply a threshold to get strict binary predictions (e.g. 0.5)
threshold = 0.5
unet_mask = (unet_output > threshold).astype(np.uint8)
mobile_mask = (mobile_output > threshold).astype(np.uint8)

# Convert the resized original image to grayscale for better contrast with colored masks
img_disp_gray = cv2.cvtColor(img_disp, cv2.COLOR_RGB2GRAY)
img_disp_gray_rgb = cv2.cvtColor(img_disp_gray, cv2.COLOR_GRAY2RGB) # back to 3 channel for colored overlay

# Function to overlay a mask in a specific color
def create_overlay(base_img, mask, color=(255, 0, 0), alpha=0.6):
    overlay = base_img.copy()
    
    # Create a colored mask
    colored_mask = np.zeros_like(base_img)
    colored_mask[mask == 1] = color
    
    # Blend the original image and the colored mask only where the mask is positive
    mask_indices = mask == 1
    overlay[mask_indices] = cv2.addWeighted(base_img, 1 - alpha, colored_mask, alpha, 0)[mask_indices]
    
    return overlay

# Create overlays (U-Net in Red, MobileUNet in Green for contrast)
overlay_unet = create_overlay(img_disp_gray_rgb, unet_mask, color=(255, 0, 0)) # Red
overlay_mobile = create_overlay(img_disp_gray_rgb, mobile_mask, color=(0, 255, 0)) # Green

# 4. Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_disp_gray, cmap='gray')
axes[0].set_title("Input Angiogram\n(Challenging Low-Contrast Frame)", fontsize=14, pad=10)
axes[0].axis('off')

axes[1].imshow(overlay_unet)
axes[1].set_title("Standard U-Net Prediction\n(False Positives indicated in Red)", fontsize=14, pad=10)
axes[1].axis('off')

axes[2].imshow(overlay_mobile)
axes[2].set_title("MobileUNet-v3 Prediction (CBAM)\n(Clean Localization indicated in Green)", fontsize=14, pad=10)
axes[2].axis('off')

os.makedirs("E:/Research/figures", exist_ok=True)
plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=300)
print(f"Figure saved to: {OUTPUT_PATH}")
