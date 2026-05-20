import cv2
import os
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np

# Create figures directory if it doesn't exist
os.makedirs("E:/Research/figures", exist_ok=True)

# 1. Load Original Image
image_path = "E:/Research/dataset/syntax/train/images/10.png"
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Apply CLAHE (Preprocessing)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(img_lab)
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
img_clahe_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
img_clahe_rgb = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB)

# 3. Apply Elastic Transform (Augmentation)
# Set seed for reproducibility
import random
random.seed(42)
np.random.seed(42)

elastic_transform = A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0) # Removed invalid alpha_affine
augmented_elastic = elastic_transform(image=img_clahe_rgb)
img_elastic_rgb = augmented_elastic['image']

# 4. Apply Random Brightness Contrast
bc_transform = A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0)
augmented_bc = bc_transform(image=img_clahe_rgb)
img_bc_rgb = augmented_bc['image']


# 5. Plot the Figure
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
plt.subplots_adjust(wspace=0.05)

axes[0].imshow(img_rgb)
axes[0].set_title("(A) Original Angiogram", fontsize=14, pad=10)
axes[0].axis('off')

axes[1].imshow(img_clahe_rgb)
axes[1].set_title("(B) CLAHE Enhanced", fontsize=14, pad=10)
axes[1].axis('off')

axes[2].imshow(img_elastic_rgb)
axes[2].set_title("(C) Elastic Deformation", fontsize=14, pad=10)
axes[2].axis('off')

axes[3].imshow(img_bc_rgb)
axes[3].set_title("(D) Intensity Perturbation", fontsize=14, pad=10)
axes[3].axis('off')

# Save the figure
save_path = "E:/Research/figures/Figure_1_Preprocessing_Augmentation.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Figure saved successfully to: {save_path}")
