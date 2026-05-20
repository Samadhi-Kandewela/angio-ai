"""
Run MobileUNet-v3 inference on validation images and save predicted masks only.
"""
import os
import sys

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from model_lightweight import MobileUNetv3

CHECKPOINT = os.path.join(
    PROJECT_ROOT, "checkpoints", "mobileunetv3", "mobileunetv3_augmented_best.pth"
)
IMAGES_DIR = os.path.join(PROJECT_ROOT, "dataset", "stenosis", "val", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "stenosis", "val_predicted_masks_only")
IMAGE_SIZE = (512, 512)

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading MobileUNet-v3 model...")
model = MobileUNetv3(n_classes=1, pretrained=False)
state_dict = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model loaded successfully.")

val_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

image_files = sorted(
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)
print(f"Processing {len(image_files)} images...")

count = 0
with torch.no_grad():
    for fname in image_files:
        img_path = os.path.join(IMAGES_DIR, fname)
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [SKIP] Could not read: {fname}")
            continue

        orig_h, orig_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = val_transform(image=image_rgb)
        input_tensor = augmented["image"].unsqueeze(0).to(device)

        output = model(input_tensor)
        logits = output["out"] if isinstance(output, dict) else output
        pred = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary_mask = (pred > 0.5).astype(np.uint8) * 255
        binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + ".png")
        cv2.imwrite(out_path, binary_mask)
        count += 1

        if count % 50 == 0:
            print(f"  Processed {count}/{len(image_files)} images...")

print(f"\nDone! {count} predicted masks saved to:\n  {OUTPUT_DIR}")
