"""
Run MobileUNet-v3 inference on validation images and overlay stenosis bounding boxes.

Output: predicted masks (white on black) with red bounding boxes around stenosis regions.
"""
import os
import sys
import json
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src to path so we can import the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model_lightweight import MobileUNetv3

# ── Paths ───────────────────────────────────────────────────────────────
CHECKPOINT  = r'E:\Research\checkpoints\mobileunetv3\mobileunetv3_augmented_best.pth'
IMAGES_DIR  = r'E:\Research\dataset\stenosis\val\images'
JSON_PATH   = r'E:\Research\dataset\stenosis\val\annotations\val.json'
OUTPUT_DIR  = r'E:\Research\dataset\stenosis\val_predicted_masks'
IMAGE_SIZE  = (512, 512)

# ── Setup ───────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Load Model ──────────────────────────────────────────────────────────
print("Loading MobileUNet-v3 model...")
model = MobileUNetv3(n_classes=1, pretrained=False)
state_dict = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model loaded successfully.")

# ── Preprocessing (same as validation in dataset.py) ────────────────────
val_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ── Load Annotations (for stenosis bounding boxes) ─────────────────────
print("Loading annotations...")
with open(JSON_PATH, 'r') as f:
    coco_data = json.load(f)

# Build lookup: filename -> list of stenosis bounding boxes
stenosis_bboxes = {}
img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

for ann in coco_data['annotations']:
    if ann['category_id'] == 26:  # stenosis
        fname = img_id_to_filename[ann['image_id']]
        if fname not in stenosis_bboxes:
            stenosis_bboxes[fname] = []
        # COCO bbox format: [x, y, width, height]
        stenosis_bboxes[fname].append(ann['bbox'])

print(f"Found stenosis annotations for {len(stenosis_bboxes)} images.")

# ── Run Inference ───────────────────────────────────────────────────────
image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Processing {len(image_files)} images...")

count = 0
with torch.no_grad():
    for fname in image_files:
        img_path = os.path.join(IMAGES_DIR, fname)

        # Load original image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [SKIP] Could not read: {fname}")
            continue
        orig_h, orig_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        augmented = val_transform(image=image_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(device)  # (1, 3, 512, 512)

        # Inference
        output = model(input_tensor)                                # dict with 'out' key
        logits = output['out'] if isinstance(output, dict) else output
        pred = torch.sigmoid(logits).squeeze().cpu().numpy()        # (512, 512) float [0,1]

        # Threshold to binary mask
        binary_mask = (pred > 0.5).astype(np.uint8) * 255           # (512, 512) uint8

        # Resize mask back to original image size
        binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Convert to 3-channel so we can draw colored bounding boxes
        mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        # Draw stenosis bounding boxes (in red)
        if fname in stenosis_bboxes:
            for bbox in stenosis_bboxes[fname]:
                x, y, w, h = [int(round(v)) for v in bbox]
                cv2.rectangle(mask_color, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box

        # Save
        out_name = os.path.splitext(fname)[0] + '.png'
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, mask_color)
        count += 1

        if count % 50 == 0:
            print(f"  Processed {count}/{len(image_files)} images...")

print(f"\nDone! {count} predicted masks with bounding boxes saved to:\n  {OUTPUT_DIR}")
