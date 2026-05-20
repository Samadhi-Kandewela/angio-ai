import os
import json
import numpy as np
import cv2

json_path = r'E:\Research\dataset\stenosis\val\annotations\val.json'
out_dir = r'E:\Research\dataset\stenosis\stenosis_val_mask'

print(f"Loading annotations from {json_path}...")
with open(json_path, 'r') as f:
    data = json.load(f)

print(f"Creating output directory at {out_dir}...")
os.makedirs(out_dir, exist_ok=True)

# Map image_id to filename, width, and height
images_info = {}
for img in data['images']:
    images_info[img['id']] = {
        'file_name': img['file_name'],
        'width': img['width'],
        'height': img['height']
    }

print(f"Found {len(images_info)} images in the validation set.")

# Group annotations by image_id
annotations_by_image = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)

print("Generating masks...")
count = 0
for img_id, info in images_info.items():
    width = info['width']
    height = info['height']
    file_name = info['file_name']
    
    # Create a blank black mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if img_id in annotations_by_image:
        for ann in annotations_by_image[img_id]:
            # category_id 26 is "stenosis". Others (1-25) are parts of the artery/vessels
            if 'segmentation' in ann and len(ann['segmentation']) > 0:
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    poly = np.round(poly).astype(np.int32)
                    
                    # Fill the polygon first (for everything, including stenosis to make it part of the mask)
                    cv2.fillPoly(mask, [poly], 255)
                    
                    if ann['category_id'] == 26:
                        # For stenosis, also draw a bounding box outline
                        x, y, w, h = cv2.boundingRect(poly)
                        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, 2) # 2px thickness
    
    # Ensure file_name ends with .png for proper mask saving
    name, ext = os.path.splitext(file_name)
    out_file_name = name + ".png"
    out_path = os.path.join(out_dir, out_file_name)
    
    cv2.imwrite(out_path, mask)
    count += 1

print(f"Successfully generated {count} groundtruth masks in '{out_dir}'.")
