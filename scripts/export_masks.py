import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def export_masks(base_path, output_base, dataset_type='syntax', splits=['train', 'test']):
    """
    Reads COCO-style JSON annotations and exports binary masks for each image.
    """
    for split in splits:
        json_path = os.path.join(base_path, dataset_type, split, 'annotations', f'{split}.json')
        img_dir = os.path.join(base_path, dataset_type, split, 'images')
        
        # Output directory for this split
        mask_out_dir = os.path.join(output_base, dataset_type, split, 'masks')
        os.makedirs(mask_out_dir, exist_ok=True)
        
        if not os.path.exists(json_path):
            print(f"Skipping {split}: JSON not found at {json_path}")
            continue
            
        print(f"Processing {split} set...")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Map image_id to annotations
        img_map = {img['id']: img for img in data['images']}
        ann_map = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in ann_map:
                ann_map[img_id] = []
            ann_map[img_id].append(ann)
            
        # Process each image
        for img_id, img_info in tqdm(img_map.items(), desc=f"Exporting {split}"):
            file_name = img_info['file_name']
            
            # Create blank mask
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            
            if img_id in ann_map:
                for ann in ann_map[img_id]:
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 255) # 255 for white/foreground
            
            # Save mask
            # We use the same filename but ensure it's png (lossless)
            base_name = os.path.splitext(file_name)[0]
            out_name = f"{base_name}.png"
            out_path = os.path.join(mask_out_dir, out_name)
            
            cv2.imwrite(out_path, mask)

    print(f"\nAll masks exported to {output_base}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_PATH = os.path.join(project_root, "dataset")
    OUTPUT_BASE = os.path.join(project_root, "exports")
    
    # We assume 'syntax' is the dataset type being used, based on other files
    export_masks(BASE_PATH, OUTPUT_BASE, dataset_type='syntax', splits=['train', 'test']) 
