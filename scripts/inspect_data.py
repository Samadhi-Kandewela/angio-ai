import json
import os
import cv2
import numpy as np
import random

def visualize_dataset(base_path, dataset_type, split='train', num_samples=1):
    json_path = os.path.join(base_path, dataset_type, split, 'annotations', f'{split}.json')
    img_dir = os.path.join(base_path, dataset_type, split, 'images')
    
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data['images'])} images and {len(data['annotations'])} annotations.")
    
    # Create a map of image_id -> image_info
    img_map = {img['id']: img for img in data['images']}
    
    # Create a map of image_id -> list of annotations
    ann_map = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann)
        
    # Select random samples
    sample_ids = random.sample(list(img_map.keys()), num_samples)
    
    for img_id in sample_ids:
        img_info = img_map[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        # Draw annotations
        if img_id in ann_map:
            for ann in ann_map[img_id]:
                cat_id = ann['category_id']
                # Polygon format [x1, y1, x2, y2, ...]
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    
                    # Color based on category (random color)
                    np.random.seed(cat_id)
                    color = np.random.randint(0, 255, 3).tolist()
                    
                    cv2.polylines(img, [poly], True, color, 2)
                    
                    # Optional: Fill
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [poly], color)
                    alpha = 0.4
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        output_filename = f"vis_{dataset_type}_{img_id}.png"
        cv2.imwrite(output_filename, img)
        print(f"Saved visualization to {output_filename}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(project_root, "dataset")
    visualize_dataset(base_path, "syntax")
    visualize_dataset(base_path, "stenosis")
