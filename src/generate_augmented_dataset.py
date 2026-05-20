import os
import cv2
import json
import numpy as np
import albumentations as A
from tqdm import tqdm
import shutil

# Paths
ROOT_DIR = '/home/samadhi21eng055/Desktop/Research/Medical-Image-Segmentation-/dataset'
DATASET_TYPE = 'syntax'
SPLIT = 'train'
OUTPUT_DIR = '/home/samadhi21eng055/Desktop/Research/Medical-Image-Segmentation-/dataset_augmented'

# Augmentation factor
AUG_FACTOR = 5 # Create 5 new images per original image

def create_output_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, DATASET_TYPE, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, DATASET_TYPE, 'train', 'annotations'), exist_ok=True)
    
    # Validation data should not be augmented usually, just copied
    os.makedirs(os.path.join(OUTPUT_DIR, DATASET_TYPE, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, DATASET_TYPE, 'test', 'annotations'), exist_ok=True)


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.Affine(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.8),
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.8), 
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5), 
])


def augment_dataset(split='train'):
    print(f"Augmenting {split} split...")
    images_dir = os.path.join(ROOT_DIR, DATASET_TYPE, split, 'images')
    json_path = os.path.join(ROOT_DIR, DATASET_TYPE, split, 'annotations', f'{split}.json')
    
    out_images_dir = os.path.join(OUTPUT_DIR, DATASET_TYPE, split, 'images')
    out_json_path = os.path.join(OUTPUT_DIR, DATASET_TYPE, split, 'annotations', f'{split}.json')
    
    if not os.path.exists(json_path):
        print(f"No json found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    image_info = {img['id']: img for img in data['images']}
    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)
        
    new_data = {
        'images': [],
        'annotations': [],
        'categories': data.get('categories', [{'id': 1, 'name': 'vessel', 'supercategory': 'vessel'}])
    }
    
    new_img_id = 0
    new_ann_id = 0
    
    for img_id in tqdm(image_info.keys()):
        img_data = image_info[img_id]
        img_path = os.path.join(images_dir, img_data['file_name'])
        image = cv2.imread(img_path)
        
        if image is None:
            continue
            
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Original mask
        mask = np.zeros((img_data['height'], img_data['width']), dtype=np.uint8)
        if img_id in annotations:
            for ann in annotations[img_id]:
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)
        
        # Save Original Image
        cv2.imwrite(os.path.join(out_images_dir, f"{new_img_id}.jpg"), image)
        
        new_data['images'].append({
            'id': new_img_id,
            'file_name': f"{new_img_id}.jpg",
            'width': img_data['width'],
            'height': img_data['height']
        })
        
        # Re-convert mask to polys for JSON (simple approach: find contours)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 2:
                segmentation = contour.flatten().tolist()
                new_data['annotations'].append({
                    'id': new_ann_id,
                    'image_id': new_img_id,
                    'category_id': 1,
                    'segmentation': [segmentation],
                    'area': cv2.contourArea(contour),
                    'bbox': cv2.boundingRect(contour),
                    'iscrowd': 0
                })
                new_ann_id += 1
                
        new_img_id += 1
        
        # Generate augmented images
        for aug_idx in range(AUG_FACTOR):
            augmented = transform(image=original_image, mask=mask)
            aug_img = augmented['image']
            aug_mask = augmented['mask']
            
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_images_dir, f"{new_img_id}.jpg"), aug_img_bgr)
            
            new_data['images'].append({
                'id': new_img_id,
                'file_name': f"{new_img_id}.jpg",
                'width': img_data['width'],
                'height': img_data['height']
            })
            
            aug_contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in aug_contours:
                 if len(contour) > 2:
                    segmentation = contour.flatten().tolist()
                    new_data['annotations'].append({
                        'id': new_ann_id,
                        'image_id': new_img_id,
                        'category_id': 1,
                        'segmentation': [segmentation],
                        'area': cv2.contourArea(contour),
                        'bbox': cv2.boundingRect(contour),
                        'iscrowd': 0
                    })
                    new_ann_id += 1
                    
            new_img_id += 1

    with open(out_json_path, 'w') as f:
        json.dump(new_data, f)
        
    print(f"Saved {len(new_data['images'])} images to {out_json_path}")


def copy_test_dataset():
    print("Copying test split as is...")
    src_images = os.path.join(ROOT_DIR, DATASET_TYPE, 'test', 'images')
    src_json = os.path.join(ROOT_DIR, DATASET_TYPE, 'test', 'annotations', 'test.json')
    
    dst_images = os.path.join(OUTPUT_DIR, DATASET_TYPE, 'test', 'images')
    dst_json = os.path.join(OUTPUT_DIR, DATASET_TYPE, 'test', 'annotations', 'test.json')
    
    if os.path.exists(src_images):
        for f in os.listdir(src_images):
            shutil.copy(os.path.join(src_images, f), os.path.join(dst_images, f))
            
    if os.path.exists(src_json):
        shutil.copy(src_json, dst_json)

if __name__ == '__main__':
    create_output_dirs()
    augment_dataset('train')
    copy_test_dataset()

