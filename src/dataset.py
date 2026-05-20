import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, dataset_type='syntax', split='train', image_size=(512, 512), transform=None):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.split = split
        self.image_size = image_size
        
        self.images_dir = os.path.join(root_dir, dataset_type, split, 'images')
        self.json_path = os.path.join(root_dir, dataset_type, split, 'annotations', f'{split}.json')
        
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
            
        self.image_info = {img['id']: img for img in self.data['images']}
        self.annotations = {}
        for ann in self.data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
            
        self.image_ids = list(self.image_info.keys())

        # Define Albumentations transforms
        if transform is not None:
            self.transform = transform
        else:
            if split == 'train':
                self.transform = A.Compose([
                    A.Resize(height=image_size[0], width=image_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.5),
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.8), # Strong CLAHE for angiograms
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3), # Deformations simulate vessels
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(height=image_size[0], width=image_size[1]),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), # Consistent CLAHE for val/test
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_data = self.image_info[img_id]
        
        # Load Image
        img_path = os.path.join(self.images_dir, img_data['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create Mask
        mask = np.zeros((img_data['height'], img_data['width']), dtype=np.uint8)
        if img_id in self.annotations:
            for ann in self.annotations[img_id]:
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1) # Binary mask (1 for foreground)
                    
        # Apply Albumentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        mask = mask.long() # H, W
        
        return image, mask

if __name__ == '__main__':
    # Test the dataset
    dataset = SegmentationDataset('dataset', split='train')
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Unique mask values: {torch.unique(mask)}")
