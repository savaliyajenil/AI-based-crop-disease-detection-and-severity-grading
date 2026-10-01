import os
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class PlantSegDataset(Dataset):
    """
    PyTorch Dataset for loading the PlantSeg dataset for Semantic Segmentation.
    Expects the directory structure:
        Dataset/plantseg/
            images/
                train/
                val/
                test/
            annotations/
                train/
                val/
                test/
    """
    def __init__(self, root_dir, split="train", target_plants=None, transforms=None):
        """
        Args:
            root_dir (str): Path to the `plantseg` directory.
            split (str): One of 'train', 'val', or 'test'.
            target_plants (list of str): List of plants to filter by (e.g. ['Potato', 'Tomato']).
                                         If None, loads all plants.
            transforms (albumentations.Compose): Data augmentation pipeline.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        
        self.img_dir = self.root_dir / "images" / split
        self.mask_dir = self.root_dir / "annotations" / split
        self.coco_file = self.root_dir / f"annotation_{split}.json"
        
        # Load COCO annotations
        with open(self.coco_file, 'r') as f:
            self.coco_data = json.load(f)
            
        # Filter images by target plants if specified
        self.images = []
        for img in self.coco_data['images']:
            if target_plants:
                # Check if any target plant name is in the filename (case insensitive)
                fname = img['file_name'].lower()
                if any(plant.lower() in fname for plant in target_plants):
                    self.images.append(img)
            else:
                self.images.append(img)
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_name = img_info['file_name']
        
        # Load image
        img_path = self.img_dir / img_name
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (PNG grayscale mask where pixel value is category ID)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = self.mask_dir / mask_name
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # If PNG mask is missing, fallback to rendering from COCO polygons
            h, w = img_info['height'], img_info['width']
            mask = np.zeros((h, w), dtype=np.uint8)
            img_id = img_info['id']
            # Find annotations for this image
            anns = [a for a in self.coco_data['annotations'] if a['image_id'] == img_id]
            for ann in anns:
                cat_id = ann['category_id']
                for seg in ann['segmentation']:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], int(cat_id))
                    
        # Apply augmentations (Albumentations handles image and mask together)
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask is a long tensor (for CrossEntropyLoss)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
            
        return image, mask

if __name__ == "__main__":
    # Simple test script
    from transforms import get_training_augmentation
    
    dataset_path = r"D:\PDEU Research\Dataset\plantseg"
    if os.path.exists(dataset_path):
        transforms = get_training_augmentation()
        ds = PlantSegDataset(dataset_path, split="train", target_plants=["Potato", "Tomato"], transforms=transforms)
        print(f"Loaded {len(ds)} training images for Potato/Tomato.")
        
        if len(ds) > 0:
            img, mask = ds[0]
            print(f"Image shape: {img.shape}, dtype: {img.dtype}")
            print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"Unique classes in mask: {torch.unique(mask)}")
