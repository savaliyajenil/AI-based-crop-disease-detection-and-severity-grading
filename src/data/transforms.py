import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_training_augmentation(image_size=(256, 256)):
    """
    Returns the data augmentation pipeline for training.
    We use heavy augmentations (flips, rotations, color jitter, noise) 
    to prevent overfitting on our limited potato/tomato datasets.
    """
    return A.Compose([
        # Geometric transformations
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, 
                           border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
        
        # Color & Lighting transformations (simulate field conditions)
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        
        # Noise & Blurring (simulate poor camera quality)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
        ], p=0.2),
        
        # Normalization for ImageNet weights (used by most transfer learning models)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_validation_augmentation(image_size=(256, 256)):
    """
    Returns the pipeline for validation and testing.
    Only resizes and normalizes the images.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
