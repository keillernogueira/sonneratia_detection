import cv2
import json

import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


class GeoObjectDetectionDataset(Dataset):
    """
    PyTorch Dataset for object detection from georeferenced patches
    """
    
    def __init__(self, annotations_file, images_dir, transforms=None):
        """
        Parameters:
        -----------
        annotations_file : str
            Path to annotations JSON file
        images_dir : str
            Directory containing patch images
        transforms : albumentations.Compose
            Augmentation transforms
        """
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        print(len(self.annotations))
        
        self.images_dir = Path(images_dir)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Load annotation
        ann = self.annotations[idx]
        
        # Load image
        img_path = self.images_dir / ann['filename']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get bboxes and labels
        boxes = []
        labels = []
        
        for bbox_data in ann['bboxes']:
            bbox = bbox_data['bbox']  # [x1, y1, x2, y2]
            boxes.append(bbox)
            labels.append(bbox_data['class'])
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Convert to tensor format expected by detection models
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([ann['image_id']]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((boxes.shape[0],), dtype=torch.int64)
        }
        
        return image, target


def get_transforms(train=True):
    """Get augmentation transforms"""
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def collate_fn(batch):
    """Custom collate function for object detection"""
    return tuple(zip(*batch))

