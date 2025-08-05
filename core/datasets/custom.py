import os
from pathlib import Path
from typing import Any, Dict, Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

from .base import BaseDataset


class CustomDataset(BaseDataset):
    """
    Custom dataset wrapper for user-provided datasets.
    Supports common formats: directory structure, CSV files.
    Provides automatic class detection and validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = config.get('path', '')
        self.format = config.get('format', 'directory')  # 'directory' or 'csv'
        self.class_names = config.get('class_names', [])
        
        # Auto-detect number of classes if not specified
        self.num_classes = config.get('num_classes', 0)
        
    def load(self) -> None:
        """Load custom dataset with automatic format detection."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")
        
        if self.format == 'directory':
            self._load_directory_format()
        elif self.format == 'csv':
            self._load_csv_format()
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def _load_directory_format(self):
        """Load dataset from directory structure (ImageFolder format)."""
        try:
            # Auto-detect classes from directory structure
            data_path = Path(self.data_path)
            if (data_path / 'train').exists():
                train_path = data_path / 'train'
                val_path = data_path / 'val' if (data_path / 'val').exists() else None
                test_path = data_path / 'test' if (data_path / 'test').exists() else None
            else:
                # Single directory with class subdirectories
                train_path = data_path
                val_path = None
                test_path = None
            
            # Auto-detect classes
            if not self.class_names:
                self.class_names = [d.name for d in train_path.iterdir() if d.is_dir()]
                self.num_classes = len(self.class_names)
            
            # Create datasets
            transform = self._get_default_transforms()
            
            self.train_dataset = datasets.ImageFolder(train_path, transform=transform)
            
            if val_path:
                self.val_dataset = datasets.ImageFolder(val_path, transform=transform)
            else:
                # Split train set for validation
                train_size = int(0.8 * len(self.train_dataset))
                val_size = len(self.train_dataset) - train_size
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.train_dataset, [train_size, val_size]
                )
            
            if test_path:
                self.test_dataset = datasets.ImageFolder(test_path, transform=transform)
            else:
                # Use validation set as test set
                self.test_dataset = self.val_dataset
            
            self.dataset_stats = {
                'class_names': self.class_names,
                'train_samples': len(self.train_dataset),
                'val_samples': len(self.val_dataset),
                'test_samples': len(self.test_dataset)
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load directory format dataset: {e}")
    
    def _load_csv_format(self):
        """Load dataset from CSV file format."""
        # Implementation for CSV-based datasets
        # This would require pandas and custom Dataset class
        raise NotImplementedError("CSV format support coming in future update")
    
    def _get_default_transforms(self):
        """Get default transforms for custom datasets."""
        if self.augment_data:
            return transforms.Compose([
                transforms.Resize(self.height_width),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.height_width),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create dataloaders for custom dataset."""
        if not hasattr(self, 'train_dataset'):
            self.load()
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_transforms(self):
        """Get transforms for custom dataset."""
        return {
            'train_transforms': self._get_default_transforms(),
            'val_transforms': self._get_default_transforms(),
            'test_transforms': self._get_default_transforms()
        }
    
    def get_num_classes(self) -> int:
        """Get number of classes for custom dataset."""
        if self.num_classes == 0 and hasattr(self, 'train_dataset'):
            # Try to auto-detect from dataset
            if hasattr(self.train_dataset, 'classes'):
                self.num_classes = len(self.train_dataset.classes)
        return self.num_classes