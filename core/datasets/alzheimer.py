import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader

from .base import BaseDataset

# Import from legacy code (pb repo implementation)
try:
    from legacy.local_utility import MedicalImageDataModule, load_data
    from legacy.config import NUM_CLASSES, HEIGHT_WIDTH, NUM_WORKERS, AUGMENT
except ImportError as e:
    print(f"Warning: Could not import legacy modules: {e}")
    # Fallback constants
    NUM_CLASSES = 4
    HEIGHT_WIDTH = (224, 224) 
    NUM_WORKERS = 4
    AUGMENT = True


class AlzheimerDataset(BaseDataset):
    """
    Alzheimer MRI Dataset wrapper.
    Wraps existing pb repo MedicalImageDataModule for Alzheimer classification.
    Preserves 100% compatibility with notebook implementations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_name = 'alzheimer'
        self.num_classes = 4  # NonDemented, VeryMildDemented, MildDemented, ModerateDemented
        
        # Use legacy constants if not specified in config
        self.batch_size = config.get('batch_size', 32)
        self.height_width = config.get('height_width', HEIGHT_WIDTH)
        self.num_workers = config.get('num_workers', NUM_WORKERS)
        self.augment_data = config.get('augment', AUGMENT)
        
        self.dm = None  # Will hold MedicalImageDataModule instance
        
    def load(self) -> None:
        """Load Alzheimer dataset using legacy implementation."""
        try:
            # Use exact same DataModule as pb repo notebooks
            self.dm = MedicalImageDataModule(
                data_name=self.data_name,
                batch_size=self.batch_size,
                height_width=self.height_width,
                num_workers=self.num_workers,
                augment_data=self.augment_data
            )
            self.dm.setup()
            
            # Store dataset statistics
            self.dataset_stats = {
                'total_samples': len(self.dm.data_full) if hasattr(self.dm, 'data_full') else 'Unknown',
                'train_samples': len(self.dm.data_train) if hasattr(self.dm, 'data_train') else 'Unknown',
                'val_samples': len(self.dm.data_val) if hasattr(self.dm, 'data_val') else 'Unknown',
                'test_samples': len(self.dm.data_test) if hasattr(self.dm, 'data_test') else 'Unknown'
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Alzheimer dataset: {e}")
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create dataloaders using legacy implementation."""
        if self.dm is None:
            self.load()
        
        # Use exact same dataloaders as pb repo
        self.train_loader = self.dm.train_dataloader()
        self.val_loader = self.dm.val_dataloader()
        self.test_loader = self.dm.test_dataloader()
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_transforms(self):
        """Get data transforms from legacy implementation."""
        if self.dm is None:
            self.load()
        
        # Return transforms used by legacy DataModule
        return {
            'train_transforms': getattr(self.dm, 'train_transforms', None),
            'val_transforms': getattr(self.dm, 'val_transforms', None),
            'test_transforms': getattr(self.dm, 'test_transforms', None)
        }
    
    def get_num_classes(self) -> int:
        """Get number of classes for Alzheimer dataset."""
        return self.num_classes
    
    def setup_federated_data(self, num_clients: int = 3) -> Dict[str, Any]:
        """Setup Alzheimer data for federated learning."""
        if self.dm is None:
            self.load()
        
        try:
            # Use legacy federated data preparation if available
            from legacy.federated import prepare_FL_dataset
            
            fl_data = prepare_FL_dataset(
                data_name=self.data_name,
                num_clients=num_clients,
                batch_size=self.batch_size
            )
            
            return {
                'num_clients': num_clients,
                'client_data': fl_data.get('client_data', {}),
                'server_data': self.test_loader,
                'federation_config': fl_data
            }
            
        except ImportError:
            # Fallback to basic client simulation
            return super().setup_federated_data(num_clients)
