from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader


class BaseDataset(ABC):
    """Abstract base class for all dataset components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_name = config.get('name', '')
        self.batch_size = config.get('batch_size', 32)
        self.height_width = config.get('height_width', (224, 224))
        self.num_workers = config.get('num_workers', 4)
        self.augment_data = config.get('augment', True)
        
        # Initialize data modules
        self.train_loader = None
        self.val_loader = None  
        self.test_loader = None
        self.num_classes = None
        self.dataset_stats = {}
    
    @abstractmethod
    def load(self) -> None:
        """Load and prepare dataset."""
        pass
    
    @abstractmethod
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders."""
        pass
    
    @abstractmethod
    def get_transforms(self):
        """Get data transforms for preprocessing."""
        pass
    
    @abstractmethod
    def get_num_classes(self) -> int:
        """Get number of classes in dataset."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'num_classes': self.get_num_classes(),
            'batch_size': self.batch_size,
            'height_width': self.height_width,
            'augmentation': self.augment_data,
            **self.dataset_stats
        }
    
    def setup_federated_data(self, num_clients: int = 3) -> Dict[str, Any]:
        """Setup data for federated learning simulation."""
        # Default implementation - to be overridden by specific datasets
        return {
            'num_clients': num_clients,
            'client_data': {},
            'server_data': self.test_loader
        }
