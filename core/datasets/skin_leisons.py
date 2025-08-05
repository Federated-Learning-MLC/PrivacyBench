from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader

from .base import BaseDataset

# Import from legacy code
try:
    from legacy.local_utility import MedicalImageDataModule
    from legacy.config import HEIGHT_WIDTH, NUM_WORKERS, AUGMENT
except ImportError as e:
    print(f"Warning: Could not import legacy modules: {e}")
    HEIGHT_WIDTH = (224, 224)
    NUM_WORKERS = 4
    AUGMENT = True


class SkinLesionsDataset(BaseDataset):
    """
    ISIC Skin Lesion Dataset wrapper.
    Wraps existing pb repo implementation for 8-class skin lesion classification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_name = 'skin_lesions'
        self.num_classes = 8  # 8 different skin lesion types
        
        # Use legacy constants if not specified
        self.batch_size = config.get('batch_size', 32)
        self.height_width = config.get('height_width', HEIGHT_WIDTH)
        self.num_workers = config.get('num_workers', NUM_WORKERS)
        self.augment_data = config.get('augment', AUGMENT)
        
        self.dm = None
        
    def load(self) -> None:
        """Load skin lesions dataset using legacy implementation."""
        try:
            self.dm = MedicalImageDataModule(
                data_name=self.data_name,
                batch_size=self.batch_size,
                height_width=self.height_width,
                num_workers=self.num_workers,
                augment_data=self.augment_data
            )
            self.dm.setup()
            
            self.dataset_stats = {
                'total_samples': len(self.dm.data_full) if hasattr(self.dm, 'data_full') else 'Unknown',
                'train_samples': len(self.dm.data_train) if hasattr(self.dm, 'data_train') else 'Unknown',
                'val_samples': len(self.dm.data_val) if hasattr(self.dm, 'data_val') else 'Unknown',
                'test_samples': len(self.dm.data_test) if hasattr(self.dm, 'data_test') else 'Unknown'
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load skin lesions dataset: {e}")
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create dataloaders using legacy implementation."""
        if self.dm is None:
            self.load()
        
        self.train_loader = self.dm.train_dataloader()
        self.val_loader = self.dm.val_dataloader() 
        self.test_loader = self.dm.test_dataloader()
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_transforms(self):
        """Get data transforms from legacy implementation.""" 
        if self.dm is None:
            self.load()
        
        return {
            'train_transforms': getattr(self.dm, 'train_transforms', None),
            'val_transforms': getattr(self.dm, 'val_transforms', None),
            'test_transforms': getattr(self.dm, 'test_transforms', None)
        }
    
    def get_num_classes(self) -> int:
        """Get number of classes for skin lesions dataset."""
        return self.num_classes
    
    def setup_federated_data(self, num_clients: int = 3) -> Dict[str, Any]:
        """Setup skin lesions data for federated learning."""
        if self.dm is None:
            self.load()
        
        try:
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
            return super().setup_federated_data(num_clients)
