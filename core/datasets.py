"""
PrivacyBench Dataset Wrappers
Wraps existing legacy/local_utility.py dataset functions into modular components.
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional, List
from abc import abstractmethod
import torch
from torch.utils.data import DataLoader

# Add project root to path for legacy imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.registry import BaseComponent, ComponentType, register_component

try:
    # Import legacy dataset functions
    from legacy.local_utility import (
        load_yaml_config,
        load_alzheimer_data,
        load_skin_lesions_data
    )
    LEGACY_AVAILABLE = True
except ImportError:
    print("⚠️  Legacy dataset functions not available - using fallback implementations")
    LEGACY_AVAILABLE = False

class BaseDataset(BaseComponent):
    """Base class for all dataset wrappers."""
    
    @property
    def component_type(self) -> ComponentType:
        return ComponentType.DATASET
    
    @abstractmethod
    def load_data(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Load and return train_loader, test_loader, and metadata."""
        pass
    
    @abstractmethod
    def simulate_clients(self, num_clients: int) -> List[DataLoader]:
        """Simulate federated learning clients."""
        pass
    
    def validate_config(self) -> bool:
        """Validate dataset configuration."""
        required_keys = ['name', 'config']
        return all(key in self.config for key in required_keys)

@register_component(
    name="alzheimer",
    component_type=ComponentType.DATASET,
    description="Alzheimer's disease dataset from OASIS",
    supported_models=["cnn", "vit"]
)
class AlzheimerDataset(BaseDataset):
    """Wrapper for Alzheimer's dataset functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_config = config.get('config', {})
        self.data_path = self.dataset_config.get('data_path', '../data/alzheimer/')
        
    def load_data(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Load Alzheimer's dataset using legacy function."""
        
        if LEGACY_AVAILABLE:
            try:
                # Use legacy dataset loading
                train_loader, test_loader, metadata = load_alzheimer_data(
                    data_path=self.data_path,
                    batch_size=self.dataset_config.get('batch_size', 32),
                    test_split=self.dataset_config.get('test_split', 0.08),
                    augmentation=self.dataset_config.get('augmentation', True),
                    num_workers=self.dataset_config.get('num_workers', 4)
                )
                
                print(f"✅ Loaded Alzheimer dataset: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
                
                return train_loader, test_loader, metadata
                
            except Exception as e:
                print(f"❌ Error loading Alzheimer dataset: {e}")
                return self._fallback_data_loading()
        else:
            return self._fallback_data_loading()
    
    def simulate_clients(self, num_clients: int = 5) -> List[DataLoader]:
        """Simulate federated learning clients for Alzheimer's dataset."""
        
        if LEGACY_AVAILABLE:
            try:
                # Use existing federated data simulation logic
                train_loader, _, _ = self.load_data()
                
                # Split dataset among clients (IID for now)
                dataset = train_loader.dataset
                client_size = len(dataset) // num_clients
                
                client_loaders = []
                for i in range(num_clients):
                    start_idx = i * client_size
                    end_idx = start_idx + client_size if i < num_clients - 1 else len(dataset)
                    
                    client_dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
                    client_loader = DataLoader(
                        client_dataset,
                        batch_size=self.dataset_config.get('batch_size', 32),
                        shuffle=True,
                        num_workers=self.dataset_config.get('num_workers', 4)
                    )
                    client_loaders.append(client_loader)
                
                print(f"✅ Created {num_clients} federated clients for Alzheimer dataset")
                return client_loaders
                
            except Exception as e:
                print(f"❌ Error simulating clients: {e}")
                return self._fallback_client_simulation(num_clients)
        else:
            return self._fallback_client_simulation(num_clients)
    
    def _fallback_data_loading(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Fallback data loading when legacy functions aren't available."""
        print("⚠️  Using fallback Alzheimer dataset simulation")
        
        # Create dummy data for testing
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        # Simulate 1000 samples with 3x224x224 images, 4 classes
        X_train = torch.randn(800, 3, 224, 224)
        y_train = torch.randint(0, 4, (800,))
        X_test = torch.randn(200, 3, 224, 224)
        y_test = torch.randint(0, 4, (200,))
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        metadata = {
            'num_classes': 4,
            'input_shape': (3, 224, 224),
            'dataset_size': {'train': 800, 'test': 200}
        }
        
        return train_loader, test_loader, metadata
    
    def _fallback_client_simulation(self, num_clients: int) -> List[DataLoader]:
        """Fallback client simulation."""
        train_loader, _, _ = self._fallback_data_loading()
        return [train_loader] * num_clients  # Simple duplication for testing

@register_component(
    name="skin_lesions", 
    component_type=ComponentType.DATASET,
    description="Skin lesions dataset (HAM10000)",
    supported_models=["cnn", "vit"]
)
class SkinLesionsDataset(BaseDataset):
    """Wrapper for Skin Lesions dataset functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_config = config.get('config', {})
        self.data_path = self.dataset_config.get('data_path', '../data/skin_lesions/')
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Load Skin Lesions dataset using legacy function."""
        
        if LEGACY_AVAILABLE:
            try:
                # Use legacy dataset loading
                train_loader, test_loader, metadata = load_skin_lesions_data(
                    data_path=self.data_path,
                    batch_size=self.dataset_config.get('batch_size', 32),
                    test_split=self.dataset_config.get('test_split', 0.08),
                    augmentation=self.dataset_config.get('augmentation', True),
                    num_workers=self.dataset_config.get('num_workers', 4)
                )
                
                print(f"✅ Loaded Skin Lesions dataset: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
                
                return train_loader, test_loader, metadata
                
            except Exception as e:
                print(f"❌ Error loading Skin Lesions dataset: {e}")
                return self._fallback_data_loading()
        else:
            return self._fallback_data_loading()
    
    def simulate_clients(self, num_clients: int = 5) -> List[DataLoader]:
        """Simulate federated learning clients for Skin Lesions dataset."""
        
        if LEGACY_AVAILABLE:
            try:
                # Similar client simulation logic as Alzheimer
                train_loader, _, _ = self.load_data()
                
                dataset = train_loader.dataset
                client_size = len(dataset) // num_clients
                
                client_loaders = []
                for i in range(num_clients):
                    start_idx = i * client_size
                    end_idx = start_idx + client_size if i < num_clients - 1 else len(dataset)
                    
                    client_dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
                    client_loader = DataLoader(
                        client_dataset,
                        batch_size=self.dataset_config.get('batch_size', 32),
                        shuffle=True,
                        num_workers=self.dataset_config.get('num_workers', 4)
                    )
                    client_loaders.append(client_loader)
                
                print(f"✅ Created {num_clients} federated clients for Skin Lesions dataset")
                return client_loaders
                
            except Exception as e:
                print(f"❌ Error simulating clients: {e}")
                return self._fallback_client_simulation(num_clients)
        else:
            return self._fallback_client_simulation(num_clients)
    
    def _fallback_data_loading(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Fallback data loading when legacy functions aren't available."""
        print("⚠️  Using fallback Skin Lesions dataset simulation")
        
        # Create dummy data for testing
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        # Simulate 2000 samples with 3x224x224 images, 7 classes (HAM10000 has 7 classes)
        X_train = torch.randn(1600, 3, 224, 224)
        y_train = torch.randint(0, 7, (1600,))
        X_test = torch.randn(400, 3, 224, 224)
        y_test = torch.randint(0, 7, (400,))
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        metadata = {
            'num_classes': 7,
            'input_shape': (3, 224, 224),
            'dataset_size': {'train': 1600, 'test': 400}
        }
        
        return train_loader, test_loader, metadata
    
    def _fallback_client_simulation(self, num_clients: int) -> List[DataLoader]:
        """Fallback client simulation."""
        train_loader, _, _ = self._fallback_data_loading()
        return [train_loader] * num_clients  # Simple duplication for testing

# Dataset factory function
def create_dataset(dataset_name: str, config: Dict[str, Any]) -> BaseDataset:
    """Factory function to create dataset instances."""
    from core.registry import registry
    
    return registry.create_component(ComponentType.DATASET, dataset_name, config)