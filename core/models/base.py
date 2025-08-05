from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class BaseModel(ABC):
    """Abstract base class for all model components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_classes = config.get('num_classes', 2)
        self.architecture = config.get('architecture', '')
        self.pretrained = config.get('pretrained', True)
        
        # Model instance will be created by subclasses
        self.model = None
        self.lightning_model = None
        self.training_config = {}
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create and return the PyTorch model."""
        pass
    
    @abstractmethod
    def create_lightning_model(self, **kwargs) -> Any:
        """Create and return the Lightning model wrapper."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        pass
    
    def setup_training(self, training_config: Dict[str, Any]) -> None:
        """Setup training configuration."""
        self.training_config = training_config
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters in model."""
        if self.model is None:
            self.model = self.create_model()
        return sum(p.numel() for p in self.model.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        if self.model is None:
            self.model = self.create_model()
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


