from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .base import BaseModel

# Import from legacy code (pb repo implementation)
try:
    from legacy.local_utility import LightningModel
    from legacy.config import NUM_CLASSES, SEED
except ImportError as e:
    print(f"Warning: Could not import legacy modules: {e}")
    NUM_CLASSES = 4
    SEED = 42


class CNNModel(BaseModel):
    """
    CNN Model wrapper using ResNet18 architecture.
    Wraps existing pb repo implementation to preserve exact model behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.architecture = 'cnn'
        self.model_name = 'resnet18'
        self.num_classes = config.get('num_classes', NUM_CLASSES)
        self.pretrained = config.get('pretrained', True)
        
    def create_model(self) -> nn.Module:
        """Create ResNet18 model exactly as in pb repo notebooks."""
        try:
            # Use exact same model creation as pb repo
            if self.pretrained:
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = resnet18(weights=None)
            
            # Replace final layer for our number of classes
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            
            self.model = model
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to create CNN model: {e}")
    
    def create_lightning_model(self, **kwargs) -> Any:
        """Create Lightning model wrapper using legacy implementation."""
        try:
            if self.model is None:
                self.model = self.create_model()
            
            # Use exact same LightningModel as pb repo
            self.lightning_model = LightningModel(
                model=self.model,
                num_classes=self.num_classes,
                **kwargs
            )
            
            return self.lightning_model
            
        except Exception as e:
            # Fallback if legacy LightningModel not available
            print(f"Warning: Could not create legacy LightningModel: {e}")
            return self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get CNN model information."""
        if self.model is None:
            self.model = self.create_model()
        
        return {
            'architecture': 'ResNet18',
            'type': 'CNN',
            'pretrained': self.pretrained,
            'num_classes': self.num_classes,
            'total_parameters': self.get_parameter_count(),
            'trainable_parameters': self.get_trainable_parameters(),
            'input_size': (3, 224, 224),
            'framework': 'PyTorch + Lightning'
        }
    
    def prepare_for_privacy(self, privacy_technique: str) -> nn.Module:
        """Prepare model for privacy techniques (DP, SMPC)."""
        if self.model is None:
            self.model = self.create_model()
        
        if privacy_technique == 'differential_privacy':
            # Make model compatible with Opacus
            try:
                from opacus.validators import ModuleValidator
                if not ModuleValidator.is_valid(self.model):
                    self.model = ModuleValidator.fix(self.model)
            except ImportError:
                print("Warning: Opacus not available for DP model validation")
        
        elif privacy_technique == 'secure_multiparty_computation':
            # Prepare for SMPC (may require model modifications)
            pass
        
        return self.model