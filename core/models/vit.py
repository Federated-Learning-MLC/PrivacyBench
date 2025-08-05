from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from transformers import ViTForImageClassification

from .base import BaseModel

# Import from legacy code
try:
    from legacy.local_utility import LightningModel
    from legacy.config import NUM_CLASSES
except ImportError as e:
    print(f"Warning: Could not import legacy modules: {e}")
    NUM_CLASSES = 4


class ViTModel(BaseModel):
    """
    Vision Transformer Model wrapper.
    Wraps existing pb repo ViT implementation using HuggingFace transformers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.architecture = 'vit'
        self.model_name = 'vit-base-patch16-224'
        self.num_classes = config.get('num_classes', NUM_CLASSES)
        self.model_checkpoint = config.get('checkpoint', 'google/vit-base-patch16-224-in21k')
        
    def create_model(self) -> nn.Module:
        """Create ViT model exactly as in pb repo notebooks."""
        try:
            # Use exact same ViT model as pb repo
            model = ViTForImageClassification.from_pretrained(
                self.model_checkpoint,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True  # Allow different number of classes
            )
            
            self.model = model
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to create ViT model: {e}")
    
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
            print(f"Warning: Could not create legacy LightningModel: {e}")
            return self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ViT model information."""
        if self.model is None:
            self.model = self.create_model()
        
        return {
            'architecture': 'ViT-Base/16',
            'type': 'Vision Transformer',
            'checkpoint': self.model_checkpoint,
            'num_classes': self.num_classes,
            'total_parameters': self.get_parameter_count(),
            'trainable_parameters': self.get_trainable_parameters(),
            'input_size': (3, 224, 224),
            'patch_size': 16,
            'framework': 'HuggingFace Transformers + Lightning'
        }
    
    def prepare_for_privacy(self, privacy_technique: str) -> nn.Module:
        """Prepare ViT model for privacy techniques."""
        if self.model is None:
            self.model = self.create_model()
        
        if privacy_technique == 'differential_privacy':
            # ViT models may need special handling for DP
            try:
                from opacus.validators import ModuleValidator
                if not ModuleValidator.is_valid(self.model):
                    # ViT models often need custom fixes for Opacus
                    print("Warning: ViT model may not be fully compatible with Opacus DP")
                    # Apply available fixes
                    self.model = ModuleValidator.fix(self.model)
            except ImportError:
                print("Warning: Opacus not available for DP model validation")
        
        elif privacy_technique == 'secure_multiparty_computation':
            # SMPC with ViT may have additional constraints
            print("Warning: ViT with SMPC may have performance implications")
        
        return self.model
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze/unfreeze ViT backbone for fine-tuning."""
        if self.model is None:
            self.model = self.create_model()
        
        # Freeze all parameters except classifier head
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = not freeze
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from ViT model (for visualization)."""
        if self.model is None:
            return None
        
        # This would require model to be in eval mode and process an input
        # Implementation depends on specific use case
        return None