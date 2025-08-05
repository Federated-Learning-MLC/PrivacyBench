from .base import BaseModel
from .cnn import CNNModel
from .vit import ViTModel

# Auto-register models with registry
from core.registry import registry

registry.register_model('cnn', CNNModel)
registry.register_model('vit', ViTModel)

__all__ = ['BaseModel', 'CNNModel', 'ViTModel']