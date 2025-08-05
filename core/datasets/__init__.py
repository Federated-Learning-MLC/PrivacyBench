from .base import BaseDataset
from .alzheimer import AlzheimerDataset  
from .skin_lesions import SkinLesionsDataset
from .custom import CustomDataset

# Auto-register datasets with registry
from core.registry import registry

registry.register_dataset('alzheimer', AlzheimerDataset)
registry.register_dataset('skin_lesions', SkinLesionsDataset)
registry.register_dataset('custom', CustomDataset)

__all__ = ['BaseDataset', 'AlzheimerDataset', 'SkinLesionsDataset', 'CustomDataset']
