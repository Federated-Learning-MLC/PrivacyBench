from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch


class BasePrivacy(ABC):
    """Abstract base class for all privacy technique components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.technique_name = config.get('name', '')
        self.enabled = config.get('enabled', True)
        
        # Privacy-specific configurations
        self.privacy_params = config.get('config', {})
        
    @abstractmethod
    def setup(self, model: torch.nn.Module, dataloader, **kwargs) -> Dict[str, Any]:
        """Setup privacy technique with model and data."""
        pass
    
    @abstractmethod
    def apply_privacy(self, **kwargs) -> Any:
        """Apply privacy technique to training process."""
        pass
    
    @abstractmethod
    def get_privacy_metrics(self) -> Dict[str, Any]:
        """Get privacy-specific metrics and parameters."""
        pass
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate privacy technique configuration."""
        errors = []
        
        if not self.technique_name:
            errors.append("Privacy technique name is required")
        
        return len(errors) == 0, errors
    
    def get_overhead_estimate(self) -> Dict[str, Any]:
        """Get estimated computational overhead."""
        return {
            'time_overhead': '10-50%',
            'memory_overhead': '5-20%', 
            'communication_overhead': '0%'
        }