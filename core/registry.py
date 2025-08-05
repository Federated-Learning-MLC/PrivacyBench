import importlib
import inspect
from typing import Dict, List, Any, Type, Optional
from abc import ABC, abstractmethod


class ComponentRegistry:
    """Central registry for all PrivacyBench components."""
    
    def __init__(self):
        self._datasets: Dict[str, Type] = {}
        self._models: Dict[str, Type] = {}
        self._privacy: Dict[str, Type] = {}
        self._metrics: Dict[str, Type] = {}
        
        # Auto-discover and register built-in components
        self._auto_discover_components()
    
    def _auto_discover_components(self):
        """Auto-discover and register built-in components."""
        try:
            # Import component modules to trigger registration
            from core.datasets import AlzheimerDataset, SkinLesionsDataset
            from core.models import CNNModel, ViTModel  
            from core.privacy import FederatedLearning, DifferentialPrivacy, SMPC
            from core.tracking import MetricsTracker
        except ImportError as e:
            # Components not yet implemented - this is expected during development
            pass
    
    # Dataset Registry Methods
    def register_dataset(self, name: str, dataset_class: Type):
        """Register a dataset component."""
        self._datasets[name] = dataset_class
        
    def get_dataset(self, name: str) -> Type:
        """Get dataset class by name."""
        if name not in self._datasets:
            raise ValueError(f"Dataset '{name}' not found. Available: {list(self._datasets.keys())}")
        return self._datasets[name]
    
    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        return list(self._datasets.keys())
    
    # Model Registry Methods  
    def register_model(self, name: str, model_class: Type):
        """Register a model component."""
        self._models[name] = model_class
        
    def get_model(self, name: str) -> Type:
        """Get model class by name."""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]
    
    def list_models(self) -> List[str]:
        """List all available models."""
        return list(self._models.keys())
    
    # Privacy Registry Methods
    def register_privacy(self, name: str, privacy_class: Type):
        """Register a privacy technique component."""
        self._privacy[name] = privacy_class
        
    def get_privacy(self, name: str) -> Type:
        """Get privacy technique class by name."""
        if name not in self._privacy:
            raise ValueError(f"Privacy technique '{name}' not found. Available: {list(self._privacy.keys())}")
        return self._privacy[name]
    
    def list_privacy(self) -> List[str]:
        """List all available privacy techniques."""
        return list(self._privacy.keys())
    
    # Metrics Registry Methods
    def register_metrics(self, name: str, metrics_class: Type):
        """Register a metrics tracker component."""
        self._metrics[name] = metrics_class
        
    def get_metrics(self, name: str) -> Type:
        """Get metrics tracker class by name."""
        if name not in self._metrics:
            raise ValueError(f"Metrics tracker '{name}' not found. Available: {list(self._metrics.keys())}")
        return self._metrics[name]
    
    def list_metrics(self) -> List[str]:
        """List all available metrics trackers."""
        return list(self._metrics.keys())
    
    # Component Validation
    def validate_component(self, component_type: str, name: str) -> bool:
        """Validate that a component exists and is properly registered."""
        registries = {
            'dataset': self._datasets,
            'model': self._models, 
            'privacy': self._privacy,
            'metrics': self._metrics
        }
        
        if component_type not in registries:
            return False
            
        return name in registries[component_type]
    
    def list_available(self) -> Dict[str, List[str]]:
        """List all available components by type."""
        return {
            'datasets': self.list_datasets(),
            'models': self.list_models(),
            'privacy': self.list_privacy(), 
            'metrics': self.list_metrics()
        }


# Global registry instance
registry = ComponentRegistry()