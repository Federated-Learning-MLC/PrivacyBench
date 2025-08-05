from typing import Dict, List, Any, Optional
from .registry import registry


class PipelineManager:
    """Manages component pipeline configuration and setup."""
    
    def __init__(self):
        self.registry = registry
        self.pipeline_config = {}
        
    def setup_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup complete pipeline from configuration."""
        pipeline = {}
        
        # Resolve dataset component
        dataset_config = config.get('dataset', {})
        dataset_name = dataset_config.get('name')
        if dataset_name:
            dataset_class = self.registry.get_dataset(dataset_name)
            pipeline['dataset'] = {
                'class': dataset_class,
                'config': dataset_config
            }
        
        # Resolve model component
        model_config = config.get('model', {})
        model_arch = model_config.get('architecture') 
        if model_arch:
            model_class = self.registry.get_model(model_arch)
            pipeline['model'] = {
                'class': model_class,
                'config': model_config
            }
        
        # Resolve privacy components
        privacy_config = config.get('privacy', {})
        techniques = privacy_config.get('techniques', [])
        if techniques:
            pipeline['privacy'] = []
            for technique in techniques:
                technique_name = technique.get('name')
                if technique_name:
                    privacy_class = self.registry.get_privacy(technique_name)
                    pipeline['privacy'].append({
                        'class': privacy_class,
                        'config': technique
                    })
        
        # Resolve metrics tracker
        if 'tracking' in config:
            metrics_class = self.registry.get_metrics('default')
            pipeline['metrics'] = {
                'class': metrics_class,
                'config': config.get('tracking', {})
            }
        
        self.pipeline_config = pipeline
        return pipeline
    
    def validate_pipeline(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate pipeline configuration."""
        errors = []
        
        # Check required components
        if 'dataset' not in config:
            errors.append("Dataset configuration is required")
        elif 'name' not in config['dataset']:
            errors.append("Dataset name is required")
        elif not self.registry.validate_component('dataset', config['dataset']['name']):
            errors.append(f"Unknown dataset: {config['dataset']['name']}")
        
        if 'model' not in config:
            errors.append("Model configuration is required")
        elif 'architecture' not in config['model']:
            errors.append("Model architecture is required")
        elif not self.registry.validate_component('model', config['model']['architecture']):
            errors.append(f"Unknown model architecture: {config['model']['architecture']}")
        
        # Check privacy techniques if specified
        privacy_config = config.get('privacy', {})
        techniques = privacy_config.get('techniques', [])
        for technique in techniques:
            technique_name = technique.get('name')
            if technique_name and not self.registry.validate_component('privacy', technique_name):
                errors.append(f"Unknown privacy technique: {technique_name}")
        
        return len(errors) == 0, errors
    
    def get_resource_requirements(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource requirements for pipeline."""
        requirements = {
            'gpu_memory': '2GB',  # Base requirement
            'cpu_cores': 4,
            'ram': '8GB',
            'estimated_time': '10 minutes'
        }
        
        # Adjust based on model architecture
        model_arch = config.get('model', {}).get('architecture', '')
        if 'vit' in model_arch.lower():
            requirements['gpu_memory'] = '4GB'
            requirements['estimated_time'] = '30 minutes'
        
        # Adjust based on privacy techniques
        privacy_techniques = config.get('privacy', {}).get('techniques', [])
        for technique in privacy_techniques:
            technique_name = technique.get('name', '')
            if 'federated_learning' in technique_name:
                requirements['estimated_time'] = '45 minutes'
            elif 'differential_privacy' in technique_name:
                requirements['gpu_memory'] = '3GB'
            elif 'secure_multiparty_computation' in technique_name:
                requirements['estimated_time'] = '60 minutes'
        
        return requirements