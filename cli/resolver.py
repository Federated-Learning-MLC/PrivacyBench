"""
Configuration resolver
Handles configuration dependencies, defaults, and inheritance
"""
from typing import Dict, Any, Optional
from pathlib import Path
import copy


class ConfigResolver:
    """Resolves configuration dependencies and applies defaults"""
    
    def __init__(self):
        self.default_configs = self._load_default_configs()
    
    def resolve_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve configuration by applying defaults and resolving dependencies"""
        resolved_config = copy.deepcopy(config)
        
        # Apply default configurations
        resolved_config = self._apply_defaults(resolved_config)
        
        # Resolve dependencies
        resolved_config = self._resolve_dependencies(resolved_config)
        
        # Apply conditional logic
        resolved_config = self._apply_conditional_logic(resolved_config)
        
        return resolved_config
    
    def _load_default_configs(self) -> Dict[str, Any]:
        """Load default configuration values"""
        return {
            "dataset": {
                "config": {
                    "augmentation": True,
                    "test_split": 0.08,
                    "validation_split": 0.1,
                    "height_width": 224,
                    "num_workers": 4
                }
            },
            "model": {
                "config": {
                    "pretrained": True,
                    "dropout": 0.1
                }
            },
            "training": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.0002,
                "optimizer": "adam",
                "tolerance": 7,
                "seed": 42
            },
            "resources": {
                "gpu": True,
                "num_workers": 4,
                "memory_limit": "8GB"
            },
            "tracking": {
                "wandb": {
                    "project": "PrivacyBench"
                },
                "energy": {
                    "track_emissions": True,
                    "country_iso_code": "USA"
                }
            },
            "output": {
                "save_model": True,
                "export_formats": ["json", "csv"]
            }
        }
    
    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to missing configuration fields"""
        for section, defaults in self.default_configs.items():
            if section not in config:
                config[section] = copy.deepcopy(defaults)
            else:
                # Merge defaults with existing config
                config[section] = self._merge_dicts(copy.deepcopy(defaults), config[section])
        
        return config
    
    def _merge_dicts(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries"""
        result = copy.deepcopy(default)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _resolve_dependencies(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve configuration dependencies"""
        
        # Resolve dataset-dependent configurations
        dataset_name = config.get("dataset", {}).get("name")
        if dataset_name:
            config = self._resolve_dataset_dependencies(config, dataset_name)
        
        # Resolve model-dependent configurations
        model_arch = config.get("model", {}).get("architecture")
        if model_arch:
            config = self._resolve_model_dependencies(config, model_arch)
        
        # Resolve privacy-dependent configurations
        privacy_techniques = config.get("privacy", {}).get("techniques", [])
        if privacy_techniques:
            config = self._resolve_privacy_dependencies(config, privacy_techniques)
        
        return config
    
    def _resolve_dataset_dependencies(self, config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """Resolve dataset-specific dependencies"""
        
        # Set num_classes based on dataset
        if dataset_name == "alzheimer":
            if "model" not in config:
                config["model"] = {}
            if "config" not in config["model"]:
                config["model"]["config"] = {}
            config["model"]["config"]["num_classes"] = 4
            
        elif dataset_name == "skin_lesions":
            if "model" not in config:
                config["model"] = {}
            if "config" not in config["model"]:
                config["model"]["config"] = {}
            config["model"]["config"]["num_classes"] = 8
        
        return config
    
    def _resolve_model_dependencies(self, config: Dict[str, Any], model_arch: str) -> Dict[str, Any]:
        """Resolve model-specific dependencies"""
        
        # Adjust learning rate for different architectures
        if model_arch == "vit":
            # ViT typically needs smaller learning rate
            if config["training"]["learning_rate"] == 0.0002:  # If using default
                config["training"]["learning_rate"] = 0.00001
                
        # Adjust batch size for GPU memory constraints
        if model_arch == "vit":
            if config["training"]["batch_size"] == 32:  # If using default
                config["training"]["batch_size"] = 16  # ViT requires more memory
        
        return config
    
    def _resolve_privacy_dependencies(self, config: Dict[str, Any], techniques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve privacy technique dependencies"""
        
        technique_names = [t.get("name") for t in techniques]
        
        # Adjust training parameters for differential privacy
        if "differential_privacy" in technique_names:
            # DP typically needs more epochs for convergence
            if config["training"]["epochs"] == 50:  # If using default
                config["training"]["epochs"] = 100
                
            # Smaller batch sizes work better with DP
            if config["training"]["batch_size"] > 16:
                config["training"]["batch_size"] = 16
        
        # Adjust parameters for federated learning
        if "federated_learning" in technique_names:
            # FL typically needs more total epochs distributed across rounds
            fl_config = next((t for t in techniques if t.get("name") == "federated_learning"), {})
            fl_params = fl_config.get("config", {})
            
            num_rounds = fl_params.get("num_rounds", 5)
            if config["training"]["epochs"] == 50:  # If using default
                # Distribute epochs across rounds
                config["training"]["epochs"] = max(10, 50 // num_rounds)
        
        return config
    
    def _apply_conditional_logic(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conditional configuration logic"""
        
        # Disable GPU if not available or requested
        if not config["resources"]["gpu"]:
            config["training"]["batch_size"] = min(config["training"]["batch_size"], 16)
            config["resources"]["num_workers"] = min(config["resources"]["num_workers"], 2)
        
        # Adjust output directory based on experiment name
        if "metadata" in config and "output" in config:
            exp_name = config["metadata"].get("name", "unknown")
            base_dir = Path(config["output"]["directory"])
            config["output"]["directory"] = str(base_dir / exp_name)
        
        # Enable additional tracking for privacy experiments
        privacy_techniques = config.get("privacy", {}).get("techniques", [])
        if privacy_techniques:
            config["tracking"]["privacy_audit"] = True
            config["output"]["export_formats"] = list(set(
                config["output"]["export_formats"] + ["privacy_report"]
            ))
        
        return config
    
    def validate_resolved_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate the resolved configuration for logical consistency"""
        issues = []
        
        # Check for conflicting parameters
        if config.get("resources", {}).get("gpu") and config.get("training", {}).get("batch_size", 0) > 64:
            issues.append("Large batch size with GPU may cause out-of-memory errors")
        
        # Check privacy technique compatibility
        privacy_techniques = [t.get("name") for t in config.get("privacy", {}).get("techniques", [])]
        if "differential_privacy" in privacy_techniques and "federated_learning" in privacy_techniques:
            # FL+DP combination requires careful parameter tuning
            dp_config = next((t for t in config["privacy"]["techniques"] 
                            if t.get("name") == "differential_privacy"), {})
            epsilon = dp_config.get("config", {}).get("epsilon", 1.0)
            if epsilon > 10:
                issues.append("High epsilon value in FL+DP combination may provide insufficient privacy")
        
        return issues
    
    def create_experiment_id(self, config: Dict[str, Any]) -> str:
        """Create a unique experiment identifier"""
        from datetime import datetime
        
        # Get key components
        dataset = config.get("dataset", {}).get("name", "unknown")
        model = config.get("model", {}).get("architecture", "unknown")
        privacy_techniques = [t.get("name", "") for t in config.get("privacy", {}).get("techniques", [])]
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build experiment ID
        privacy_str = "_".join(privacy_techniques) if privacy_techniques else "baseline"
        exp_id = f"{model}_{dataset}_{privacy_str}_{timestamp}"
        
        return exp_id
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """Generate a human-readable configuration summary"""
        summary_lines = []
        
        # Basic info
        metadata = config.get("metadata", {})
        summary_lines.append(f"Experiment: {metadata.get('name', 'Unknown')}")
        summary_lines.append(f"Dataset: {config.get('dataset', {}).get('name', 'Unknown')}")
        summary_lines.append(f"Model: {config.get('model', {}).get('architecture', 'Unknown')}")
        
        # Privacy techniques
        techniques = config.get("privacy", {}).get("techniques", [])
        if techniques:
            technique_names = [t.get("name", "Unknown") for t in techniques]
            summary_lines.append(f"Privacy: {', '.join(technique_names)}")
        else:
            summary_lines.append("Privacy: None (Baseline)")
        
        # Training parameters
        training = config.get("training", {})
        summary_lines.append(f"Epochs: {training.get('epochs', 'Unknown')}")
        summary_lines.append(f"Batch Size: {training.get('batch_size', 'Unknown')}")
        summary_lines.append(f"Learning Rate: {training.get('learning_rate', 'Unknown')}")
        
        return "\n".join(summary_lines)