"""
Configuration parser that converts CLI args + YAML configs
Maps your existing experiments.yaml to CLI interface
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Safe imports with error handling
try:
    from legacy.local_utility import load_yaml_config
    LEGACY_UTILITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import legacy utilities: {e}")
    LEGACY_UTILITY_AVAILABLE = False
    
    def load_yaml_config(yaml_path=None, key=None, item_name=None):
        """Fallback when legacy utilities aren't available."""
        print(f"Warning: Using fallback for load_yaml_config")
        return []

try:
    from legacy.config import ExperimentName
    LEGACY_CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import legacy config: {e}")
    LEGACY_CONFIG_AVAILABLE = False
    
    # Fallback enum-like class
    class ExperimentName:
        """Fallback for ExperimentName enum."""
        CNN_BASE = "CNN Baseline"
        VIT_BASE = "ViT Baseline"
        FL_CNN = "FL (CNN)"
        FL_VIT = "FL (ViT)"
        DP_CNN = "DP (CNN)"
        DP_VIT = "DP (ViT)"
        FL_CDP_SF_CNN = "FL + CDP-SF (CNN)"
        FL_SMPC_CNN = "FL + SMPC (CNN)"


class ConfigParser:
    """Parses CLI arguments and YAML configurations"""
    
    def __init__(self):
        self.legacy_experiments_path = Path(__file__).parent.parent / "legacy" / "experiments.yaml"
        
    def parse_cli_to_config(self, args) -> Dict[str, Any]:
        """Convert CLI arguments to experiment configuration"""
        
        if args.config:
            # Load custom YAML config
            return self._load_custom_config(args.config)
        else:
            # Generate config from CLI args using your existing system
            return self._generate_config_from_cli(args)
    
    def _generate_config_from_cli(self, args) -> Dict[str, Any]:
        """Generate configuration from CLI arguments using existing experiments.yaml"""
        
        # Map CLI experiment names to your ACTUAL ExperimentName enum values
        experiment_mapping = {
            # Baseline experiments
            "cnn_baseline": ExperimentName.CNN_BASE,
            "cnn_base": ExperimentName.CNN_BASE,
            "vit_baseline": ExperimentName.VIT_BASE,
            "vit_base": ExperimentName.VIT_BASE,
            
            # Federated Learning experiments
            "fl_cnn": ExperimentName.FL_CNN,
            "fl_vit": ExperimentName.FL_VIT,
            
            # Differential Privacy experiments
            "dp_cnn": ExperimentName.DP_CNN,
            "dp_vit": ExperimentName.DP_VIT,
            
            # FL + SMPC combinations
            "fl_smpc_cnn": ExperimentName.FL_SMPC_CNN,
            "fl_smpc_vit": ExperimentName.FL_SMPC_VIT,
            "smpc_cnn": ExperimentName.FL_SMPC_CNN,
            
            # FL + DP combinations (FIXED: FL_CDP_SF_CNN not FL_CCDP_SF_CNN)
            "fl_dp_cnn": ExperimentName.FL_CDP_SF_CNN,
            "fl_cdp_sf_cnn": ExperimentName.FL_CDP_SF_CNN,
            "fl_cdp_sf_vit": ExperimentName.FL_CDP_SF_VIT,
            "fl_cdp_sa_cnn": ExperimentName.FL_CDP_SA_CNN,
            "fl_cdp_sa_vit": ExperimentName.FL_CDP_SA_VIT,
            "fl_cdp_cf_cnn": ExperimentName.FL_CDP_CF_CNN,
            "fl_cdp_cf_vit": ExperimentName.FL_CDP_CF_VIT,
            "fl_cdp_ca_cnn": ExperimentName.FL_CDP_CA_CNN,
            "fl_cdp_ca_vit": ExperimentName.FL_CDP_CA_VIT,
            "fl_ldp_mod_cnn": ExperimentName.FL_LDP_MOD_CNN,
            "fl_ldp_mod_vit": ExperimentName.FL_LDP_MOD_VIT,
            "fl_ldp_pe_cnn": ExperimentName.FL_LDP_PE_CNN,
            "fl_ldp_pe_vit": ExperimentName.FL_LDP_PE_VIT,
        }
        
        experiment_name = experiment_mapping.get(args.experiment)
        if not experiment_name:
            available_experiments = list(experiment_mapping.keys())
            raise ValueError(f"Unknown experiment: {args.experiment}. Available: {available_experiments}")
        
        # Load base config - FIXED: Use direct YAML loading instead of item_name parameter
        try:
            # First try to load the experiments.yaml directly
            base_config = self._load_experiment_config(experiment_name.value)
        except Exception as e:
            print(f"Warning: Could not load experiment config for '{experiment_name.value}': {e}")
            # Create a fallback config
            base_config = self._create_fallback_config(experiment_name, args)
        
        # Convert to new CLI format
        return self._convert_legacy_config(base_config, args, experiment_name)
    
    def _load_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """Load specific experiment configuration from experiments.yaml"""
        try:
            # Load the entire experiments.yaml file
            with open(self.legacy_experiments_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            # Handle different YAML structures
            if isinstance(yaml_content, dict):
                # Structure: { "experiments": [...] }
                if "experiments" in yaml_content:
                    experiments_list = yaml_content["experiments"]
                    # Find experiment by name
                    for exp in experiments_list:
                        if isinstance(exp, dict) and exp.get("name") == experiment_name:
                            return exp
                    raise ValueError(f"Experiment '{experiment_name}' not found in experiments list")
                
                # Structure: { "experiment_name": {...}, ... }
                elif experiment_name in yaml_content:
                    return yaml_content[experiment_name]
                else:
                    # Search through all values for matching name
                    for key, value in yaml_content.items():
                        if isinstance(value, dict) and value.get("name") == experiment_name:
                            return value
                    raise ValueError(f"Experiment '{experiment_name}' not found in YAML")
            
            # Structure: [...] (direct list)
            elif isinstance(yaml_content, list):
                for exp in yaml_content:
                    if isinstance(exp, dict) and exp.get("name") == experiment_name:
                        return exp
                raise ValueError(f"Experiment '{experiment_name}' not found in experiments list")
            
            else:
                raise ValueError(f"Unsupported YAML structure: {type(yaml_content)}")
                
        except FileNotFoundError:
            raise ValueError(f"Experiments file not found: {self.legacy_experiments_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    
    def _create_fallback_config(self, experiment_name, args) -> Dict[str, Any]:
        """Create fallback configuration when experiments.yaml loading fails"""
        # Extract base config from your research results
        fallback_configs = {
            "CNN Baseline": {
                "name": "CNN Baseline",
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.0002,
                "optimizer": "adam",
                "tolerance": 7
            },
            "ViT Baseline": {
                "name": "ViT Baseline", 
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.0002,
                "optimizer": "adam",
                "tolerance": 7
            },
            "FL (CNN)": {
                "name": "FL (CNN)",
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.0002,
                "num_clients": 3,
                "num_rounds": 5
            },
            "FL (ViT)": {
                "name": "FL (ViT)",
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.0002,
                "num_clients": 3,
                "num_rounds": 5
            }
        }
        
        return fallback_configs.get(experiment_name.value, {
            "name": experiment_name.value,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.0002
        })
    
    def _convert_legacy_config(self, legacy_config: Dict[str, Any], args, experiment_name) -> Dict[str, Any]:
        """Convert your existing config format to new CLI format"""
        
        # Extract privacy techniques from experiment name
        privacy_techniques = self._extract_privacy_config(args.experiment, legacy_config, experiment_name)
        
        # Determine model architecture
        model_architecture = "cnn" if "cnn" in args.experiment.lower() else "vit"
        
        return {
            "metadata": {
                "name": f"{args.experiment}_{args.dataset}",
                "experiment_type": args.experiment,
                "dataset": args.dataset,
                "source": "cli_generated",
                "original_experiment": legacy_config.get("name", experiment_name.value)
            },
            "dataset": {
                "name": args.dataset,
                "config": {
                    "augmentation": True,
                    "test_split": 0.08,
                    "validation_split": 0.1,
                    "height_width": 224,
                    "num_workers": 4
                }
            },
            "model": {
                "architecture": model_architecture,
                "config": {
                    "name": model_architecture,
                    "pretrained": True,
                    "num_classes": 4 if args.dataset == "alzheimer" else 8,
                    "dropout": 0.1
                }
            },
            "privacy": privacy_techniques,
            "training": {
                "epochs": legacy_config.get("epochs", 50),
                "batch_size": legacy_config.get("batch_size", 32),
                "learning_rate": legacy_config.get("learning_rate", 0.0002),
                "optimizer": legacy_config.get("optimizer", "adam"),
                "tolerance": legacy_config.get("tolerance", 7),
                "seed": 42
            },
            "output": {
                "directory": getattr(args, 'output', Path("./results")),
                "save_model": True,
                "export_formats": ["json", "csv"]
            }
        }
    
    def _extract_privacy_config(self, experiment: str, legacy_config: Dict[str, Any], experiment_name) -> Dict[str, Any]:
        """Extract privacy configuration from experiment name and legacy config"""
        techniques = []
        
        # Check for Federated Learning
        if "fl" in experiment or "FL" in experiment_name.value:
            techniques.append({
                "name": "federated_learning",
                "config": {
                    "num_clients": legacy_config.get("num_clients", 3),
                    "num_rounds": legacy_config.get("num_rounds", 5),
                    "strategy": "FedAvg"
                }
            })
        
        # Check for Differential Privacy
        if "dp" in experiment or "DP" in experiment_name.value or "CDP" in experiment_name.value or "LDP" in experiment_name.value:
            techniques.append({
                "name": "differential_privacy", 
                "config": {
                    "epsilon": legacy_config.get("epsilon", 1.0),
                    "delta": legacy_config.get("delta", 1e-5),
                    "noise_multiplier": legacy_config.get("noise_multiplier", 1.0),
                    "max_grad_norm": legacy_config.get("max_grad_norm", 1.0)
                }
            })
        
        # Check for SMPC
        if "smpc" in experiment or "SMPC" in experiment_name.value:
            techniques.append({
                "name": "secure_multiparty_computation",
                "config": {
                    "protocol": "SecAgg",
                    "threshold": 2
                }
            })
        
        return {
            "techniques": techniques
        }
    
    def _load_custom_config(self, config_path: Path) -> Dict[str, Any]:
        """Load custom YAML configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def get_available_experiments(self) -> Dict[str, Any]:
        """Get all available experiments from your existing experiments.yaml"""
        try:
            with open(self.legacy_experiments_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            # Handle different YAML structures
            if isinstance(yaml_content, dict) and "experiments" in yaml_content:
                return yaml_content["experiments"]
            elif isinstance(yaml_content, list):
                return yaml_content
            else:
                return yaml_content
        except Exception as e:
            print(f"Warning: Could not load experiments.yaml: {e}")
            return []
    
    def validate_experiment_dataset_combination(self, experiment: str, dataset: str) -> bool:
        """Validate that experiment and dataset combination is supported"""
        valid_combinations = {
            "alzheimer": [
                "cnn_baseline", "vit_baseline", 
                "fl_cnn", "fl_vit",
                "dp_cnn", "dp_vit",
                "fl_dp_cnn", "fl_smpc_cnn"
            ],
            "skin_lesions": [
                "cnn_baseline", "vit_baseline", 
                "fl_cnn", "fl_vit",
                "dp_cnn", "dp_vit",
                "fl_dp_cnn", "fl_smpc_cnn"
            ]
        }
        
        return experiment in valid_combinations.get(dataset, [])