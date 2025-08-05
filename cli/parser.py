"""
Configuration Parser - Phase 1 with Safe Legacy Imports
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import sys

# Safe legacy imports with fallbacks
try:
    from legacy.local_utility import load_yaml_config
    LEGACY_UTILITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import legacy utilities: {e}")
    LEGACY_UTILITY_AVAILABLE = False
    
    # Fallback function
    def load_yaml_config(key=None):
        """Fallback when legacy utilities aren't available."""
        print(f"Warning: Using fallback for load_yaml_config (key: {key})")
        return {}

try:
    from legacy.config import ExperimentName
    LEGACY_CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import legacy config: {e}")
    LEGACY_CONFIG_AVAILABLE = False
    
    # Fallback enum-like class
    class ExperimentName:
        """Fallback for ExperimentName enum."""
        CNN_BASELINE = "cnn_baseline"
        VIT_BASELINE = "vit_baseline"
        FL_CNN = "fl_cnn"
        FL_VIT = "fl_vit"
        # Add other experiment names as needed


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
        
        # Map CLI experiment names to your existing ExperimentName enum
        experiment_mapping = {
            # CNN experiments
            "cnn_baseline": ExperimentName.CNN_BASE,
            "cnn_base": ExperimentName.CNN_BASE,
            
            # ViT experiments  
            "vit_baseline": ExperimentName.VIT_BASE,
            "vit_base": ExperimentName.VIT_BASE,
            
            # Federated Learning experiments
            "fl_cnn": ExperimentName.FL_CNN,
            "fl_vit": ExperimentName.FL_VIT,
            "fl_cnn_base": ExperimentName.FL_CNN,
            "fl_vit_base": ExperimentName.FL_VIT,
            
            # Differential Privacy experiments
            "dp_cnn": ExperimentName.DP_CNN,
            "dp_vit": ExperimentName.DP_VIT,
            "dp_cnn_base": ExperimentName.DP_CNN,
            "dp_vit_base": ExperimentName.DP_VIT,
            
            # Hybrid combinations
            "fl_dp_cnn": ExperimentName.FL_CCDP_SF_CNN,
            "fl_smpc_cnn": ExperimentName.FL_SMPC_CNN,
            "fl_cdp_sf_cnn": ExperimentName.FL_CCDP_SF_CNN,
            
            # SMPC experiments
            "smpc_cnn": ExperimentName.SMPC_CNN,
            "smpc_vit": ExperimentName.SMPC_VIT,
        }
        
        experiment_name = experiment_mapping.get(args.experiment)
        if not experiment_name:
            raise ValueError(f"Unknown experiment: {args.experiment}")
        
        # Load base config from your existing experiments.yaml
        try:
            base_config = load_yaml_config(
                yaml_path=self.legacy_experiments_path,
                key="experiments",
                item_name=experiment_name.value
            )
        except Exception as e:
            raise ValueError(f"Failed to load experiment '{experiment_name.value}' from experiments.yaml: {e}")
        
        # Convert to new CLI format
        return self._convert_legacy_config(base_config, args)
    
    def _convert_legacy_config(self, legacy_config: Dict[str, Any], args) -> Dict[str, Any]:
        """Convert your existing config format to new CLI format"""
        
        # Extract privacy techniques from experiment name
        privacy_techniques = self._extract_privacy_config(args.experiment, legacy_config)
        
        # Determine model architecture
        model_architecture = "cnn" if "cnn" in args.experiment.lower() else "vit"
        
        return {
            "metadata": {
                "name": f"{args.experiment}_{args.dataset}",
                "experiment_type": args.experiment,
                "dataset": args.dataset,
                "source": "cli_generated",
                "original_experiment": legacy_config.get("name", "Unknown")
            },
            "dataset": {
                "name": args.dataset,
                "config": {
                    "augmentation": True,  # From your existing setup
                    "test_split": 0.08,    # From your existing setup
                    "validation_split": 0.1,
                    "height_width": 224,   # From your existing HEIGHT_WIDTH
                    "num_workers": 4       # From your existing NUM_WORKERS
                }
            },
            "model": {
                "architecture": model_architecture,
                "config": {
                    "pretrained": True,
                    "num_classes": 4 if args.dataset == "alzheimer" else 8,  # From your existing NUM_CLASSES logic
                    "dropout": legacy_config.get("dropout", 0.1),
                    **legacy_config  # Include all existing hyperparameters
                }
            },
            "privacy": privacy_techniques,
            "training": {
                "epochs": legacy_config.get("epochs", 50),
                "batch_size": legacy_config.get("batch_size", 32),
                "learning_rate": legacy_config.get("learning_rate", 0.0002),
                "optimizer": legacy_config.get("optimizer", "adam"),
                "tolerance": legacy_config.get("tolerance", 7),  # Early stopping patience
                "seed": getattr(args, 'seed', 42)
            },
            "resources": {
                "gpu": getattr(args, 'gpu', True),
                "num_workers": 4,
                "memory_limit": "8GB"
            },
            "tracking": {
                "wandb": {
                    "project": "PrivacyBench",
                    "entity": "your-entity"  # Update with your W&B entity
                },
                "energy": {
                    "track_emissions": True,
                    "country_iso_code": "USA"
                }
            },
            "output": {
                "directory": str(args.output),
                "save_model": True,
                "export_formats": ["json", "csv"]
            }
        }
    
    def _extract_privacy_config(self, experiment: str, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract privacy configuration from experiment name"""
        
        techniques = []
        
        # Check for federated learning
        if "fl" in experiment:
            techniques.append({
                "name": "federated_learning",
                "config": {
                    "num_clients": 3,      # From your existing FL setup
                    "num_rounds": 5,       # From your existing FL setup
                    "strategy": "FedAvg"   # Default strategy
                }
            })
        
        # Check for differential privacy
        if "dp" in experiment:
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
        if "smpc" in experiment:
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
            experiments = load_yaml_config(
                yaml_path=self.legacy_experiments_path,
                key="experiments"
            )
            return experiments
        except Exception as e:
            print(f"Warning: Could not load experiments.yaml: {e}")
            return []
    
    def validate_experiment_dataset_combination(self, experiment: str, dataset: str) -> bool:
        """Validate that experiment and dataset combination is supported"""
        # Add any specific validation logic here
        valid_combinations = {
            "alzheimer": ["cnn_baseline", "vit_baseline", "fl_cnn", "fl_vit", "dp_cnn", "dp_vit"],
            "skin_lesions": ["cnn_baseline", "vit_baseline", "fl_cnn", "fl_vit", "dp_cnn", "dp_vit"]
        }
        
        return experiment in valid_combinations.get(dataset, [])