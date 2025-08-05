import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import Phase 1 functionality (preserved)
try:
    from legacy.config import ExperimentName
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False
    ExperimentName = None

# Phase 1 experiment mapping (preserved)
experiment_mapping = {
    # Baselines
    "cnn_baseline": ExperimentName.CNN_BASE if LEGACY_AVAILABLE else "CNN_BASE",
    "vit_baseline": ExperimentName.VIT_BASE if LEGACY_AVAILABLE else "VIT_BASE",
    # Federated Learning
    "fl_cnn": ExperimentName.FL_CNN if LEGACY_AVAILABLE else "FL_CNN",
    "fl_vit": ExperimentName.FL_VIT if LEGACY_AVAILABLE else "FL_VIT",
    # Differential Privacy
    "dp_cnn": ExperimentName.DP_CNN if LEGACY_AVAILABLE else "DP_CNN",
    "dp_vit": ExperimentName.DP_VIT if LEGACY_AVAILABLE else "DP_VIT",
    # FL + SMPC
    "fl_smpc_cnn": ExperimentName.FL_SMPC_CNN if LEGACY_AVAILABLE else "FL_SMPC_CNN",
    "fl_smpc_vit": ExperimentName.FL_SMPC_VIT if LEGACY_AVAILABLE else "FL_SMPC_VIT",
    # FL + DP Variants (8 different combinations)
    "fl_cdp_sf_cnn": ExperimentName.FL_CDP_SF_CNN if LEGACY_AVAILABLE else "FL_CDP_SF_CNN",
    "fl_cdp_sf_vit": ExperimentName.FL_CDP_SF_VIT if LEGACY_AVAILABLE else "FL_CDP_SF_VIT",
    "fl_cdp_sa_cnn": ExperimentName.FL_CDP_SA_CNN if LEGACY_AVAILABLE else "FL_CDP_SA_CNN",
    "fl_cdp_sa_vit": ExperimentName.FL_CDP_SA_VIT if LEGACY_AVAILABLE else "FL_CDP_SA_VIT",
    "fl_cdp_cf_cnn": ExperimentName.FL_CDP_CF_CNN if LEGACY_AVAILABLE else "FL_CDP_CF_CNN",
    "fl_cdp_cf_vit": ExperimentName.FL_CDP_CF_VIT if LEGACY_AVAILABLE else "FL_CDP_CF_VIT",
    "fl_cdp_ca_cnn": ExperimentName.FL_CDP_CA_CNN if LEGACY_AVAILABLE else "FL_CDP_CA_CNN",
    "fl_cdp_ca_vit": ExperimentName.FL_CDP_CA_VIT if LEGACY_AVAILABLE else "FL_CDP_CA_VIT",
    # Additional experiments
    "smpc_cnn": "SMPC_CNN",
    "smpc_vit": "SMPC_VIT",
    "fl_dp_cnn": "FL_DP_CNN",
    "fl_dp_vit": "FL_DP_VIT",
}


def parse_experiment_config(args) -> Dict[str, Any]:
    """
    Enhanced configuration parser supporting both methods:
    1. Legacy: CLI args + experiments.yaml (Phase 1)
    2. Individual: --config flag with individual YAML files (Phase 3)
    """
    
    # Method 1: Individual config file (Phase 3)
    if hasattr(args, 'config') and args.config:
        return parse_individual_config(args.config, args)
    
    # Method 2: Legacy CLI args (Phase 1) - preserved functionality
    return parse_legacy_config(args)


def parse_individual_config(config_path: str, args) -> Dict[str, Any]:
    """Parse individual experiment configuration file (Phase 3)."""
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Try relative to configs/experiments/
        configs_dir = Path("configs/experiments")
        potential_paths = [
            configs_dir / config_path,
            configs_dir / f"{config_path}.yaml",
            configs_dir / "baselines" / f"{config_path}.yaml",
            configs_dir / "federated" / f"{config_path}.yaml", 
            configs_dir / "privacy" / f"{config_path}.yaml",
            configs_dir / "hybrid" / f"{config_path}.yaml"
        ]
        
        for path in potential_paths:
            if path.exists():
                config_file = path
                break
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML configuration
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_file}: {e}")
    
    # Handle multi-experiment files (like fl_cnn_configurations.yaml)
    if isinstance(config, dict) and len(config) == 1 and list(config.keys())[0] != 'experiment':
        # This is likely a multi-experiment file, extract the first experiment
        experiment_key = list(config.keys())[0]
        config = config[experiment_key]
    
    # Override with command line arguments if provided
    config = override_with_cli_args(config, args)
    
    # Validate required fields
    validate_individual_config(config, config_file)
    
    return config


def parse_legacy_config(args) -> Dict[str, Any]:
    """Parse legacy CLI arguments (Phase 1 functionality preserved)."""
    
    # Get experiment configuration
    experiment_name = args.experiment
    dataset_name = args.dataset
    
    # Map CLI experiment name to legacy enum
    experiment_enum = experiment_mapping.get(experiment_name)
    if not experiment_enum:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    # Load legacy experiments.yaml if available
    legacy_config = load_legacy_experiments_yaml()
    
    # Build configuration from CLI args
    config = {
        "experiment": experiment_name,
        "experiment_enum": experiment_enum,
        "dataset": {
            "name": dataset_name,
            "batch_size": getattr(args, 'batch_size', 32),
            "augment": getattr(args, 'augment', True),
            "height_width": [224, 224],
            "num_workers": 4
        },
        "model": {
            "architecture": extract_model_arch(experiment_name),
            "pretrained": True,
            "num_classes": 4 if dataset_name == 'alzheimer' else 8
        },
        "training": {
            "epochs": getattr(args, 'epochs', 50),
            "learning_rate": getattr(args, 'learning_rate', 0.00025),
            "batch_size": getattr(args, 'batch_size', 32),
            "seed": 42,
            "patience": 10
        },
        "output": {
            "directory": getattr(args, 'output', './results'),
            "save_model": True,
            "formats": ["json", "csv"]
        },
        "tracking": {
            "wandb": True,
            "codecarbon": True,
            "log_level": "INFO"
        }
    }
    
    # Add privacy configuration based on experiment type
    config["privacy"] = detect_privacy_techniques(experiment_name)
    
    # Override with legacy YAML config if available
    if legacy_config and experiment_name in legacy_config:
        legacy_exp_config = legacy_config[experiment_name]
        config = merge_configs(config, legacy_exp_config)
    
    return config


def load_legacy_experiments_yaml() -> Optional[Dict[str, Any]]:
    """Load legacy experiments.yaml file (Phase 1 functionality)."""
    
    legacy_paths = [
        "legacy/experiments.yaml",
        "src/experiments.yaml",  # Fallback to original location
        "experiments.yaml"
    ]
    
    for path in legacy_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
    
    return None


def override_with_cli_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Override config with command line arguments."""
    
    # Dataset overrides
    if hasattr(args, 'dataset') and args.dataset:
        config.setdefault('dataset', {})['name'] = args.dataset
    
    if hasattr(args, 'batch_size') and args.batch_size:
        config.setdefault('dataset', {})['batch_size'] = args.batch_size
        config.setdefault('training', {})['batch_size'] = args.batch_size
    
    # Training overrides  
    if hasattr(args, 'epochs') and args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    
    # Output overrides
    if hasattr(args, 'output') and args.output:
        config.setdefault('output', {})['directory'] = args.output
    
    # Verbose/logging overrides
    if hasattr(args, 'verbose') and args.verbose:
        config.setdefault('tracking', {})['log_level'] = 'DEBUG'
    
    return config


def validate_individual_config(config: Dict[str, Any], config_file: Path) -> None:
    """Validate individual configuration file structure."""
    
    required_fields = ['experiment', 'dataset', 'model']
    missing_fields = []
    
    for field in required_fields:
        if field not in config:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Missing required fields in {config_file}: {missing_fields}")
    
    # Validate dataset
    if 'name' not in config['dataset']:
        raise ValueError(f"Dataset name required in {config_file}")
    
    # Validate model
    if 'architecture' not in config['model']:
        raise ValueError(f"Model architecture required in {config_file}")


def extract_model_arch(experiment_name: str) -> str:
    """Extract model architecture from experiment name."""
    if 'cnn' in experiment_name:
        return 'cnn'
    elif 'vit' in experiment_name:
        return 'vit'
    else:
        return 'cnn'  # Default


def detect_privacy_techniques(experiment_name: str) -> Dict[str, Any]:
    """Detect privacy techniques from experiment name (Phase 1 functionality)."""
    
    techniques = []
    
    if 'fl' in experiment_name:
        techniques.append({
            "name": "federated_learning",
            "config": {
                "num_clients": 3,
                "num_rounds": 5,
                "client_fraction": 1.0,
                "min_clients": 3,
                "strategy": "FedAvg"
            }
        })
    
    if 'dp' in experiment_name:
        techniques.append({
            "name": "differential_privacy", 
            "config": {
                "epsilon": 1.0,
                "delta": 1e-5,
                "max_grad_norm": 1.0
            }
        })
    
    if 'smpc' in experiment_name:
        techniques.append({
            "name": "secure_multiparty_computation",
            "config": {
                "num_parties": 3,
                "threshold": 2,
                "protocol": "secret_sharing"
            }
        })
    
    return {"techniques": techniques}


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries."""
    
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def list_available_configs() -> Dict[str, List[str]]:
    """List all available configuration files by category (Phase 3)."""
    
    configs_dir = Path("configs/experiments")
    available = {
        "baselines": [],
        "federated": [],
        "privacy": [], 
        "hybrid": [],
        "legacy": list(experiment_mapping.keys())
    }
    
    if not configs_dir.exists():
        return available
    
    # Scan for individual config files
    for category in ["baselines", "federated", "privacy", "hybrid"]:
        category_dir = configs_dir / category
        if category_dir.exists():
            for yaml_file in category_dir.glob("*.yaml"):
                available[category].append(yaml_file.stem)
    
    return available