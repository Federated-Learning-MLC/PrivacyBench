import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Enhanced configuration validator for both legacy and individual configs.
    Returns (is_valid, list_of_errors).
    """
    
    errors = []
    
    # Core validation (applies to both legacy and individual configs)
    errors.extend(validate_core_fields(config))
    errors.extend(validate_dataset_config(config))
    errors.extend(validate_model_config(config))
    errors.extend(validate_training_config(config))
    
    # Privacy validation (if present)
    if 'privacy' in config:
        errors.extend(validate_privacy_config(config['privacy']))
    
    # Schema validation for individual configs
    if is_individual_config(config):
        errors.extend(validate_against_schema(config))
    
    return len(errors) == 0, errors


def is_individual_config(config: Dict[str, Any]) -> bool:
    """Detect if this is an individual config file vs legacy CLI config."""
    # Individual configs have more structured format
    return ('output' in config and 'tracking' in config and 
            isinstance(config.get('dataset', {}), dict) and 
            'name' in config.get('dataset', {}))


def validate_core_fields(config: Dict[str, Any]) -> List[str]:
    """Validate core required fields."""
    
    errors = []
    
    if 'experiment' not in config:
        errors.append("Missing 'experiment' field")
    
    if 'dataset' not in config:
        errors.append("Missing 'dataset' configuration")
    
    if 'model' not in config:
        errors.append("Missing 'model' configuration")
    
    return errors


def validate_dataset_config(config: Dict[str, Any]) -> List[str]:
    """Validate dataset configuration."""
    
    errors = []
    dataset_config = config.get('dataset', {})
    
    # Dataset name validation
    dataset_name = dataset_config.get('name')
    if not dataset_name:
        errors.append("Dataset name is required")
    elif dataset_name not in ['alzheimer', 'skin_lesions']:
        errors.append(f"Unsupported dataset: {dataset_name}")
    
    # Batch size validation
    batch_size = dataset_config.get('batch_size', 32)
    if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 512:
        errors.append(f"Invalid batch size: {batch_size} (must be 1-512)")
    
    # Height/width validation
    height_width = dataset_config.get('height_width', [224, 224])
    if not isinstance(height_width, list) or len(height_width) != 2:
        errors.append("height_width must be a list of [height, width]")
    
    return errors


def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """Validate model configuration."""
    
    errors = []
    model_config = config.get('model', {})
    
    # Architecture validation
    architecture = model_config.get('architecture')
    if not architecture:
        errors.append("Model architecture is required")
    elif architecture not in ['cnn', 'vit']:
        errors.append(f"Unsupported model architecture: {architecture}")
    
    # Number of classes validation
    num_classes = model_config.get('num_classes')
    if num_classes is not None:
        if not isinstance(num_classes, int) or num_classes < 2 or num_classes > 1000:
            errors.append(f"Invalid num_classes: {num_classes} (must be 2-1000)")
    
    return errors


def validate_training_config(config: Dict[str, Any]) -> List[str]:
    """Validate training configuration."""
    
    errors = []
    training_config = config.get('training', {})
    
    # Epochs validation
    epochs = training_config.get('epochs', 50)
    if not isinstance(epochs, int) or epochs < 1 or epochs > 1000:
        errors.append(f"Invalid epochs: {epochs} (must be 1-1000)")
    
    # Learning rate validation
    lr = training_config.get('learning_rate', 0.001)
    if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1.0:
        errors.append(f"Invalid learning rate: {lr} (must be > 0 and <= 1.0)")
    
    # Batch size validation (if specified in training)
    batch_size = training_config.get('batch_size')
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 512:
            errors.append(f"Invalid training batch size: {batch_size}")
    
    return errors


def validate_privacy_config(privacy_config: Dict[str, Any]) -> List[str]:
    """Validate privacy configuration."""
    
    errors = []
    techniques = privacy_config.get('techniques', [])
    
    if not isinstance(techniques, list):
        errors.append("Privacy techniques must be a list")
        return errors
    
    for i, technique in enumerate(techniques):
        if not isinstance(technique, dict):
            errors.append(f"Privacy technique {i} must be a dictionary")
            continue
        
        # Validate technique name
        name = technique.get('name')
        if not name:
            errors.append(f"Privacy technique {i} missing name")
            continue
        
        valid_techniques = [
            'federated_learning', 
            'differential_privacy', 
            'secure_multiparty_computation'
        ]
        if name not in valid_techniques:
            errors.append(f"Unknown privacy technique: {name}")
        
        # Technique-specific validation
        if name == 'federated_learning':
            errors.extend(validate_fl_config(technique.get('config', {}), i))
        elif name == 'differential_privacy':
            errors.extend(validate_dp_config(technique.get('config', {}), i))
        elif name == 'secure_multiparty_computation':
            errors.extend(validate_smpc_config(technique.get('config', {}), i))
    
    return errors


def validate_fl_config(fl_config: Dict[str, Any], technique_index: int) -> List[str]:
    """Validate federated learning configuration."""
    
    errors = []
    
    num_clients = fl_config.get('num_clients', 3)
    if not isinstance(num_clients, int) or num_clients < 2 or num_clients > 100:
        errors.append(f"FL technique {technique_index}: invalid num_clients (must be 2-100)")
    
    num_rounds = fl_config.get('num_rounds', 5)
    if not isinstance(num_rounds, int) or num_rounds < 1 or num_rounds > 100:
        errors.append(f"FL technique {technique_index}: invalid num_rounds (must be 1-100)")
    
    client_fraction = fl_config.get('client_fraction', 1.0)
    if not isinstance(client_fraction, (int, float)) or client_fraction <= 0 or client_fraction > 1.0:
        errors.append(f"FL technique {technique_index}: invalid client_fraction (must be 0-1)")
    
    return errors


def validate_dp_config(dp_config: Dict[str, Any], technique_index: int) -> List[str]:
    """Validate differential privacy configuration."""
    
    errors = []
    
    epsilon = dp_config.get('epsilon', 1.0)
    if not isinstance(epsilon, (int, float)) or epsilon <= 0 or epsilon > 50:
        errors.append(f"DP technique {technique_index}: invalid epsilon (must be > 0 and <= 50)")
    
    delta = dp_config.get('delta', 1e-5)
    if not isinstance(delta, (int, float)) or delta <= 0 or delta >= 1:
        errors.append(f"DP technique {technique_index}: invalid delta (must be > 0 and < 1)")
    
    max_grad_norm = dp_config.get('max_grad_norm', 1.0)  
    if not isinstance(max_grad_norm, (int, float)) or max_grad_norm <= 0 or max_grad_norm > 10:
        errors.append(f"DP technique {technique_index}: invalid max_grad_norm (must be > 0 and <= 10)")
    
    return errors


def validate_smpc_config(smpc_config: Dict[str, Any], technique_index: int) -> List[str]:
    """Validate SMPC configuration."""
    
    errors = []
    
    num_parties = smpc_config.get('num_parties', 3)
    if not isinstance(num_parties, int) or num_parties < 2 or num_parties > 10:
        errors.append(f"SMPC technique {technique_index}: invalid num_parties (must be 2-10)")
    
    threshold = smpc_config.get('threshold', 2)
    if not isinstance(threshold, int) or threshold < 1 or threshold > num_parties:
        errors.append(f"SMPC technique {technique_index}: invalid threshold (must be 1-{num_parties})")
    
    return errors


def validate_against_schema(config: Dict[str, Any]) -> List[str]:
    """Validate configuration against schema (Phase 3)."""
    
    errors = []
    
    # Load schema
    schema_path = Path("configs/schema.yaml")
    if not schema_path.exists():
        # Schema validation not available, skip
        return errors
    
    try:
        with open(schema_path, 'r') as f:
            schema = yaml.safe_load(f)
        
        # Basic schema validation (simplified)
        errors.extend(validate_field_types(config, schema))
        
    except Exception as e:
        errors.append(f"Schema validation failed: {e}")
    
    return errors


def validate_field_types(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Basic field type validation against schema."""
    
    errors = []
    
    # This is a simplified schema validator
    # In production, you'd use a library like jsonschema or cerberus
    
    return errors  # Placeholder for now
