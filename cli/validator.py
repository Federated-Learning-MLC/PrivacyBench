"""
Configuration validator
Ensures configs are valid before experiment execution
"""
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationError:
    """Represents a configuration validation error"""
    field: str
    message: str
    severity: str = "error"  # "error", "warning", "info"


class ConfigValidator:
    """Validates experiment configurations"""
    
    def __init__(self):
        # Define valid choices based on your existing setup
        self.valid_datasets = ["alzheimer", "skin_lesions"]
        self.valid_architectures = ["cnn", "vit"]
        self.valid_privacy_techniques = [
            "federated_learning", 
            "differential_privacy", 
            "secure_multiparty_computation"
        ]
        self.valid_optimizers = ["adam", "sgd", "adamw"]
    
    def validate(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate required sections
        errors.extend(self._validate_required_sections(config))
        
        # Validate metadata
        if "metadata" in config:
            errors.extend(self._validate_metadata(config["metadata"]))
        
        # Validate dataset configuration
        if "dataset" in config:
            errors.extend(self._validate_dataset(config["dataset"]))
        
        # Validate model configuration
        if "model" in config:
            errors.extend(self._validate_model(config["model"]))
        
        # Validate privacy configuration
        if "privacy" in config:
            errors.extend(self._validate_privacy(config["privacy"]))
        
        # Validate training configuration
        if "training" in config:
            errors.extend(self._validate_training(config["training"]))
        
        # Validate resource configuration
        if "resources" in config:
            errors.extend(self._validate_resources(config["resources"]))
        
        # Validate output configuration
        if "output" in config:
            errors.extend(self._validate_output(config["output"]))
        
        # Cross-validation checks
        errors.extend(self._validate_combinations(config))
        
        return errors
    
    def _validate_required_sections(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate that required configuration sections are present"""
        errors = []
        required_sections = ["metadata", "dataset", "model", "training"]
        
        for section in required_sections:
            if section not in config:
                errors.append(ValidationError(
                    field=section,
                    message=f"Required section '{section}' is missing"
                ))
        
        return errors
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> List[ValidationError]:
        """Validate metadata section"""
        errors = []
        required_fields = ["name", "experiment_type", "dataset"]
        
        for field in required_fields:
            if field not in metadata:
                errors.append(ValidationError(
                    field=f"metadata.{field}",
                    message=f"Required metadata field '{field}' is missing"
                ))
        
        return errors
    
    def _validate_dataset(self, dataset: Dict[str, Any]) -> List[ValidationError]:
        """Validate dataset configuration"""
        errors = []
        
        # Check dataset name
        dataset_name = dataset.get("name")
        if not dataset_name:
            errors.append(ValidationError(
                field="dataset.name",
                message="Dataset name is required"
            ))
        elif dataset_name not in self.valid_datasets:
            errors.append(ValidationError(
                field="dataset.name",
                message=f"Unsupported dataset: {dataset_name}. Must be one of {self.valid_datasets}"
            ))
        
        # Validate dataset config if present
        if "config" in dataset:
            config = dataset["config"]
            
            # Validate test_split
            test_split = config.get("test_split")
            if test_split and (not isinstance(test_split, (int, float)) or test_split <= 0 or test_split >= 1):
                errors.append(ValidationError(
                    field="dataset.config.test_split",
                    message="test_split must be a number between 0 and 1"
                ))
            
            # Validate validation_split
            val_split = config.get("validation_split")
            if val_split and (not isinstance(val_split, (int, float)) or val_split <= 0 or val_split >= 1):
                errors.append(ValidationError(
                    field="dataset.config.validation_split",
                    message="validation_split must be a number between 0 and 1"
                ))
        
        return errors
    
    def _validate_model(self, model: Dict[str, Any]) -> List[ValidationError]:
        """Validate model configuration"""
        errors = []
        
        # Check architecture
        architecture = model.get("architecture")
        if not architecture:
            errors.append(ValidationError(
                field="model.architecture",
                message="Model architecture is required"
            ))
        elif architecture not in self.valid_architectures:
            errors.append(ValidationError(
                field="model.architecture",
                message=f"Unsupported architecture: {architecture}. Must be one of {self.valid_architectures}"
            ))
        
        # Validate model config if present
        if "config" in model:
            config = model["config"]
            
            # Validate num_classes
            num_classes = config.get("num_classes")
            if num_classes and (not isinstance(num_classes, int) or num_classes <= 0):
                errors.append(ValidationError(
                    field="model.config.num_classes",
                    message="num_classes must be a positive integer"
                ))
            
            # Validate dropout
            dropout = config.get("dropout")
            if dropout and (not isinstance(dropout, (int, float)) or dropout < 0 or dropout > 1):
                errors.append(ValidationError(
                    field="model.config.dropout",
                    message="dropout must be a number between 0 and 1"
                ))
        
        return errors
    
    def _validate_privacy(self, privacy: Dict[str, Any]) -> List[ValidationError]:
        """Validate privacy configuration"""
        errors = []
        
        techniques = privacy.get("techniques", [])
        if not isinstance(techniques, list):
            errors.append(ValidationError(
                field="privacy.techniques",
                message="privacy.techniques must be a list"
            ))
            return errors
        
        for i, technique in enumerate(techniques):
            if not isinstance(technique, dict):
                errors.append(ValidationError(
                    field=f"privacy.techniques[{i}]",
                    message="Each privacy technique must be a dictionary"
                ))
                continue
            
            # Validate technique name
            name = technique.get("name")
            if not name:
                errors.append(ValidationError(
                    field=f"privacy.techniques[{i}].name",
                    message="Privacy technique name is required"
                ))
            elif name not in self.valid_privacy_techniques:
                errors.append(ValidationError(
                    field=f"privacy.techniques[{i}].name",
                    message=f"Unsupported privacy technique: {name}. Must be one of {self.valid_privacy_techniques}"
                ))
            
            # Validate specific technique configurations
            if name == "differential_privacy":
                errors.extend(self._validate_dp_config(technique.get("config", {}), f"privacy.techniques[{i}].config"))
            elif name == "federated_learning":
                errors.extend(self._validate_fl_config(technique.get("config", {}), f"privacy.techniques[{i}].config"))
        
        return errors
    
    def _validate_dp_config(self, config: Dict[str, Any], field_prefix: str) -> List[ValidationError]:
        """Validate differential privacy configuration"""
        errors = []
        
        # Validate epsilon
        epsilon = config.get("epsilon")
        if epsilon and (not isinstance(epsilon, (int, float)) or epsilon <= 0):
            errors.append(ValidationError(
                field=f"{field_prefix}.epsilon",
                message="epsilon must be a positive number"
            ))
        
        # Validate delta
        delta = config.get("delta")
        if delta and (not isinstance(delta, (int, float)) or delta <= 0 or delta >= 1):
            errors.append(ValidationError(
                field=f"{field_prefix}.delta",
                message="delta must be a number between 0 and 1"
            ))
        
        return errors
    
    def _validate_fl_config(self, config: Dict[str, Any], field_prefix: str) -> List[ValidationError]:
        """Validate federated learning configuration"""
        errors = []
        
        # Validate num_clients
        num_clients = config.get("num_clients")
        if num_clients and (not isinstance(num_clients, int) or num_clients <= 0):
            errors.append(ValidationError(
                field=f"{field_prefix}.num_clients",
                message="num_clients must be a positive integer"
            ))
        
        # Validate num_rounds
        num_rounds = config.get("num_rounds")
        if num_rounds and (not isinstance(num_rounds, int) or num_rounds <= 0):
            errors.append(ValidationError(
                field=f"{field_prefix}.num_rounds",
                message="num_rounds must be a positive integer"
            ))
        
        return errors
    
    def _validate_training(self, training: Dict[str, Any]) -> List[ValidationError]:
        """Validate training configuration"""
        errors = []
        
        # Validate epochs
        epochs = training.get("epochs")
        if epochs and (not isinstance(epochs, int) or epochs <= 0):
            errors.append(ValidationError(
                field="training.epochs",
                message="epochs must be a positive integer"
            ))
        
        # Validate batch_size
        batch_size = training.get("batch_size")
        if batch_size and (not isinstance(batch_size, int) or batch_size <= 0):
            errors.append(ValidationError(
                field="training.batch_size",
                message="batch_size must be a positive integer"
            ))
        
        # Validate learning_rate
        learning_rate = training.get("learning_rate")
        if learning_rate and (not isinstance(learning_rate, (int, float)) or learning_rate <= 0):
            errors.append(ValidationError(
                field="training.learning_rate",
                message="learning_rate must be a positive number"
            ))
        
        # Validate optimizer
        optimizer = training.get("optimizer")
        if optimizer and optimizer not in self.valid_optimizers:
            errors.append(ValidationError(
                field="training.optimizer",
                message=f"Unsupported optimizer: {optimizer}. Must be one of {self.valid_optimizers}"
            ))
        
        return errors
    
    def _validate_resources(self, resources: Dict[str, Any]) -> List[ValidationError]:
        """Validate resource configuration"""
        errors = []
        
        # Validate num_workers
        num_workers = resources.get("num_workers")
        if num_workers and (not isinstance(num_workers, int) or num_workers < 0):
            errors.append(ValidationError(
                field="resources.num_workers",
                message="num_workers must be a non-negative integer"
            ))
        
        return errors
    
    def _validate_output(self, output: Dict[str, Any]) -> List[ValidationError]:
        """Validate output configuration"""
        errors = []
        
        # Validate directory path
        directory = output.get("directory")
        if directory:
            try:
                Path(directory)
            except Exception:
                errors.append(ValidationError(
                    field="output.directory",
                    message=f"Invalid directory path: {directory}"
                ))
        
        # Validate export formats
        export_formats = output.get("export_formats")
        if export_formats and not isinstance(export_formats, list):
            errors.append(ValidationError(
                field="output.export_formats",
                message="export_formats must be a list"
            ))
        
        return errors
    
    def _validate_combinations(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate logical combinations and constraints"""
        errors = []
        
        # Check dataset-model compatibility
        dataset_name = config.get("dataset", {}).get("name")
        model_arch = config.get("model", {}).get("architecture")
        
        if dataset_name == "alzheimer" and model_arch:
            num_classes = config.get("model", {}).get("config", {}).get("num_classes")
            if num_classes and num_classes != 4:
                errors.append(ValidationError(
                    field="model.config.num_classes",
                    message="Alzheimer dataset requires num_classes=4",
                    severity="warning"
                ))
        
        if dataset_name == "skin_lesions" and model_arch:
            num_classes = config.get("model", {}).get("config", {}).get("num_classes")
            if num_classes and num_classes != 8:
                errors.append(ValidationError(
                    field="model.config.num_classes",
                    message="Skin lesions dataset requires num_classes=8",
                    severity="warning"
                ))
        
        return errors
    
    def is_valid(self, config: Dict[str, Any]) -> bool:
        """Check if configuration is valid (no fatal errors)"""
        errors = self.validate(config)
        fatal_errors = [e for e in errors if e.severity == "error"]
        return len(fatal_errors) == 0
    
    def print_validation_results(self, errors: List[ValidationError]) -> None:
        """Print validation results in a user-friendly format"""
        if not errors:
            print("✅ Configuration is valid!")
            return
        
        # Group errors by severity
        error_count = len([e for e in errors if e.severity == "error"])
        warning_count = len([e for e in errors if e.severity == "warning"])
        
        print(f"\n📋 Validation Results: {error_count} errors, {warning_count} warnings")
        print("=" * 60)
        
        for error in errors:
            if error.severity == "error":
                print(f"❌ Error in {error.field}: {error.message}")
            elif error.severity == "warning":
                print(f"⚠️  Warning in {error.field}: {error.message}")
            else:
                print(f"ℹ️  Info in {error.field}: {error.message}")
        
        if error_count > 0:
            print(f"\n❌ Configuration has {error_count} errors that must be fixed before running.")
        elif warning_count > 0:
            print(f"\n⚠️  Configuration has {warning_count} warnings but can still run.")
        else:
            print("\n✅ Configuration passed validation!")