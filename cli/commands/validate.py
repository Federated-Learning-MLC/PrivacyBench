"""
Validate command to check experiment configurations
"""
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from cli.validator import ConfigValidator
from cli.resolver import ConfigResolver


class ValidateCommand:
    """Handles the 'validate' command"""
    
    def __init__(self):
        self.validator = ConfigValidator()
        self.resolver = ConfigResolver()
    
    def execute(self, args: argparse.Namespace) -> int:
        """Execute validate command"""
        try:
            # Load configuration file
            config = self._load_config_file(args.config)
            
            if args.verbose:
                print(f"📁 Loading configuration from: {args.config}")
                print("📋 Configuration loaded successfully")
            
            # Resolve configuration (apply defaults and dependencies)
            resolved_config = self.resolver.resolve_config(config)
            
            if args.verbose:
                print("🔧 Configuration resolution completed")
                print("\n📊 Configuration Summary:")
                print("-" * 40)
                print(self.resolver.get_config_summary(resolved_config))
                print("-" * 40)
            
            # Validate configuration
            validation_errors = self.validator.validate(resolved_config)
            
            # Check for resolver warnings
            resolver_issues = self.resolver.validate_resolved_config(resolved_config)
            
            # Print results
            self._print_validation_results(validation_errors, resolver_issues, args.verbose)
            
            # Return appropriate exit code
            error_count = len([e for e in validation_errors if e.severity == "error"])
            return 1 if error_count > 0 else 0
            
        except FileNotFoundError:
            print(f"❌ Error: Configuration file not found: {args.config}")
            return 1
        except yaml.YAMLError as e:
            print(f"❌ Error: Invalid YAML syntax in {args.config}")
            print(f"   {e}")
            return 1
        except Exception as e:
            print(f"❌ Error: Failed to validate configuration: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load and parse YAML configuration file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise ValueError("Configuration file must contain a YAML dictionary")
        
        return config
    
    def _print_validation_results(self, validation_errors, resolver_issues, verbose: bool):
        """Print validation results in a user-friendly format"""
        
        # Count errors and warnings
        error_count = len([e for e in validation_errors if e.severity == "error"])
        warning_count = len([e for e in validation_errors if e.severity == "warning"])
        issue_count = len(resolver_issues)
        
        # Print header
        print(f"\n🔍 Validation Results")
        print("=" * 60)
        
        if error_count == 0 and warning_count == 0 and issue_count == 0:
            print("✅ Configuration is valid and ready for execution!")
            if verbose:
                print("   • All required fields are present")
                print("   • All values are within valid ranges")
                print("   • No conflicting parameters detected")
                print("   • Dependencies resolved successfully")
            return
        
        # Print summary
        print(f"📊 Summary: {error_count} errors, {warning_count} warnings, {issue_count} issues")
        
        # Print validation errors
        if validation_errors:
            print(f"\n📋 Validation Results:")
            print("-" * 40)
            
            for error in validation_errors:
                if error.severity == "error":
                    print(f"❌ ERROR in {error.field}:")
                    print(f"   {error.message}")
                elif error.severity == "warning":
                    print(f"⚠️  WARNING in {error.field}:")
                    print(f"   {error.message}")
                elif error.severity == "info":
                    print(f"ℹ️  INFO in {error.field}:")
                    print(f"   {error.message}")
        
        # Print resolver issues
        if resolver_issues:
            print(f"\n⚙️  Configuration Issues:")
            print("-" * 40)
            for i, issue in enumerate(resolver_issues, 1):
                print(f"⚠️  {i}. {issue}")
        
        # Print final status
        print("\n" + "=" * 60)
        if error_count > 0:
            print(f"❌ Configuration has {error_count} errors that must be fixed before running.")
            print("   Please correct the errors above and validate again.")
        elif warning_count > 0 or issue_count > 0:
            print(f"⚠️  Configuration has {warning_count + issue_count} warnings/issues but can still run.")
            print("   Consider reviewing the warnings above for optimal performance.")
        else:
            print("✅ Configuration passed validation and is ready for execution!")
        
        # Provide helpful tips
        if verbose:
            self._print_validation_tips(validation_errors)
    
    def _print_validation_tips(self, validation_errors):
        """Print helpful tips for fixing validation errors"""
        error_types = set(e.field.split('.')[0] for e in validation_errors if e.severity == "error")
        
        if not error_types:
            return
        
        print(f"\n💡 Tips for Fixing Errors:")
        print("-" * 30)
        
        if "metadata" in error_types:
            print("• Metadata errors: Ensure 'name', 'experiment_type', and 'dataset' are specified")
        
        if "dataset" in error_types:
            print("• Dataset errors: Use 'alzheimer' or 'skin_lesions' as dataset name")
            print("  Ensure test_split and validation_split are between 0 and 1")
        
        if "model" in error_types:
            print("• Model errors: Use 'cnn' or 'vit' as architecture")
            print("  Ensure num_classes matches dataset (alzheimer=4, skin_lesions=8)")
        
        if "privacy" in error_types:
            print("• Privacy errors: Use valid technique names:")
            print("  'federated_learning', 'differential_privacy', 'secure_multiparty_computation'")
        
        if "training" in error_types:
            print("• Training errors: Ensure epochs and batch_size are positive integers")
            print("  Ensure learning_rate is a positive number")
            print("  Use valid optimizer: 'adam', 'sgd', or 'adamw'")
        
        print("\n📖 For more help, see: https://github.com/privacybench/privacybench/docs")
    
    def validate_experiment_combination(self, experiment: str, dataset: str) -> bool:
        """Validate specific experiment-dataset combination"""
        valid_combinations = {
            "alzheimer": [
                "cnn_baseline", "vit_baseline", 
                "fl_cnn", "fl_vit",
                "dp_cnn", "dp_vit", 
                "fl_dp_cnn", "smpc_cnn"
            ],
            "skin_lesions": [
                "cnn_baseline", "vit_baseline",
                "fl_cnn", "fl_vit", 
                "dp_cnn", "dp_vit",
                "fl_dp_cnn", "smpc_cnn"
            ]
        }
        
        return experiment in valid_combinations.get(dataset, [])
    
    def quick_validate_cli_args(self, experiment: str, dataset: str) -> tuple[bool, list[str]]:
        """Quick validation for CLI arguments before full config processing"""
        errors = []
        
        # Validate dataset
        valid_datasets = ["alzheimer", "skin_lesions"]
        if dataset not in valid_datasets:
            errors.append(f"Invalid dataset '{dataset}'. Must be one of: {valid_datasets}")
        
        # Validate experiment
        valid_experiments = [
            "cnn_baseline", "vit_baseline",
            "fl_cnn", "fl_vit", 
            "dp_cnn", "dp_vit",
            "fl_dp_cnn", "smpc_cnn"
        ]
        if experiment not in valid_experiments:
            errors.append(f"Invalid experiment '{experiment}'. Must be one of: {valid_experiments}")
        
        # Validate combination
        if not errors and not self.validate_experiment_combination(experiment, dataset):
            errors.append(f"Experiment '{experiment}' is not supported with dataset '{dataset}'")
        
        return len(errors) == 0, errors