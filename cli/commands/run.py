"""
Run command to execute privacy-preserving ML experiments
Phase 1: Configuration parsing and validation only
Phase 3: Will add actual execution
"""
import argparse
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from cli.parser import ConfigParser
from cli.validator import ConfigValidator
from cli.resolver import ConfigResolver


class RunCommand:
    """Handles the 'run' command"""
    
    def __init__(self):
        self.parser = ConfigParser()
        self.validator = ConfigValidator()
        self.resolver = ConfigResolver()
    
    def execute(self, args: argparse.Namespace) -> int:
        """Execute run command"""
        try:
            # Phase 1: Configuration processing
            print("⚙️ Parsing configuration...")
            config = self.parser.parse_cli_to_config(args)
            
            print("✅ Validating configuration...")
            validation_errors = self.validator.validate(config)
            
            # Check for fatal errors
            fatal_errors = [e for e in validation_errors if e.severity == "error"]
            if fatal_errors:
                print("❌ Configuration validation failed:")
                for error in fatal_errors:
                    print(f"   • {error.field}: {error.message}")
                return 1
            
            # Resolve configuration
            print("🔧 Resolving configuration dependencies...")
            resolved_config = self.resolver.resolve_config(config)
            
            # Print warnings if any
            warnings = [e for e in validation_errors if e.severity == "warning"]
            if warnings:
                print("⚠️ Configuration warnings:")
                for warning in warnings:
                    print(f"   • {warning.field}: {warning.message}")
            
            # Phase 1: Show what would be executed (no actual execution yet)
            if args.dry_run:
                return self._show_dry_run_results(resolved_config)
            else:
                # Phase 1: Just show configuration, actual execution in Phase 3
                print("🚀 Configuration ready for execution!")
                self._show_experiment_summary(resolved_config)
                
                print("\n" + "="*60)
                print("📝 Phase 1: Configuration parsing complete")
                print("🔧 Phase 3 will add actual experiment execution")
                print("   For now, use --dry-run to see experiment details")
                print("="*60)
                
                return 0
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    def _show_dry_run_results(self, config: Dict[str, Any]) -> int:
        """Show what would be executed without running"""
        print("\n" + "="*60)
        print("🔍 DRY RUN - Experiment Configuration")
        print("="*60)
        
        # Show experiment summary
        self._show_experiment_summary(config)
        
        # Show detailed configuration
        self._show_detailed_config(config)
        
        # Show execution plan
        self._show_execution_plan(config)
        
        print("\n✅ Configuration is valid and ready for execution!")
        print("   Remove --dry-run to execute the experiment (in Phase 3)")
        
        return 0
    
    def _show_experiment_summary(self, config: Dict[str, Any]):
        """Show a summary of the experiment configuration"""
        metadata = config.get("metadata", {})
        dataset_config = config.get("dataset", {})
        model_config = config.get("model", {})
        privacy_config = config.get("privacy", {})
        training_config = config.get("training", {})
        
        print(f"\n📋 Experiment Summary:")
        print(f"   Name: {metadata.get('name', 'Unknown')}")
        print(f"   Type: {metadata.get('experiment_type', 'Unknown')}")
        print(f"   Dataset: {dataset_config.get('name', 'Unknown')}")
        print(f"   Model: {model_config.get('architecture', 'Unknown')}")
        
        # Privacy techniques
        techniques = privacy_config.get("techniques", [])
        if techniques:
            technique_names = [t.get("name", "Unknown") for t in techniques]
            print(f"   Privacy: {', '.join(technique_names)}")
        else:
            print(f"   Privacy: None (Baseline)")
        
        # Training parameters
        print(f"   Epochs: {training_config.get('epochs', 'Unknown')}")
        print(f"   Batch Size: {training_config.get('batch_size', 'Unknown')}")
        print(f"   Learning Rate: {training_config.get('learning_rate', 'Unknown')}")
    
    def _show_detailed_config(self, config: Dict[str, Any]):
        """Show detailed configuration breakdown"""
        print(f"\n🔧 Detailed Configuration:")
        
        # Dataset details
        dataset_config = config.get("dataset", {})
        print(f"\n   📊 Dataset Configuration:")
        print(f"      Name: {dataset_config.get('name')}")
        dataset_details = dataset_config.get("config", {})
        for key, value in dataset_details.items():
            print(f"      {key}: {value}")
        
        # Model details
        model_config = config.get("model", {})
        print(f"\n   🧠 Model Configuration:")
        print(f"      Architecture: {model_config.get('architecture')}")
        model_details = model_config.get("config", {})
        for key, value in model_details.items():
            if key != "name":  # Skip redundant name field
                print(f"      {key}: {value}")
        
        # Privacy details
        privacy_config = config.get("privacy", {})
        techniques = privacy_config.get("techniques", [])
        if techniques:
            print(f"\n   🔒 Privacy Configuration:")
            for i, technique in enumerate(techniques, 1):
                print(f"      {i}. {technique.get('name', 'Unknown')}")
                technique_config = technique.get("config", {})
                for key, value in technique_config.items():
                    print(f"         {key}: {value}")
        
        # Training details
        training_config = config.get("training", {})
        print(f"\n   🏋️ Training Configuration:")
        for key, value in training_config.items():
            print(f"      {key}: {value}")
        
        # Output details
        output_config = config.get("output", {})
        print(f"\n   📁 Output Configuration:")
        for key, value in output_config.items():
            print(f"      {key}: {value}")
    
    def _show_execution_plan(self, config: Dict[str, Any]):
        """Show the execution plan"""
        print(f"\n📋 Execution Plan:")
        
        # Phase 1: Setup
        print(f"   1. 🔧 Setup Phase:")
        print(f"      • Set random seed: {config.get('training', {}).get('seed', 42)}")
        print(f"      • Initialize tracking (W&B, CodeCarbon)")
        print(f"      • Create output directory: {config.get('output', {}).get('directory')}")
        
        # Phase 2: Data preparation
        dataset_name = config.get("dataset", {}).get("name")
        print(f"   2. 📊 Data Preparation:")
        print(f"      • Load {dataset_name} dataset")
        print(f"      • Apply data transformations and augmentation")
        print(f"      • Create train/validation/test splits")
        
        # Phase 3: Privacy setup (if applicable)
        techniques = config.get("privacy", {}).get("techniques", [])
        if techniques:
            print(f"   3. 🔒 Privacy Setup:")
            for technique in techniques:
                name = technique.get("name", "Unknown")
                if name == "federated_learning":
                    fl_config = technique.get("config", {})
                    print(f"      • Setup federated learning with {fl_config.get('num_clients', 3)} clients")
                elif name == "differential_privacy":
                    dp_config = technique.get("config", {})
                    print(f"      • Setup differential privacy (ε={dp_config.get('epsilon', 1.0)})")
                elif name == "secure_multiparty_computation":
                    print(f"      • Setup secure multi-party computation")
        
        # Phase 4: Model training
        model_arch = config.get("model", {}).get("architecture")
        epochs = config.get("training", {}).get("epochs")
        print(f"   4. 🏋️ Model Training:")
        print(f"      • Initialize {model_arch.upper()} model")
        print(f"      • Train for {epochs} epochs")
        print(f"      • Monitor performance and resource usage")
        
        # Phase 5: Evaluation and results
        print(f"   5. 📈 Evaluation & Results:")
        print(f"      • Evaluate model performance")
        print(f"      • Collect resource consumption metrics")
        print(f"      • Generate results in multiple formats")
        
        # Expected results (based on your existing experiments)
        self._show_expected_results(config)
    
    def _show_expected_results(self, config: Dict[str, Any]):
        """Show expected results based on existing experiment data"""
        exp_type = config.get("metadata", {}).get("experiment_type", "")
        dataset = config.get("dataset", {}).get("name", "")
        
        print(f"\n📊 Expected Results (based on previous experiments):")
        
        # Map to known results from your existing experiments
        expected_results = {
            ("cnn_baseline", "alzheimer"): {"accuracy": "97.9%", "energy": "0.026 kWh", "time": "~588s"},
            ("vit_baseline", "alzheimer"): {"accuracy": "99.0%", "energy": "0.119 kWh", "time": "~3246s"},
            ("fl_cnn", "alzheimer"): {"accuracy": "~98.0%", "energy": "0.036 kWh", "time": "~882s"},
            ("cnn_baseline", "skin_lesions"): {"accuracy": "~95.2%", "energy": "0.031 kWh", "time": "~642s"},
        }
        
        key = (exp_type, dataset)
        if key in expected_results:
            results = expected_results[key]
            print(f"      • Accuracy: {results['accuracy']}")
            print(f"      • Energy Consumption: {results['energy']}")
            print(f"      • Training Time: {results['time']}")
        else:
            print(f"      • No historical data available for this combination")
            print(f"      • Results will vary based on model and privacy techniques")
        
        print(f"\n⚠️  Note: Actual results may vary based on:")
        print(f"      • Hardware specifications")
        print(f"      • Random seed and initialization")
        print(f"      • Privacy technique parameters")
    
    def _validate_cli_arguments(self, args: argparse.Namespace) -> bool:
        """Validate CLI arguments before processing"""
        from cli.commands.validate import ValidateCommand
        
        validator = ValidateCommand()
        is_valid, errors = validator.quick_validate_cli_args(args.experiment, args.dataset)
        
        if not is_valid:
            print("❌ Invalid CLI arguments:")
            for error in errors:
                print(f"   • {error}")
            print("\n💡 Use 'privacybench list all' to see available options")
            return False
        
        return True