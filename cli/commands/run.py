import sys
import time
from pathlib import Path
from typing import Any, Dict

from cli.parser import parse_experiment_config
from cli.validator import validate_config
from cli.resolver import resolve_dependencies
from execution.manager import ExecutionEngine


class RunCommand:
    """
    Enhanced Run command with actual experiment execution.
    Builds on Phase 1 dry-run functionality to enable real training.
    """
    
    def __init__(self):
        self.execution_engine = None
        
    def execute(self, args) -> int:
        """Execute run command with actual training."""
        try:
            print("🚀 PrivacyBench CLI - Experiment Execution")
            print("=" * 50)
            
            # Parse configuration (Phase 1 functionality preserved)
            config = parse_experiment_config(args)
            
            # Validate configuration
            is_valid, errors = validate_config(config)
            if not is_valid:
                print("❌ Configuration validation failed:")
                for error in errors:
                    print(f"   • {error}")
                return 1
            
            # Resolve dependencies
            resolved_config = resolve_dependencies(config)
            
            # Check if dry-run mode
            if args.dry_run:
                return self._execute_dry_run(resolved_config)
            else:
                return self._execute_actual_experiment(resolved_config, args)
                
        except KeyboardInterrupt:
            print("\n⚠️ Experiment interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _execute_dry_run(self, config: Dict[str, Any]) -> int:
        """Execute dry-run validation (Phase 1 functionality preserved)."""
        print("\n🧪 DRY RUN MODE - Configuration Validation")
        print("-" * 40)
        
        # Show configuration summary (Phase 1 code preserved)
        self._show_config_summary(config)
        self._show_execution_plan(config)
        
        print("\n✅ Configuration valid, experiment ready to run")
        print("💡 Remove --dry-run flag to execute actual experiment")
        
        return 0
    
    def _execute_actual_experiment(self, config: Dict[str, Any], args) -> int:
        """Execute actual experiment using Phase 2 execution engine."""
        print("\n🏃 EXECUTING EXPERIMENT")
        print("-" * 40)
        
        try:
            # Initialize execution engine
            self.execution_engine = ExecutionEngine(config)
            
            # Setup experiment
            print("📋 Setting up experiment...")
            setup_result = self.execution_engine.setup_experiment()
            
            if setup_result.get('pipeline_setup'):
                print("✅ Pipeline setup completed")
                components = setup_result.get('components', [])
                print(f"   Components loaded: {', '.join(components)}")
            
            # Execute experiment
            print("\n🚀 Starting experiment execution...")
            print("   This may take several minutes...")
            
            results = self.execution_engine.execute_experiment()
            
            # Display results
            self._display_results(results, config)
            
            # Save results if output directory specified
            if 'output' in config and config['output'].get('directory'):
                self._save_results(results, config)
            
            return 0
            
        except Exception as e:
            print(f"❌ Experiment execution failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _show_config_summary(self, config: Dict[str, Any]):
        """Show configuration summary (Phase 1 code preserved)."""
        print(f"\n📋 Experiment Configuration:")
        print(f"   🧪 Experiment: {config.get('experiment', 'unknown')}")
        
        # Dataset info
        dataset_config = config.get('dataset', {})
        print(f"   📊 Dataset: {dataset_config.get('name', 'unknown')}")
        
        # Model info  
        model_config = config.get('model', {})
        print(f"   🧠 Model: {model_config.get('architecture', 'unknown')}")
        
        # Privacy techniques
        privacy_config = config.get('privacy', {})
        techniques = privacy_config.get('techniques', [])
        if techniques:
            print(f"   🔒 Privacy Techniques:")
            for technique in techniques:
                print(f"      • {technique.get('name', 'Unknown')}")
                technique_config = technique.get("config", {})
                for key, value in technique_config.items():
                    print(f"         {key}: {value}")
        else:
            print(f"   🔒 Privacy: None (Baseline)")
        
        # Training details
        training_config = config.get("training", {})
        print(f"\n   🏋️ Training Configuration:")
        for key, value in training_config.items():
            print(f"      {key}: {value}")
    
    def _show_execution_plan(self, config: Dict[str, Any]):
        """Show execution plan (Phase 1 code preserved)."""
        print(f"\n📋 Execution Plan:")
        
        # Phase 1: Setup
        print(f"   1. 🔧 Setup Phase:")
        print(f"      • Set random seed: {config.get('training', {}).get('seed', 42)}")
        print(f"      • Initialize tracking (W&B, CodeCarbon)")
        output_dir = config.get('output', {}).get('directory', './results')
        print(f"      • Create output directory: {output_dir}")
        
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
                    num_clients = fl_config.get('num_clients', 3)
                    print(f"      • Setup federated learning with {num_clients} clients")
                elif name == "differential_privacy":
                    dp_config = technique.get("config", {})
                    epsilon = dp_config.get('epsilon', 1.0)
                    print(f"      • Setup differential privacy (ε={epsilon})")
                elif name == "secure_multiparty_computation":
                    print(f"      • Setup secure multi-party computation")
        
        # Phase 4: Model training
        model_arch = config.get("model", {}).get("architecture")
        epochs = config.get("training", {}).get("epochs", 50)
        print(f"   4. 🏋️ Model Training:")
        print(f"      • Initialize {model_arch.upper()} model")
        print(f"      • Train for {epochs} epochs")
        print(f"      • Monitor performance and resource usage")
        
        # Phase 5: Evaluation and results
        print(f"   5. 📈 Evaluation & Results:")
        print(f"      • Evaluate model on test set")
        print(f"      • Calculate performance metrics")
        print(f"      • Generate experiment report")
    
    def _display_results(self, results: Dict[str, Any], config: Dict[str, Any]):
        """Display experiment results in formatted output."""
        print("\n" + "🎉" * 2 + " EXPERIMENT COMPLETED SUCCESSFULLY " + "🎉" * 2)
        print("=" * 60)
        
        # Experiment identification
        experiment_name = config.get('experiment', 'unknown')
        dataset_name = config.get('dataset', {}).get('name', 'unknown')
        model_arch = config.get('model', {}).get('architecture', 'unknown')
        
        print(f"📋 Experiment: {experiment_name}_{dataset_name}")
        print(f"📊 Dataset: {dataset_name}")
        print(f"🧠 Model: {model_arch}")
        
        # Privacy information
        privacy_tech = results.get('privacy_technique', 'None (Baseline)')
        print(f"🔒 Privacy: {privacy_tech}")
        
        # Duration
        duration = results.get('duration', 0)
        print(f"⏱️ Duration: {duration:.1f} seconds")
        
        # Performance metrics
        print(f"\n📈 PERFORMANCE METRICS:")
        if 'accuracy' in results:
            print(f" • Accuracy: {results['accuracy']:.2%}")
        if 'f1_score' in results:
            print(f" • F1 Score: {results['f1_score']:.4f}")
        if 'roc_auc' in results:
            print(f" • ROC AUC: {results['roc_auc']:.4f}")
        
        # Resource consumption
        print(f"\n⚡ RESOURCE CONSUMPTION:")
        print(f" • Training Time: {duration:.1f} seconds")
        
        if 'peak_gpu_memory' in results:
            print(f" • Peak GPU Memory: {results['peak_gpu_memory']:.2f} GB")
        
        if 'energy_consumed' in results:
            energy = results['energy_consumed']
            print(f" • Energy Consumed: {energy:.6f} kWh")
        
        if 'co2_emissions' in results:
            co2 = results['co2_emissions']
            print(f" • CO2 Emissions: {co2:.6f} kg")
        
        # Privacy-specific metrics
        if privacy_tech != 'None (Baseline)':
            print(f"\n🔒 PRIVACY METRICS:")
            if 'privacy_epsilon' in results:
                print(f" • Privacy Budget (ε): {results['privacy_epsilon']}")
            if 'privacy_delta' in results:
                print(f" • Privacy Budget (δ): {results['privacy_delta']}")
            if 'num_clients' in results:
                print(f" • FL Clients: {results['num_clients']}")
            if 'num_rounds' in results:
                print(f" • FL Rounds: {results['num_rounds']}")
        
        # Simulation warning
        if results.get('simulation_mode'):
            print(f"\n⚠️ NOTE: Results from simulation mode")
        
        # Output location
        output_dir = config.get('output', {}).get('directory')
        if output_dir:
            print(f"\n📁 Results saved to: {output_dir}")
        
        print("=" * 60)
    
    def _save_results(self, results: Dict[str, Any], config: Dict[str, Any]):
        """Save experiment results to files."""
        try:
            output_dir = Path(config['output']['directory'])
            
            # Create timestamped experiment directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_name = config.get('experiment', 'unknown')
            exp_dir = output_dir / f"exp_{experiment_name}_{timestamp}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results.json - Complete results
            import json
            results_file = exp_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'config': config,
                    'results': results,
                    'timestamp': timestamp
                }, f, indent=2, default=str)
            
            # Save metrics.csv - Key metrics table
            self._save_metrics_csv(exp_dir / "metrics.csv", results, config)
            
            # Save summary.md - Human-readable summary
            self._save_summary_markdown(exp_dir / "summary.md", results, config)
            
            # Save config.yaml - Experiment configuration
            import yaml
            config_file = exp_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print(f"✅ Results saved to: {exp_dir}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save results: {e}")
    
    def _save_metrics_csv(self, filepath: Path, results: Dict[str, Any], config: Dict[str, Any]):
        """Save key metrics to CSV format."""
        import csv
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Experiment', config.get('experiment', 'unknown')],
            ['Dataset', config.get('dataset', {}).get('name', 'unknown')],
            ['Model', config.get('model', {}).get('architecture', 'unknown')],
            ['Privacy', results.get('privacy_technique', 'None')],
            ['Accuracy', f"{results.get('accuracy', 0):.4f}"],
            ['F1 Score', f"{results.get('f1_score', 0):.4f}"],
            ['ROC AUC', f"{results.get('roc_auc', 0):.4f}"],
            ['Duration (s)', f"{results.get('duration', 0):.1f}"],
            ['Energy (kWh)', f"{results.get('energy_consumed', 0):.6f}"],
            ['CO2 (kg)', f"{results.get('co2_emissions', 0):.6f}"]
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(metrics_data)
    
    def _save_summary_markdown(self, filepath: Path, results: Dict[str, Any], config: Dict[str, Any]):
        """Save human-readable summary in Markdown format."""
        experiment = config.get('experiment', 'unknown')
        dataset = config.get('dataset', {}).get('name', 'unknown')
        model = config.get('model', {}).get('architecture', 'unknown')
        privacy = results.get('privacy_technique', 'None (Baseline)')
        
        summary = f"""# PrivacyBench Experiment Results

## Experiment Details
- **Experiment:** {experiment}
- **Dataset:** {dataset}
- **Model:** {model}
- **Privacy Technique:** {privacy}
- **Duration:** {results.get('duration', 0):.1f} seconds

## Performance Metrics
- **Accuracy:** {results.get('accuracy', 0):.2%}
- **F1 Score:** {results.get('f1_score', 0):.4f}
- **ROC AUC:** {results.get('roc_auc', 0):.4f}

## Resource Consumption
- **Training Time:** {results.get('duration', 0):.1f} seconds
- **Energy Consumed:** {results.get('energy_consumed', 0):.6f} kWh
- **CO2 Emissions:** {results.get('co2_emissions', 0):.6f} kg

## Configuration
```yaml
{self._config_to_yaml_string(config)}
```

## Notes
"""
        
        if results.get('simulation_mode'):
            summary += "- Results generated in simulation mode\n"
        
        summary += f"- Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        with open(filepath, 'w') as f:
            f.write(summary)
    
    def _config_to_yaml_string(self, config: Dict[str, Any]) -> str:
        """Convert config to YAML string for markdown."""
        try:
            import yaml
            return yaml.dump(config, default_flow_style=False).strip()
        except ImportError:
            return str(config)


# =============================================================================
# USAGE EXAMPLES AND TESTING
# =============================================================================

def test_phase2_run_command():
    """Test function for Phase 2 run command."""
    print("Testing Phase 2 Run Command...")
    
    # Mock arguments for testing
    class MockArgs:
        experiment = 'cnn_baseline'
        dataset = 'alzheimer'
        dry_run = False
        verbose = True
        output = './test_results'
    
    args = MockArgs()
    
    # Test dry run
    print("\n1. Testing dry-run mode:")
    args.dry_run = True
    run_cmd = RunCommand()
    result = run_cmd.execute(args)
    print(f"Dry-run result: {result}")
    
    # Test actual execution (simulation)
    print("\n2. Testing actual execution:")
    args.dry_run = False
    result = run_cmd.execute(args)
    print(f"Execution result: {result}")


if __name__ == "__main__":
    test_phase2_run_command()