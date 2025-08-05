"""
List command to show available experiments, datasets, models
"""
import argparse
from tabulate import tabulate
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directories to path to import legacy modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from legacy.local_utility import load_yaml_config


class ListCommand:
    """Handles the 'list' command"""
    
    def __init__(self):
        self.legacy_experiments_path = Path(__file__).parent.parent.parent / "legacy" / "experiments.yaml"
    
    def execute(self, args: argparse.Namespace) -> int:
        """Execute list command"""
        try:
            if args.component == "experiments" or args.component == "all":
                self._list_experiments()
            
            if args.component == "datasets" or args.component == "all":
                self._list_datasets()
            
            if args.component == "models" or args.component == "all":
                self._list_models()
            
            if args.component == "privacy" or args.component == "all":
                self._list_privacy_techniques()
            
            return 0
            
        except Exception as e:
            print(f"❌ Error listing components: {e}")
            return 1
    
    def _list_experiments(self):
        """List all available experiments from your existing experiments.yaml"""
        print("\n📋 Available Experiments:")
        print("=" * 80)
        
        try:
            # Load your existing experiments
            experiments = load_yaml_config(
                yaml_path=self.legacy_experiments_path,
                key="experiments"
            )
            
            headers = ["CLI Name", "Experiment Name", "Type", "Model", "Epochs", "Batch Size", "Learning Rate"]
            rows = []
            
            # Map your existing experiments to CLI names
            experiment_mapping = {
                "CNN Baseline": "cnn_baseline",
                "ViT Baseline": "vit_baseline", 
                "FL CNN": "fl_cnn",
                "FL ViT": "fl_vit",
                "DP CNN": "dp_cnn",
                "DP ViT": "dp_vit",
                "FL+DP CNN": "fl_dp_cnn",
                "SMPC CNN": "smpc_cnn",
            }
            
            for exp in experiments:
                exp_name = exp.get("name", "Unknown")
                cli_name = experiment_mapping.get(exp_name, exp_name.lower().replace(" ", "_"))
                
                # Determine experiment type based on name
                exp_type = "Baseline"
                if "FL" in exp_name and "DP" in exp_name:
                    exp_type = "Federated Learning + Differential Privacy"
                elif "FL" in exp_name:
                    exp_type = "Federated Learning"
                elif "DP" in exp_name:
                    exp_type = "Differential Privacy"
                elif "SMPC" in exp_name:
                    exp_type = "Secure Multi-Party Computation"
                
                # Determine model type
                model = "CNN" if "CNN" in exp_name else "ViT" if "ViT" in exp_name else "Unknown"
                
                rows.append([
                    cli_name,
                    exp_name,
                    exp_type,
                    model,
                    exp.get("epochs", "N/A"),
                    exp.get("batch_size", "N/A"),
                    exp.get("learning_rate", "N/A")
                ])
            
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            print(f"\nTotal: {len(rows)} experiments available")
            
            # Show usage examples
            print("\n💡 Usage Examples:")
            print("  privacybench run --experiment cnn_baseline --dataset alzheimer")
            print("  privacybench run --experiment fl_cnn --dataset skin_lesions")
            print("  privacybench run --experiment vit_baseline --dataset alzheimer")
            
        except Exception as e:
            print(f"⚠️  Could not load experiments from {self.legacy_experiments_path}: {e}")
            print("\n📝 Available CLI experiment names:")
            # Fallback list
            cli_experiments = [
                "cnn_baseline", "vit_baseline", 
                "fl_cnn", "fl_vit",
                "dp_cnn", "dp_vit",
                "fl_dp_cnn", "smpc_cnn"
            ]
            for exp in cli_experiments:
                print(f"  • {exp}")
    
    def _list_datasets(self):
        """List available datasets"""
        print("\n📊 Available Datasets:")
        print("=" * 60)
        
        datasets = [
            ["alzheimer", "Alzheimer MRI Classification", "4", "~6,400 images", "Medical imaging"],
            ["skin_lesions", "ISIC Skin Lesion Classification", "8", "~10,000 images", "Medical imaging"]
        ]
        
        headers = ["Name", "Description", "Classes", "Size", "Type"]
        print(tabulate(datasets, headers=headers, tablefmt="grid"))
        
        print("\n💡 Usage:")
        print("  privacybench run --experiment cnn_baseline --dataset alzheimer")
        print("  privacybench run --experiment vit_baseline --dataset skin_lesions")
    
    def _list_models(self):
        """List available models"""
        print("\n🧠 Available Models:")
        print("=" * 60)
        
        models = [
            ["cnn", "ResNet18", "11.2M", "Convolutional Neural Network", "Fast training, good baseline"],
            ["vit", "ViT-Base/16", "86.6M", "Vision Transformer", "State-of-the-art accuracy, slower training"]
        ]
        
        headers = ["Name", "Architecture", "Parameters", "Description", "Notes"]
        print(tabulate(models, headers=headers, tablefmt="grid"))
        
        print("\n💡 Usage:")
        print("  privacybench run --experiment cnn_baseline --dataset alzheimer  # Uses CNN")
        print("  privacybench run --experiment vit_baseline --dataset alzheimer  # Uses ViT")
    
    def _list_privacy_techniques(self):
        """List available privacy techniques"""
        print("\n🔒 Available Privacy Techniques:")
        print("=" * 80)
        
        techniques = [
            ["baseline", "No Privacy", "Standard training without privacy", "None", "Fastest"],
            ["fl", "Federated Learning", "Distributed training without sharing raw data", "Data locality", "Moderate"],
            ["dp", "Differential Privacy", "Mathematical privacy guarantees via noise injection", "ε-differential privacy", "Slow"],
            ["smpc", "Secure Multi-Party Computation", "Cryptographic secure aggregation", "Cryptographic", "Slowest"],
            ["fl+dp", "FL + Differential Privacy", "Combined federated training with privacy guarantees", "Strong privacy", "Very slow"],
            ["fl+smpc", "FL + SMPC", "Federated training with cryptographic security", "Very strong privacy", "Very slow"]
        ]
        
        headers = ["CLI Name", "Full Name", "Description", "Privacy Guarantee", "Performance"]
        print(tabulate(techniques, headers=headers, tablefmt="grid"))
        
        print("\n💡 Usage Examples:")
        print("  privacybench run --experiment cnn_baseline --dataset alzheimer   # No privacy")
        print("  privacybench run --experiment fl_cnn --dataset alzheimer        # Federated learning")
        print("  privacybench run --experiment dp_cnn --dataset alzheimer        # Differential privacy")
        print("  privacybench run --experiment fl_dp_cnn --dataset alzheimer     # FL + DP combination")
        
        print("\n⚠️  Privacy-Utility Trade-offs:")
        print("  • Higher privacy → Lower accuracy")
        print("  • More techniques → Slower training")
        print("  • Baseline typically achieves best accuracy")
        print("  • FL+DP provides strongest privacy but slowest training")
    
    def _get_experiment_stats(self) -> Dict[str, int]:
        """Get statistics about available experiments"""
        try:
            experiments = load_yaml_config(
                yaml_path=self.legacy_experiments_path,
                key="experiments"
            )
            
            stats = {
                "total": len(experiments),
                "baseline": 0,
                "federated": 0,
                "privacy": 0,
                "hybrid": 0
            }
            
            for exp in experiments:
                name = exp.get("name", "")
                if "FL" in name and "DP" in name:
                    stats["hybrid"] += 1
                elif "FL" in name:
                    stats["federated"] += 1
                elif "DP" in name:
                    stats["privacy"] += 1
                else:
                    stats["baseline"] += 1
            
            return stats
            
        except Exception:
            return {"total": 0, "baseline": 0, "federated": 0, "privacy": 0, "hybrid": 0}