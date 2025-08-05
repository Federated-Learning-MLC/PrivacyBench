import sys
from pathlib import Path
from tabulate import tabulate
from typing import Dict, List, Any

from cli.parser import experiment_mapping, list_available_configs


class ListCommand:
    """Enhanced list command with Phase 3 individual config support."""
    
    def execute(self, args) -> int:
        """Execute list command with support for individual configs."""
        
        list_type = args.list_type if hasattr(args, 'list_type') else 'all'
        
        if list_type == 'experiments':
            return self._list_experiments()
        elif list_type == 'datasets':
            return self._list_datasets()
        elif list_type == 'models':
            return self._list_models()
        elif list_type == 'privacy':
            return self._list_privacy()
        elif list_type == 'configs':
            return self._list_individual_configs()
        else:
            return self._list_all()
    
    def _list_all(self) -> int:
        """List all available components and configurations."""
        
        print("🔍 PrivacyBench - Available Components")
        print("=" * 50)
        
        self._list_experiments()
        print()
        self._list_datasets() 
        print()
        self._list_models()
        print()
        self._list_privacy()
        print()
        self._list_individual_configs()
        
        return 0
    
    def _list_experiments(self) -> int:
        """List all available experiments (legacy + individual)."""
        
        print("📋 Available Experiments:")
        
        # Legacy experiments (Phase 1)
        legacy_data = []
        for cli_name, enum_name in experiment_mapping.items():
            exp_type = self._classify_experiment_type(cli_name)
            model = "CNN" if "cnn" in cli_name else "ViT" if "vit" in cli_name else "Mixed"
            legacy_data.append([cli_name, str(enum_name), exp_type, model, "Legacy"])
        
        if legacy_data:
            print("\n   Legacy Experiments (CLI arguments):")
            headers = ["CLI Name", "Experiment Name", "Type", "Model", "Source"]
            print(tabulate(legacy_data, headers=headers, tablefmt="grid"))
        
        # Individual config files (Phase 3)
        individual_configs = list_available_configs()
        
        individual_data = []
        for category, configs in individual_configs.items():
            if category == 'legacy':
                continue
            for config in configs:
                individual_data.append([config, category.title(), self._detect_model_from_config(config), "Individual"])
        
        if individual_data:
            print("\n   Individual Configuration Files:")
            headers = ["Config Name", "Category", "Model", "Source"]
            print(tabulate(individual_data, headers=headers, tablefmt="grid"))
        
        return 0
    
    def _list_datasets(self) -> int:
        """List available datasets."""
        
        print("📊 Available Datasets:")
        
        datasets_data = [
            ["alzheimer", "Alzheimer MRI Classification", "4", "~6,400", "Medical Imaging"],
            ["skin_lesions", "ISIC Skin Lesion Classification", "8", "~10,000", "Medical Imaging"]
        ]
        
        headers = ["Name", "Description", "Classes", "Samples", "Domain"]
        print(tabulate(datasets_data, headers=headers, tablefmt="grid"))
        
        return 0
    
    def _list_models(self) -> int:
        """List available model architectures."""
        
        print("🧠 Available Models:")
        
        models_data = [
            ["cnn", "ResNet18", "Convolutional Neural Network", "~11M", "Fast"],
            ["vit", "ViT-Base/16", "Vision Transformer", "~86M", "High Accuracy"]
        ]
        
        headers = ["Architecture", "Model", "Type", "Parameters", "Notes"]
        print(tabulate(models_data, headers=headers, tablefmt="grid"))
        
        return 0
    
    def _list_privacy(self) -> int:
        """List available privacy techniques."""
        
        print("🔒 Available Privacy Techniques:")
        
        privacy_data = [
            ["federated_learning", "Federated Learning", "Moderate", "Distributed training without data sharing"],
            ["differential_privacy", "Differential Privacy", "High", "Mathematical privacy guarantees via noise"],
            ["secure_multiparty_computation", "Secure MPC", "Very High", "Cryptographic secure computation"]
        ]
        
        headers = ["Technique", "Name", "Privacy Level", "Description"]
        print(tabulate(privacy_data, headers=headers, tablefmt="grid"))
        
        return 0
    
    def _list_individual_configs(self) -> int:
        """List individual configuration files by category."""
        
        print("📁 Individual Configuration Files:")
        
        available_configs = list_available_configs()
        
        for category, configs in available_configs.items():
            if category == 'legacy' or not configs:
                continue
            
            print(f"\n   {category.title()}:")
            for config in configs:
                config_path = f"configs/experiments/{category}/{config}.yaml"
                print(f"      • {config:<30} ({config_path})")
        
        if Path("configs/experiments").exists():
            print(f"\n💡 Usage:")
            print(f"   privacybench run --config cnn_alzheimer")
            print(f"   privacybench run --config configs/experiments/baselines/cnn_alzheimer.yaml")
            print(f"   privacybench validate --config configs/experiments/privacy/dp_configurations.yaml")
        else:
            
