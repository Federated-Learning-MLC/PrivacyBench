"""
PrivacyBench CLI - List Command
===============================

Provides the 'list' command functionality to display available:
- Experiments
- Datasets  
- Models
- Privacy techniques

FIXED: Handles experiments.yaml loading without item_name parameter
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any
from tabulate import tabulate

class ListCommand:
    """Handles the 'list' command and its subcommands."""
    
    def __init__(self):
        """Initialize the list command."""
        self.legacy_experiments_path = Path("legacy/experiments.yaml")
    
    def execute(self, list_type: str, verbose: bool = False) -> int:
        """Execute the list command."""
        if verbose:
            print(f"📋 Listing {list_type}...")
            
        if list_type == "experiments":
            return self._list_experiments(verbose)
        elif list_type == "datasets":
            return self._list_datasets(verbose)
        elif list_type == "models":
            return self._list_models(verbose)
        elif list_type == "privacy":
            return self._list_privacy_techniques(verbose)
        else:
            print(f"❌ Unknown list type: {list_type}")
            return 1
    
    def _list_experiments(self, verbose: bool = False) -> int:
        """List available experiments."""
        print("\n📋 Available Experiments:")
        print("="*84)
        
        # Try to load experiments from YAML file
        experiments_data = []
        yaml_loaded = False
        
        try:
            experiments = self._load_experiments_yaml()
            yaml_loaded = True
            
            # Map experiment names to CLI names
            experiment_mapping = {
                "CNN Baseline": "cnn_baseline",
                "ViT Baseline": "vit_baseline", 
                "FL (CNN)": "fl_cnn",
                "FL (ViT)": "fl_vit",
                "DP (CNN)": "dp_cnn",
                "DP (ViT)": "dp_vit",
                "FL + SMPC (CNN)": "fl_smpc_cnn",
                "FL + SMPC (ViT)": "fl_smpc_vit",
                "FL + CDP-SF (CNN)": "fl_cdp_sf_cnn",
                "FL + CDP-SF (ViT)": "fl_cdp_sf_vit",
                "FL + CDP-SA (CNN)": "fl_cdp_sa_cnn",
                "FL + CDP-SA (ViT)": "fl_cdp_sa_vit",
                "FL + CDP-CF (CNN)": "fl_cdp_cf_cnn",
                "FL + CDP-CF (ViT)": "fl_cdp_cf_vit",
                "FL + CDP-CA (CNN)": "fl_cdp_ca_cnn",
                "FL + CDP-CA (ViT)": "fl_cdp_ca_vit",
                "FL + LDP-Mod (CNN)": "fl_ldp_mod_cnn",
                "FL + LDP-Mod (ViT)": "fl_ldp_mod_vit",
                "FL + LDP-PE (CNN)": "fl_ldp_pe_cnn",
                "FL + LDP-PE (ViT)": "fl_ldp_pe_vit",
            }
            
            for exp in experiments:
                if isinstance(exp, dict):
                    exp_name = exp.get("name", "Unknown")
                    cli_name = experiment_mapping.get(exp_name, exp_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").replace("-", "_"))
                    
                    # Determine experiment type
                    exp_type = self._categorize_experiment(exp_name)
                    
                    # Determine model type  
                    model = "CNN" if "CNN" in exp_name else "ViT" if "ViT" in exp_name else "Unknown"
                    
                    experiments_data.append([
                        cli_name,
                        exp_name,
                        exp_type,
                        model,
                        exp.get("epochs", "50"),
                        exp.get("batch_size", "32"),
                        exp.get("learning_rate", "0.0002")
                    ])
        
        except Exception as e:
            print(f"⚠️  Could not load experiments from {self.legacy_experiments_path}: {e}")
            yaml_loaded = False
        
        # Show results or fallback
        if yaml_loaded and experiments_data:
            headers = ["CLI Name", "Experiment Name", "Type", "Model", "Epochs", "Batch Size", "Learning Rate"]
            print(tabulate(experiments_data, headers=headers, tablefmt="grid"))
            print(f"\n✅ Found {len(experiments_data)} experiments from {self.legacy_experiments_path}")
        else:
            print("⚠️  Using fallback experiment list (experiments.yaml not loaded)")
        
        # Always show available CLI experiment names
        print(f"\n📝 Available CLI experiment names:")
        cli_experiments = [
            "cnn_baseline", "vit_baseline", 
            "fl_cnn", "fl_vit",
            "dp_cnn", "dp_vit",
            "fl_smpc_cnn", "fl_smpc_vit",
            "fl_cdp_sf_cnn", "fl_cdp_sf_vit",
            "fl_cdp_sa_cnn", "fl_cdp_sa_vit",
            "fl_cdp_cf_cnn", "fl_cdp_cf_vit", 
            "fl_cdp_ca_cnn", "fl_cdp_ca_vit",
            "fl_ldp_mod_cnn", "fl_ldp_mod_vit",
            "fl_ldp_pe_cnn", "fl_ldp_pe_vit"
        ]
        
        for exp in cli_experiments:
            print(f"  • {exp}")
        
        if verbose:
            print(f"\n💡 Usage Examples:")
            print(f"   privacybench run --experiment cnn_baseline --dataset alzheimer")
            print(f"   privacybench run --experiment fl_cnn --dataset skin_lesions") 
            print(f"   privacybench run --experiment vit_baseline --dataset alzheimer --dry-run")
        
        return 0
    
    def _load_experiments_yaml(self) -> List[Dict[str, Any]]:
        """Load experiments from YAML file with flexible structure handling."""
        try:
            with open(self.legacy_experiments_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            # Handle different YAML structures
            if isinstance(yaml_content, dict):
                # Structure: { "experiments": [...] }
                if "experiments" in yaml_content:
                    experiments_list = yaml_content["experiments"]
                    if isinstance(experiments_list, list):
                        return experiments_list
                    else:
                        # experiments is a dict, convert to list
                        return [{"name": k, **v} for k, v in experiments_list.items()]
                else:
                    # Structure: { "exp1": {...}, "exp2": {...} }
                    return [{"name": k, **v} for k, v in yaml_content.items()]
            
            # Structure: [...] (direct list)
            elif isinstance(yaml_content, list):
                return yaml_content
            
            else:
                raise ValueError(f"Unsupported YAML structure: {type(yaml_content)}")
                
        except FileNotFoundError:
            raise ValueError(f"Experiments file not found: {self.legacy_experiments_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    
    def _categorize_experiment(self, exp_name: str) -> str:
        """Categorize experiment type based on name."""
        exp_name_upper = exp_name.upper()
        
        if "FL" in exp_name_upper and ("CDP" in exp_name_upper or "LDP" in exp_name_upper):
            return "Federated Learning + Differential Privacy"
        elif "FL" in exp_name_upper and "SMPC" in exp_name_upper:
            return "Federated Learning + SMPC"
        elif "FL" in exp_name_upper:
            return "Federated Learning"
        elif ("DP" in exp_name_upper or "CDP" in exp_name_upper or "LDP" in exp_name_upper):
            return "Differential Privacy"
        elif "SMPC" in exp_name_upper:
            return "Secure Multi-Party Computation"
        else:
            return "Baseline"
    
    def _list_datasets(self, verbose: bool = False) -> int:
        """List available datasets."""
        print("\n📊 Available Datasets")
        print("="*40)
        
        datasets_data = [
            ["alzheimer", "Medical Imaging", "Alzheimer's Disease Classification", "Binary", "MRI Images"],
            ["skin_lesions", "Medical Imaging", "Skin Lesion Classification", "Multi-class", "Dermoscopic Images"],
        ]
        
        headers = ["Dataset Name", "Domain", "Description", "Task Type", "Data Type"]
        print(tabulate(datasets_data, headers=headers, tablefmt="grid"))
        print(f"\n✅ Found {len(datasets_data)} datasets")
        
        if verbose:
            print(f"\n💡 Datasets are automatically downloaded and preprocessed")
            print(f"📁 Data will be stored in: ./data/")
        
        return 0
    
    def _list_models(self, verbose: bool = False) -> int:
        """List available model architectures."""
        print("\n🧠 Available Models")
        print("="*35)
        
        models_data = [
            ["CNN", "Convolutional Neural Network", "Image Classification", "Computer Vision", "PyTorch"],
            ["ViT", "Vision Transformer", "Image Classification", "Computer Vision", "Transformers"],
        ]
        
        headers = ["Model", "Full Name", "Task", "Domain", "Framework"]
        print(tabulate(models_data, headers=headers, tablefmt="grid"))
        print(f"\n✅ Found {len(models_data)} model architectures")
        
        if verbose:
            print(f"\n🔧 Models support both baseline and privacy-preserving training")
            print(f"⚙️  Architectures are automatically configured based on dataset")
        
        return 0
    
    def _list_privacy_techniques(self, verbose: bool = False) -> int:
        """List available privacy-preserving techniques."""
        print("\n🔒 Available Privacy Techniques")
        print("="*45)
        
        privacy_data = [
            ["FL", "Federated Learning", "Distributed training across clients", "Medium", "High"],
            ["DP", "Differential Privacy", "Noise injection for privacy", "High", "Medium"],
            ["SMPC", "Secure Multi-Party Computation", "Cryptographic secure computation", "Very High", "Low"],
            ["CDP-SF", "Central DP + Secure FL", "Combined DP and FL", "High", "Medium"],
            ["CDP-SA", "Central DP + Secure Aggregation", "Adaptive central DP", "High", "Medium"],
            ["CDP-CF", "Central DP + Client Filtering", "Client-filtered central DP", "High", "Medium"], 
            ["CDP-CA", "Central DP + Client Aggregation", "Client-aggregated central DP", "High", "Medium"],
            ["LDP-Mod", "Local DP + Modified", "Modified local differential privacy", "Very High", "Medium"],
            ["LDP-PE", "Local DP + Privacy Engine", "Privacy engine local DP", "Very High", "Medium"],
            ["FL+SMPC", "FL + Secure Multi-Party Computation", "Federated learning with SMPC", "Very High", "Low"],
        ]
        
        headers = ["Technique", "Full Name", "Description", "Privacy Level", "Efficiency"]
        print(tabulate(privacy_data, headers=headers, tablefmt="grid"))
        print(f"\n✅ Found {len(privacy_data)} privacy techniques")
        
        if verbose:
            print(f"\n🔐 Privacy Level: How much privacy protection is provided")
            print(f"⚡ Efficiency: Computational and communication efficiency")
            print(f"🔄 Techniques can be combined for hybrid approaches")
        
        return 0