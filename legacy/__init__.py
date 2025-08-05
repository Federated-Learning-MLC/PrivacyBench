"""
Legacy Module - Preserved Existing PrivacyBench Code
All your existing src/ code is preserved here unchanged

This module maintains 100% compatibility with existing notebooks
while allowing the new CLI framework to delegate to proven implementations.
"""

# Import all existing functionality to maintain compatibility
try:
    from .config import *
    from .local_utility import *
    from .train import *
    
    # Optional imports (only if files exist)
    try:
        from .federated import *
    except ImportError:
        pass
    
    try:
        from .privacy_engine import *
    except ImportError:
        pass
    
    try:
        from .tracker import *
    except ImportError:
        pass
    
    try:
        from .FL_client import *
    except ImportError:
        pass
    
    try:
        from .paths import *
    except ImportError:
        pass

except ImportError as e:
    print(f"Warning: Could not import some legacy modules: {e}")
    print("This is normal if you haven't moved your src/ files to legacy/ yet")

# Make key components available for CLI framework
__all__ = [
    # Export main functions that CLI will use
    "load_yaml_config",
    "train_model", 
    "ExperimentName",
    "set_seed",
    "load_data",
    # Add other key functions as needed
]