"""
Legacy Module - Safe Import Handling
====================================

Safely imports all existing functionality from the legacy codebase.
Handles missing dependencies gracefully for Phase 1 CLI functionality.
"""

import warnings
import sys
from pathlib import Path

# Suppress warnings for missing modules during CLI operations
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add legacy directory to path for imports
legacy_dir = Path(__file__).parent
if str(legacy_dir) not in sys.path:
    sys.path.insert(0, str(legacy_dir))

# Safe imports with error handling
def safe_import(module_name, fallback=None):
    """Safely import a module with fallback."""
    try:
        return __import__(module_name)
    except ImportError as e:
        print(f"Warning: Could not import {module_name}: {e}")
        return fallback

# Try importing core legacy modules
try:
    from . import config
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import legacy.config: {e}")
    CONFIG_AVAILABLE = False

try:
    from . import local_utility
    LOCAL_UTILITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import legacy.local_utility: {e}")
    LOCAL_UTILITY_AVAILABLE = False

# Create fallback functions for CLI when legacy modules aren't available
class LegacyFallback:
    """Fallback class when legacy modules aren't available."""
    
    def load_yaml_config(self, key=None):
        """Fallback for load_yaml_config."""
        print(f"Warning: load_yaml_config not available (legacy import failed)")
        return {}
    
    def ExperimentName(self):
        """Fallback for ExperimentName enum."""
        print(f"Warning: ExperimentName not available (legacy import failed)")
        return None

# Export what's available
if LOCAL_UTILITY_AVAILABLE:
    from .local_utility import load_yaml_config
else:
    load_yaml_config = LegacyFallback().load_yaml_config

if CONFIG_AVAILABLE:
    from .config import ExperimentName
else:
    ExperimentName = LegacyFallback().ExperimentName

__all__ = ['load_yaml_config', 'ExperimentName', 'CONFIG_AVAILABLE', 'LOCAL_UTILITY_AVAILABLE']
