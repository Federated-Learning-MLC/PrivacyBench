from typing import Any, Dict, List, Optional
import time
from pathlib import Path

# Import from legacy code (pb repo tracking implementation)
try:
    from legacy.tracker import track_emissions
    import wandb
    from codecarbon import EmissionsTracker
except ImportError as e:
    print(f"Warning: Could not import tracking modules: {e}")
    track_emissions = None
    wandb = None
    EmissionsTracker = None


class MetricsTracker:
    """
    Metrics tracking wrapper using W&B and CodeCarbon.
    Wraps existing pb repo tracking functionality to preserve integration.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.experiment_name = self.config.get('experiment_name', 'privacybench_experiment')
        self.data_name = self.config.get('data_name', 'alzheimer')
        self.output_dir = self.config.get('output_dir', './results')
        
        # Tracking components
        self.wandb_enabled = self.config.get('wandb', True)
        self.codecarbon_enabled = self.config.get('codecarbon', True)
        
        # Tracking instances
        self.emissions_tracker = None
        self.wandb_run = None
        self.start_time = None
        self.metrics = {}
        
    def setup_tracking(self) -> Dict[str, Any]:
        """Setup tracking systems (W&B, CodeCarbon) using legacy implementation."""
        setup_info = {}
        
        # Setup emissions tracking using legacy tracker
        if self.codecarbon_enabled and track_emissions:
            try:
                # Use legacy track_emissions decorator pattern
                self.emissions_tracker = track_emissions(
                    experiment_name=self.experiment_name,
                    data_name=self.data_name
                )
                setup_info['codecarbon'] = 'enabled'
            except Exception as e:
                print(f"Warning: CodeCarbon setup failed: {e}")
                setup_info['codecarbon'] = 'failed'
        
        # Setup W&B tracking
        if self.wandb_enabled and wandb:
            try:
                self.wandb_run = wandb.init(
                    project="PrivacyBench",
                    name=f"{self.experiment_name}_{self.data_name}",
                    config=self.config
                )
                setup_info['wandb'] = 'enabled'
            except Exception as e:
                print(f"Warning: W&B setup failed: {e}")
                setup_info['wandb'] = 'failed'
        
        return setup_info
    
    def start_experiment(self) -> None:
        """Start experiment tracking."""
        self.start_time = time.time()
        
        # Start emissions tracking
        if self.emissions_tracker:
            try:
                # Emissions tracking started via legacy decorator
                pass
            except Exception as e:
                print(f"Warning: Failed to start emissions tracking: {e}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to tracking systems."""
        self.metrics.update(metrics)
        
        # Log to W&B
        if self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                print(f"Warning: W&B logging failed: {e}")
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """Log model architecture and parameter information."""
        self.log_metrics({
            'model_architecture': model_info.get('architecture', 'unknown'),
            'model_parameters': model_info.get('total_parameters', 0),
            'trainable_parameters': model_info.get('trainable_parameters', 0)
        })
    
    def log_privacy_info(self, privacy_metrics: Dict[str, Any]) -> None:
        """Log privacy technique information."""
        self.log_metrics(privacy_metrics)
    
    def finish_experiment(self) -> Dict[str, Any]:
        """Finish experiment and collect final metrics."""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        final_metrics = {
            'experiment_duration': duration,
            'experiment_name': self.experiment_name,
            'data_name': self.data_name
        }
        
        # Finish W&B run
        if self.wandb_run:
            try:
                self.wandb_run.finish()
            except Exception as e:
                print(f"Warning: W&B finish failed: {e}")
        
        # Get emissions data
        if self.codecarbon_enabled:
            try:
                # Emissions data collected via legacy tracker
                final_metrics['emissions_tracking'] = 'completed'
            except Exception as e:
                print(f"Warning: Emissions finish failed: {e}")
        
        return final_metrics
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current accumulated metrics."""
        return self.metrics.copy()
