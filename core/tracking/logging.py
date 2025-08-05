import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class LoggingManager:
    """
    Centralized logging manager for PrivacyBench CLI.
    Provides structured logging with different verbosity levels.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.log_level = self.config.get('log_level', 'INFO')
        self.log_file = self.config.get('log_file', None)
        self.verbose = self.config.get('verbose', False)
        
        self.logger = None
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create logger
        self.logger = logging.getLogger('privacybench')
        self.logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.log_file:
            try:
                Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(self.log_file)
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_format)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Could not setup file logging: {e}")
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def log_experiment_start(self, config: Dict[str, Any]) -> None:
        """Log experiment start with configuration."""
        self.info("=" * 60)
        self.info("🚀 PRIVACYBENCH EXPERIMENT STARTED")
        self.info("=" * 60)
        self.info(f"Experiment: {config.get('experiment', 'unknown')}")
        self.info(f"Dataset: {config.get('dataset', {}).get('name', 'unknown')}")
        self.info(f"Model: {config.get('model', {}).get('architecture', 'unknown')}")
        
        privacy = config.get('privacy', {}).get('techniques', [])
        if privacy:
            privacy_names = [t.get('name', 'unknown') for t in privacy]
            self.info(f"Privacy: {', '.join(privacy_names)}")
        else:
            self.info("Privacy: None (Baseline)")
        
        self.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("-" * 60)
    
    def log_experiment_end(self, results: Dict[str, Any]) -> None:
        """Log experiment completion with results."""
        self.info("-" * 60)
        self.info("🎉 EXPERIMENT COMPLETED SUCCESSFULLY")
        
        if 'accuracy' in results:
            self.info(f"Final Accuracy: {results['accuracy']:.2%}")
        if 'duration' in results:
            self.info(f"Total Duration: {results['duration']:.1f} seconds")
        if 'energy' in results:
            self.info(f"Energy Consumed: {results['energy']:.6f} kWh")
        
        self.info("=" * 60)
    
    def log_progress(self, stage: str, progress: float, message: str = "") -> None:
        """Log progress information."""
        progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
        self.info(f"{stage}: [{progress_bar}] {progress:.1%} {message}")