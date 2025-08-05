#!/usr/bin/env python3
"""
PrivacyBench CLI - Main Entry Point
====================================

This is the main entry point for the PrivacyBench command-line interface.
It provides commands for running privacy-preserving machine learning experiments.

Usage:
    privacybench --help
    privacybench list experiments
    privacybench run --config configs/experiments/baselines/cnn_alzheimer.yaml
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Import CLI commands with error handling
try:
    from cli.commands.list import ListCommand
except ImportError as e:
    print(f"Warning: Could not import ListCommand: {e}")
    ListCommand = None

try:
    from cli.commands.run import RunCommand
except ImportError as e:
    print(f"Warning: Could not import RunCommand: {e}")
    RunCommand = None

try:
    from cli.commands.validate import ValidateCommand
except ImportError as e:
    print(f"Warning: Could not import ValidateCommand: {e}")
    ValidateCommand = None

__version__ = "1.0.0"

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="privacybench",
        description="PPML Privacy-utility-cost Benchmarking Framework for Efficiency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                    Examples:
                    privacybench list experiments
                    privacybench run --experiment cnn_baseline --dataset alzheimer
                    privacybench run --experiment fl_cnn --dataset alzheimer
                    privacybench validate --config configs/experiments/baselines/cnn_alzheimer.yaml
                    
                    For more information, visit: ...
                """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"%(prog)s {__version__}"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="Available commands",
        dest="command",
        help="Available commands"
    )
    
    # LIST command - FIXED: Proper structure for "privacybench list experiments"
    list_parser = subparsers.add_parser(
        "list",
        help="List available components",
        description="Display formatted tables of available options"
    )
    list_parser.add_argument(
        "component",
        choices=["experiments", "datasets", "models", "privacy"],
        help="Component type to list"
    )
    
    # VALIDATE command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate experiment configurations",
        description="Check configuration syntax and required fields"
    )
    validate_parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to configuration file to validate"
    )
    
    # RUN command
    run_parser = subparsers.add_parser(
        "run",
        help="Run privacy-preserving ML experiments",
        description="Execute experiments with specified configurations"
    )
    
    # Run command arguments - mutually exclusive group
    run_group = run_parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to YAML configuration file"
    )
    run_group.add_argument(
        "--experiment", "-e",  
        type=str,
        choices=[
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
        ],
        help="Experiment name (requires --dataset)"
    )
    
    # Additional run arguments
    run_parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["alzheimer", "skin_lesions"],
        help="Dataset name (required with --experiment)"
    )
    run_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory for results (default: ./results)"
    )
    run_parser.add_argument(
        "--gpu", "-g",
        type=int,
        help="GPU device ID to use (default: auto-detect)"
    )
    run_parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )
    
    return parser

def validate_run_args(args) -> bool:
    """Validate run command arguments."""
    if args.experiment and not args.dataset:
        print("Error: --dataset is required when using --experiment")
        return False
    
    if args.config and not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        return False
        
    return True

def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
        
    args = parser.parse_args()
    
    try:
        # Route to appropriate command handler
        if args.command == "list":
            if ListCommand is None:
                print("❌ List command not available - import error")
                return 1
            command = ListCommand()
            return command.execute(args.component, verbose=args.verbose)
            
        elif args.command == "validate":
            if ValidateCommand is None:
                print("❌ Validate command not available - import error")
                return 1
            command = ValidateCommand()
            return command.execute(args.config, verbose=args.verbose)
            
        elif args.command == "run":
            if not validate_run_args(args):
                return 1
            if RunCommand is None:
                print("❌ Run command not available - import error")
                return 1
            command = RunCommand()
            return command.execute(args)  # FIXED: Remove verbose parameter
            
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        return 130
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())