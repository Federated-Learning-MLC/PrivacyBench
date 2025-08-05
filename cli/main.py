# CLI - Main entry point
# Converts: jupyter notebook → privacybench run command

import sys
import argparse
from pathlib import Path
from typing import Optional

from cli.commands.run import RunCommand
from cli.commands.list import ListCommand
from cli.commands.validate import ValidateCommand


def setup_parser() -> argparse.ArgumentParser:
    # main CLI setup argument parser
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
        version="PrivacyBench 1.0.0"
    )
    
    # subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command", 
        help="Available commands",
        metavar="COMMAND"
    )
    
    # run command
    run_parser = subparsers.add_parser(
        "run", 
        help="Run privacy-preserving ML experiments",
        description="Execute experiments from CLI instead of notebooks"
    )
    setup_run_parser(run_parser)
    
    # list command  
    list_parser = subparsers.add_parser(
        "list",
        help="List available components",
        description="Display available experiments, datasets, models, and privacy techniques"
    )
    setup_list_parser(list_parser)
    
    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate experiment configurations", 
        description="Check YAML configuration files for errors"
    )
    setup_validate_parser(validate_parser)
    
    return parser


def setup_run_parser(parser: argparse.ArgumentParser) -> None:
    
    """Setup run command arguments"""
    
    parser.add_argument(
        "--experiment",
        choices=[
            # CNN exps
            "cnn_baseline", "cnn_base",
            # ViT exps  
            "vit_baseline", "vit_base",
            # FL exps
            "fl_cnn", "fl_vit", "fl_cnn_base", "fl_vit_base",
            # DP exps
            "dp_cnn", "dp_vit", "dp_cnn_base", "dp_vit_base", 
            # Hybrid exps
            "fl_dp_cnn", "fl_smpc_cnn", "fl_cdp_sf_cnn",
            # SMPC exps
            "smpc_cnn", "smpc_vit"
        ],
        help="Experiment to run (maps to your existing notebooks)"
    )
    
    # dataset selection (existing datasets)
    parser.add_argument(
        "--dataset",
        choices=["alzheimer", "skin_lesions"],
        required=True,
        help="Dataset to use for training"
    )
    
    # config file
    parser.add_argument(
        "--config",
        type=Path,
        help="Custom YAML configuration file"
    )
    
    # output dir
    parser.add_argument(
        "--output",
        type=Path,
        default="./results",
        help="Output directory for results (default: ./results)"
    )
    
    # dry run (validate without execution)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running experiment"
    )
    
    # Additional options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU if available (default: True)"
    )


def setup_list_parser(parser: argparse.ArgumentParser) -> None:
    
    """Setup list command arguments"""
    
    parser.add_argument(
        "component",
        choices=["experiments", "datasets", "models", "privacy", "all"],
        nargs="?",
        default="all",
        help="Component type to list (default: all)"
    )


def setup_validate_parser(parser: argparse.ArgumentParser) -> None:
    
    """Setup validate command arguments"""
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration file to validate"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed validation information"
    )


def route_command(args: argparse.Namespace) -> int:
    
    """Route parsed arguments to appropriate command handler"""
    
    try:
        if args.command == "run":
            cmd = RunCommand()
            return cmd.execute(args)
        elif args.command == "list":
            cmd = ListCommand()
            return cmd.execute(args)
        elif args.command == "validate":
            cmd = ValidateCommand()
            return cmd.execute(args)
        else:
            print("❌ Error: No command specified. Use --help for usage information.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    
    """Main CLI entry point called by 'privacybench' command"""
    
    parser = setup_parser()
    args = parser.parse_args()
    
    # show help if no command specified
    if not args.command:
        parser.print_help()
        return 0
    
    # route to appropriate command handler
    return route_command(args)

if __name__ == "__main__":
    sys.exit(main())