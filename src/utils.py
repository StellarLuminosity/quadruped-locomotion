# Utility Functions and Configurations for Go2 Quadruped Training
"""
This module contains utility functions, teleoperation configurations,
and other helper code that doesn't belong in the main config file.
"""

import os
import sys
import shutil
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import torch
import numpy as np


def setup_experiment_directory(exp_name: str, overwrite: bool = False) -> Path:
    """Setup experiment directory with proper handling of existing directories."""
    log_dir = Path(f"logs/{exp_name}")
    
    if log_dir.exists():
        if not overwrite:
            response = input(f"Experiment '{exp_name}' exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                sys.exit(0)
        shutil.rmtree(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def save_experiment_metadata(log_dir: Path, metadata: Dict[str, Any]) -> None:
    """Save experiment metadata for reproducibility."""
    metadata_path = log_dir / "experiment_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


def find_available_checkpoints(exp_name: str) -> List[int]:
    """Find all available model checkpoints for an experiment."""
    log_dir = Path(f"logs/{exp_name}")
    if not log_dir.exists():
        return []
    
    checkpoints = []
    for model_file in log_dir.glob("model_*.pt"):
        try:
            ckpt_num = int(model_file.stem.split('_')[1])
            checkpoints.append(ckpt_num)
        except (ValueError, IndexError):
            continue
    
    return sorted(checkpoints)


def list_available_experiments() -> List[str]:
    """List all available experiments in the logs directory."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return []
    
    experiments = []
    for exp_dir in logs_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / "configs.pkl").exists():
            experiments.append(exp_dir.name)
    
    return sorted(experiments)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_training_args(args) -> None:
    """Validate command line arguments for training."""
    if args.num_envs <= 0:
        raise ValueError("Number of environments must be positive")
    
    if args.max_iterations <= 0:
        raise ValueError("Max iterations must be positive")
    
    if args.device not in ["cpu", "cuda:0"]:
        raise ValueError("Device must be 'cpu' or 'cuda:0'")
    
    if args.device == "cuda:0" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"


def validate_experiment_exists(exp_name: str, ckpt: int = None) -> Tuple[Path, List[int]]:
    """Validate that experiment exists and optionally check for specific checkpoint."""
    log_dir = Path(f"logs/{exp_name}")
    
    if not log_dir.exists():
        available_exps = list_available_experiments()
        if available_exps:
            raise FileNotFoundError(
                f"Experiment '{exp_name}' not found. Available experiments: {available_exps}"
            )
        else:
            raise FileNotFoundError("No experiments found. Run training first.")
    
    config_path = log_dir / "configs.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    available_ckpts = find_available_checkpoints(exp_name)
    
    if ckpt is not None:
        model_path = log_dir / f"model_{ckpt}.pt"
        if not model_path.exists():
            if available_ckpts:
                raise FileNotFoundError(
                    f"Checkpoint {ckpt} not found. Available checkpoints: {available_ckpts}"
                )
            else:
                raise FileNotFoundError(f"No model checkpoints found in {log_dir}")
    
    return log_dir, available_ckpts


# ============================================================================
# DISPLAY AND LOGGING UTILITIES
# ============================================================================

def print_training_header(args) -> None:
    """Print professional training information header."""
    print("\n" + "="*70)
    print("🤖 GO2 QUADRUPED LOCOMOTION TRAINING")
    print("="*70)
    print(f"📊 Experiment: {args.exp_name}")
    print(f"🏃 Environments: {args.num_envs:,}")
    print(f"🔄 Max Iterations: {args.max_iterations:,}")
    print(f"💻 Device: {args.device}")
    print(f"🎓 Curriculum: {'ADAPTIVE' if args.adaptive_curriculum else 'STANDARD'}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def print_evaluation_header(exp_name: str, ckpt: int, mode: str) -> None:
    """Print evaluation information header."""
    print("\n" + "="*70)
    print("🎬 GO2 QUADRUPED EVALUATION")
    print("="*70)
    print(f"📊 Experiment: {exp_name}")
    print(f"🔢 Checkpoint: {ckpt}")
    print(f"🎮 Mode: {mode.upper()}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def print_success_message(message: str, details: Dict[str, Any] = None) -> None:
    """Print a formatted success message."""
    print(f"\n✅ {message}")
    if details:
        for key, value in details.items():
            print(f"   {key}: {value}")


def print_error_message(message: str, suggestion: str = None) -> None:
    """Print a formatted error message with optional suggestion."""
    print(f"\n❌ {message}")
    if suggestion:
        print(f"💡 Suggestion: {suggestion}")


# ============================================================================
# COMMAND GENERATION UTILITIES
# ============================================================================

def generate_demo_commands(step: int, mode: str = "forward_varying") -> List[float]:
    """Generate demonstration commands for different demo modes."""
    if mode == "forward_varying":
        # Vary forward speed sinusoidally
        speed = 0.5 + 1.5 * (np.sin(2 * np.pi * step / 200) + 1) / 2
        return [float(speed), 0.0, 0.0, 0.3, 0.0]
    
    elif mode == "circular":
        # Walk in a circle
        forward_speed = 1.0
        turn_speed = 0.3 * np.sin(2 * np.pi * step / 300)
        return [forward_speed, 0.0, float(turn_speed), 0.3, 0.0]
    
    elif mode == "figure_eight":
        # Walk in a figure-eight pattern
        forward_speed = 0.8
        turn_speed = 0.5 * np.sin(2 * np.pi * step / 400)
        return [forward_speed, 0.0, float(turn_speed), 0.3, 0.0]
    
    else:
        # Default: simple forward walking
        return [1.0, 0.0, 0.0, 0.3, 0.0]


def create_command_tensor(command: List[float], device: str) -> torch.Tensor:
    """Create a properly formatted command tensor for the environment."""
    return torch.tensor([command]).to(device)