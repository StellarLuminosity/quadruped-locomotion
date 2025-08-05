# Enhanced Training Script for Go2 Quadruped Robot
"""
Professional training script with proper error handling, logging, and user experience.
This replaces the basic go2_train.py with a more robust implementation.
"""

import os
import sys
import argparse
import pickle
import shutil
import time
from pathlib import Path
from datetime import datetime

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from go2_env import Go2Env
from configs import get_all_configs


def setup_logging(log_dir: Path) -> None:
    """Setup logging directory and clean previous runs."""
    if log_dir.exists():
        response = input(f"Log directory '{log_dir}' exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            sys.exit(0)
        shutil.rmtree(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created log directory: {log_dir}")


def validate_args(args) -> None:
    """Validate command line arguments."""
    if args.num_envs <= 0:
        raise ValueError("Number of environments must be positive")
    
    if args.max_iterations <= 0:
        raise ValueError("Max iterations must be positive")
    
    if args.device not in ["cpu", "cuda:0"]:
        raise ValueError("Device must be 'cpu' or 'cuda:0'")
    
    if args.device == "cuda:0" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"


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


def save_training_info(log_dir: Path, args, configs) -> None:
    """Save training configuration and metadata."""
    # Save configs
    config_path = log_dir / "configs.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(configs, f)
    
    # Save training metadata
    metadata = {
        "experiment_name": args.exp_name,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "device": args.device,
        "adaptive_curriculum": args.adaptive_curriculum,
        "start_time": datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
    }
    
    metadata_path = log_dir / "training_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Saved training configuration to {config_path}")


def create_environment(args, configs) -> Go2Env:
    """Create and configure the training environment."""
    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = configs
    
    print("🔧 Creating training environment...")
    
    try:
        env = Go2Env(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            device=args.device,
            use_adaptive_curriculum=args.adaptive_curriculum,
        )
        print("✅ Environment created successfully")
        return env
        
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        raise


def main():
    """Main training function with comprehensive error handling."""
    parser = argparse.ArgumentParser(
        description="Train Go2 quadruped robot using PPO reinforcement learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training arguments
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking",
                       help="Experiment name for logging")
    parser.add_argument("-B", "--num_envs", type=int, default=4096,
                       help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=10000,
                       help="Maximum training iterations")
    parser.add_argument("--device", type=str, default="cuda:0",
                       choices=["cpu", "cuda:0"], help="Training device")
    parser.add_argument("--adaptive_curriculum", action="store_true",
                       help="Enable adaptive curriculum learning")
    
    # Parse and validate arguments
    args = parser.parse_args()
    
    try:
        validate_args(args)
        print_training_header(args)
        
        # Initialize Genesis physics engine
        print("🔧 Initializing Genesis physics engine...")
        backend = gs.constants.backend.gpu if args.device == "cuda:0" else gs.constants.backend.cpu
        gs.init(logging_level="warning", backend=backend)
        print("✅ Genesis initialized")
        
        # Setup logging
        log_dir = Path(f"logs/{args.exp_name}")
        setup_logging(log_dir)
        
        # Get configurations
        print("📋 Loading configurations...")
        configs = get_all_configs(args.exp_name, args.max_iterations)
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = configs
        print("✅ Configurations loaded")
        
        # Save training info
        save_training_info(log_dir, args, configs)
        
        # Create environment and runner
        env = create_environment(args, configs)
        
        print("🚀 Initializing PPO trainer...")
        runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=args.device)
        print("✅ Trainer initialized")
        
        # Start training
        print("\n🏃 Starting training...")
        print("💡 Tip: Monitor progress in TensorBoard with: tensorboard --logdir logs")
        
        start_time = time.time()
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        end_time = time.time()
        
        # Training completed
        training_time = end_time - start_time
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️  Total training time: {training_time/3600:.2f} hours")
        print(f"📁 Results saved to: {log_dir}")
        print(f"🔍 Evaluate with: python eval.py -e {args.exp_name}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
