# Enhanced Evaluation Script for Go2 Quadruped Robot
"""
Professional evaluation script with multiple evaluation modes and better user experience.
This replaces the basic go2_eval.py with a more robust implementation.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from go2_env import Go2Env
from configs import get_eval_configs


def validate_experiment(exp_name: str, ckpt: int) -> Path:
    """Validate that experiment and checkpoint exist."""
    log_dir = Path(f"logs/{exp_name}")
    
    if not log_dir.exists():
        raise FileNotFoundError(f"Experiment '{exp_name}' not found. Available experiments:")
    
    config_path = log_dir / "configs.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    model_path = log_dir / f"model_{ckpt}.pt"
    if not model_path.exists():
        # Find available checkpoints
        available_ckpts = []
        for f in log_dir.glob("model_*.pt"):
            try:
                ckpt_num = int(f.stem.split('_')[1])
                available_ckpts.append(ckpt_num)
            except:
                continue
        
        if available_ckpts:
            available_ckpts.sort()
            raise FileNotFoundError(
                f"Checkpoint {ckpt} not found. Available checkpoints: {available_ckpts}"
            )
        else:
            raise FileNotFoundError(f"No model checkpoints found in {log_dir}")
    
    return log_dir


def load_model(log_dir: Path, ckpt: int, device: str):
    """Load trained model and create inference policy."""
    print("Loading experiment configuration")
    
    # Load configs (try new format first, fallback to old)
    config_path = log_dir / "configs.pkl"
    try:
        configs = pickle.load(open(config_path, "rb"))
        if len(configs) == 5:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = configs
        else:
            # Old format fallback
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = configs
    except:
        print("Failed to load configs. Using default evaluation configs.")
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_eval_configs()
        train_cfg = {}  # Will be reconstructed
    
    print("Creating evaluation environment")
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device=device,
    )
    
    print("Loading trained model")
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=device)
    model_path = log_dir / f"model_{ckpt}.pt"
    runner.load(str(model_path))
    policy = runner.get_inference_policy(device=device)
    
    print("Model loaded successfully")
    return env, policy


def run_demo_mode(env, policy, device: str, duration: int = 300):
    """Run basic demonstration with forward walking."""
    print("Running demo mode for {duration} steps")
    print("Robot will walk forward at varying speeds")
    
    obs, _ = env.reset()
    
    with torch.no_grad():
        for step in range(duration):
            # Vary forward speed sinusoidally
            speed = 0.5 + 1.5 * (np.sin(2 * np.pi * step / 200) + 1) / 2
            env.commands = torch.tensor([[speed, 0.0, 0.0, 0.3, 0.0]]).to(device)
            
            actions = policy(obs)
            obs, _, rewards, dones, infos = env.step(actions, is_train=False)
            
            if step % 50 == 0:
                print(f"Step {step:3d}: Speed = {speed:.2f} m/s")
            
            if dones.any():
                print("Environment reset due to termination")
                obs, _ = env.reset()


def run_interactive_mode(env, policy, device: str):
    """Run interactive mode with keyboard control."""
    print("Interactive mode started")
    print("Controls:")
    print("  W - Forward    S - Backward")
    print("  A - Left       D - Right") 
    print("  Q - Turn Left  E - Turn Right")
    print("  Space - Stop   J - Jump")
    print("  X - Exit")
    
    obs, _ = env.reset()
    
    # Command mappings
    commands = {
        'w': [1.0, 0.0, 0.0, 0.3, 0.0],   # forward
        's': [-1.0, 0.0, 0.0, 0.3, 0.0],  # backward
        'a': [0.0, 1.0, 0.0, 0.3, 0.0],   # left
        'd': [0.0, -1.0, 0.0, 0.3, 0.0],  # right
        'q': [0.0, 0.0, 1.0, 0.3, 0.0],   # turn left
        'e': [0.0, 0.0, -1.0, 0.3, 0.0],  # turn right
        ' ': [0.0, 0.0, 0.0, 0.3, 0.0],   # stop
        'j': [0.0, 0.0, 0.0, 0.3, 1.0],   # jump
    }
    
    current_cmd = [0.0, 0.0, 0.0, 0.3, 0.0]  # start stopped
    
    try:
        import keyboard
        keyboard_available = True
    except ImportError:
        print("keyboard package not available. Install with: pip install keyboard")
        keyboard_available = False
        
    if not keyboard_available:
        print("Running automatic demo instead")
        run_demo_mode(env, policy, device)
        return
    
    with torch.no_grad():
        step = 0
        while True:
            # Check for key presses
            for key, cmd in commands.items():
                if keyboard.is_pressed(key):
                    if key == 'x':
                        print("Exiting interactive mode")
                        return
                    current_cmd = cmd.copy()
                    break
            
            env.commands = torch.tensor([current_cmd]).to(device)
            actions = policy(obs)
            obs, _, rewards, dones, infos = env.step(actions, is_train=False)
            
            step += 1
            if step % 100 == 0:
                cmd_str = f"[{current_cmd[0]:.1f}, {current_cmd[1]:.1f}, {current_cmd[2]:.1f}]"
                print(f"Step {step}: Command = {cmd_str}")
            
            if dones.any():
                print("Environment reset")
                obs, _ = env.reset()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Go2 quadruped robot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-e", "--exp_name", type=str, required=True,
                       help="Experiment name to evaluate")
    parser.add_argument("--ckpt", type=int, default=200,
                       help="Model checkpoint to load")
    parser.add_argument("--device", type=str, default="cuda:0",
                       choices=["cuda:0", "cpu"], help="Evaluation device")
    parser.add_argument("--mode", type=str, default="demo",
                       choices=["demo", "interactive"], 
                       help="Evaluation mode")
    parser.add_argument("--duration", type=int, default=300,
                       help="Duration for demo mode (steps)")
    
    args = parser.parse_args()
    
    try:
        print("🔍 Validating experiment and checkpoint...")
        log_dir = validate_experiment(args.exp_name, args.ckpt)
        
        print("🔧 Initializing Genesis physics engine...")
        backend = gs.constants.backend.gpu if args.device == "cuda:0" else gs.constants.backend.cpu
        gs.init(backend=backend)
        
        # Load model and environment
        env, policy = load_model(log_dir, args.ckpt, args.device)
        
        # Run evaluation
        if args.mode == "demo":
            run_demo_mode(env, policy, args.device, args.duration)
        elif args.mode == "interactive":
            run_interactive_mode(env, policy, args.device)
        
        print("Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
