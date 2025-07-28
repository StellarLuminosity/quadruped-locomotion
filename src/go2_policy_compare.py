"""
go2_policy_compare.py

Compare performance between two trained policies (baseline vs curriculum learning)
for the Go2 quadruped robot.

Example usage:
python src/go2_policy_compare.py --baseline my_experiment --curriculum test_Adaptive --ckpt 20000
"""

import argparse
import os
import pickle
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import genesis as gs


def evaluate_policy(exp_name, ckpt, device="cuda:0", num_steps=1000, verbose=False):
    """
    Evaluate a single policy and return performance metrics
    """
    print(f"\n{'='*50}")
    print(f"Evaluating {exp_name} at checkpoint {ckpt}")
    print(f"{'='*50}")
    
    # Load experiment configuration
    log_dir = f"logs/{exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # Configure environment
    env_cfg["termination_if_roll_greater_than"] = 50  # degree
    env_cfg["termination_if_pitch_greater_than"] = 50  # degree

    # Create environment
    env = Go2Env(
        num_envs=4,  # Using multiple environments for better statistics
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,  # Headless for faster evaluation
    )

    # Load policy
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    # Reset environment
    obs, _ = env.reset()
    
    # Metrics to track
    metrics = {
        "rewards": [],
        "tracking_lin_vel_error": [],
        "tracking_ang_vel_error": [],
        "lin_vel_x": [],
        "ang_vel_z": [],
        "base_height": [],
        "falls": 0,
        "distance_traveled": 0,
        "energy_used": 0,
    }
    
    # Test commands - sequence of different movement patterns
    commands = [
        # [lin_x, lin_y, ang_z, base_height, jump]
        [1.0, 0.0, 0.0, 0.3, 0.0],  # Forward walking
        [0.0, 0.5, 0.0, 0.3, 0.0],  # Lateral walking
        [0.0, 0.0, 0.5, 0.3, 0.0],  # Turning
        [1.0, 0.5, 0.0, 0.3, 0.0],  # Diagonal walking
        [1.5, 0.0, 0.0, 0.3, 0.0],  # Fast forward
        [1.0, 0.0, 0.0, 0.3, 1.0],  # Jump forward
    ]
    
    # Run evaluation
    steps_per_command = num_steps // len(commands)
    total_steps = 0
    command_idx = 0
    positions = []
    
    with torch.no_grad():
        while total_steps < num_steps:
            # Set command for current phase
            current_command = commands[command_idx]
            if verbose and total_steps % 100 == 0:
                print(f"Step {total_steps}, Command: {current_command}")
                
            env.commands = torch.tensor([current_command], dtype=torch.float).to(device).repeat(env.num_envs, 1)
            
            # Step environment
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions, is_train=False)
            
            # Record metrics
            metrics["rewards"].extend(rews.cpu().numpy())
            
            # Extract tracking errors and other metrics from infos
            for i in range(env.num_envs):
                if "tracking_lin_vel_error" in infos[i]:
                    metrics["tracking_lin_vel_error"].append(infos[i]["tracking_lin_vel_error"])
                if "tracking_ang_vel_error" in infos[i]:
                    metrics["tracking_ang_vel_error"].append(infos[i]["tracking_ang_vel_error"])
                if "lin_vel_x" in infos[i]:
                    metrics["lin_vel_x"].append(infos[i]["lin_vel_x"])
                if "ang_vel_z" in infos[i]:
                    metrics["ang_vel_z"].append(infos[i]["ang_vel_z"])
                if "base_height" in infos[i]:
                    metrics["base_height"].append(infos[i]["base_height"])
                
                # Track robot position for distance calculation
                if "position" in infos[i]:
                    positions.append(infos[i]["position"])
                
                # Track energy use if available
                if "energy" in infos[i]:
                    metrics["energy_used"] += infos[i]["energy"]
                    
            # Count falls
            metrics["falls"] += sum(dones.cpu().numpy())
            
            # Update step counters
            total_steps += 1
            
            # Change command if needed
            if total_steps % steps_per_command == 0 and command_idx < len(commands) - 1:
                command_idx += 1
    
    # Calculate distance traveled if position data is available
    if positions:
        start_pos = np.array(positions[0])
        end_pos = np.array(positions[-1])
        metrics["distance_traveled"] = np.linalg.norm(end_pos - start_pos)
    
    # Clean up
    env.close()
    del env
    del runner
    del policy
    torch.cuda.empty_cache()
    
    # Calculate aggregate metrics
    metrics["mean_reward"] = np.mean(metrics["rewards"])
    metrics["std_reward"] = np.std(metrics["rewards"])
    metrics["mean_tracking_lin_vel_error"] = np.mean(metrics["tracking_lin_vel_error"]) if metrics["tracking_lin_vel_error"] else 0
    metrics["mean_tracking_ang_vel_error"] = np.mean(metrics["tracking_ang_vel_error"]) if metrics["tracking_ang_vel_error"] else 0
    metrics["mean_base_height"] = np.mean(metrics["base_height"]) if metrics["base_height"] else 0
    
    # Print summary
    print(f"\nEvaluation Summary for {exp_name} (Checkpoint {ckpt}):")
    print(f"Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
    print(f"Linear Velocity Tracking Error: {metrics['mean_tracking_lin_vel_error']:.4f}")
    print(f"Angular Velocity Tracking Error: {metrics['mean_tracking_ang_vel_error']:.4f}")
    print(f"Falls: {metrics['falls']}")
    print(f"Distance Traveled: {metrics['distance_traveled']:.2f} m")
    print(f"Energy Used: {metrics['energy_used']:.2f}")
    
    return metrics


def compare_policies(baseline_exp, curriculum_exp, ckpt, device="cuda:0", num_steps=1000, verbose=False):
    """
    Compare two policies and generate comparison report
    """
    # Initialize Genesis
    gs.init(
        backend=gs.constants.backend.gpu if device == "cuda:0" else gs.constants.backend.cpu,
        logger_verbose_time=False,
        logging_level="warning",
    )
    
    # Evaluate both policies
    baseline_metrics = evaluate_policy(baseline_exp, ckpt, device, num_steps, verbose)
    curriculum_metrics = evaluate_policy(curriculum_exp, ckpt, device, num_steps, verbose)
    
    # Calculate improvements
    improvements = {}
    for key in baseline_metrics:
        if isinstance(baseline_metrics[key], (int, float)) and key in curriculum_metrics:
            if baseline_metrics[key] != 0:
                # For error metrics, lower is better
                if "error" in key or key == "falls":
                    improvements[key] = ((baseline_metrics[key] - curriculum_metrics[key]) / baseline_metrics[key]) * 100
                else:
                    # For other metrics, higher is better
                    improvements[key] = ((curriculum_metrics[key] - baseline_metrics[key]) / baseline_metrics[key]) * 100
    
    # Print comparison
    print("\n" + "="*80)
    print("POLICY COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<30} {'Baseline':<15} {'Curriculum':<15} {'Improvement':<15}")
    print("-"*80)
    
    for key in ["mean_reward", "mean_tracking_lin_vel_error", "mean_tracking_ang_vel_error", 
                "falls", "distance_traveled", "energy_used"]:
        if key in baseline_metrics and key in curriculum_metrics:
            baseline_val = baseline_metrics[key]
            curriculum_val = curriculum_metrics[key]
            
            if key in improvements:
                improvement = improvements[key]
                print(f"{key:<30} {baseline_val:<15.4f} {curriculum_val:<15.4f} {improvement:+.2f}%")
            else:
                print(f"{key:<30} {baseline_val:<15.4f} {curriculum_val:<15.4f} {'N/A':<15}")
    
    # Save results to file
    results = {
        "baseline": baseline_metrics,
        "curriculum": curriculum_metrics,
        "improvements": improvements
    }
    
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(output_dir, f"comparison_{baseline_exp}_vs_{curriculum_exp}_{ckpt}_{timestamp}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_file}")
    
    # Generate simple plots
    plot_comparison(baseline_metrics, curriculum_metrics, baseline_exp, curriculum_exp, ckpt, output_dir, timestamp)
    
    return results


def plot_comparison(baseline_metrics, curriculum_metrics, baseline_name, curriculum_name, ckpt, output_dir, timestamp):
    """Generate comparison plots"""
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Bar chart of key metrics
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ["mean_reward", "mean_tracking_lin_vel_error", "mean_tracking_ang_vel_error", "falls"]
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in baseline_metrics and metric in curriculum_metrics:
            plt.subplot(2, 2, i+1)
            values = [baseline_metrics[metric], curriculum_metrics[metric]]
            bars = plt.bar(["Baseline", "Curriculum"], values, color=['blue', 'green'])
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.title(metric.replace('_', ' ').title())
            
            # For error metrics, add note that lower is better
            if "error" in metric or metric == "falls":
                plt.figtext(0.5, 0.01, "Lower is better", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"metrics_comparison_{ckpt}_{timestamp}.png"))
    
    print(f"Plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare performance between baseline and curriculum learning policies")
    parser.add_argument("--baseline", type=str, default="my_experiment", help="Baseline experiment name")
    parser.add_argument("--curriculum", type=str, default="test_Adaptive", help="Curriculum learning experiment name")
    parser.add_argument("--ckpt", type=int, default=20000, help="Checkpoint to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"], help="Device to run evaluation on")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    
    compare_policies(
        args.baseline,
        args.curriculum,
        args.ckpt,
        args.device,
        args.steps,
        args.verbose
    )


if __name__ == "__main__":
    main()
