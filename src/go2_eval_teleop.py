import argparse
import os
import pickle
import cv2  # For video rendering
import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import genesis as gs
import config
from utils import create_video_with_overlay, interpolate_commands, get_checkpoints, process_checkpoint

def process_checkpoint(exp_name, ckpt, use_adaptive_curriculum=False):
    """Load a checkpoint, run evaluation, and generate a video with command overlays."""
    print(f"\n{'='*50}")
    print(f"Processing checkpoint: {ckpt}")
    if use_adaptive_curriculum:
        print("Using adaptive curriculum learning")
    print(f"{'='*50}")
    
    log_dir = f"logs/{exp_name}"
    
    # -------------------------------
    # Load experiment configuration
    # -------------------------------
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(os.path.join(log_dir, "cfgs.pkl"), "rb"))
    reward_cfg["reward_scales"] = {}

    # -------------------------------
    # Initialize environment
    # -------------------------------
    env_cfg["termination_if_roll_greater_than"] = 50  # degree
    env_cfg["termination_if_pitch_greater_than"] = 50  # degree
    num_envs = config.num_envs  # number of robots
    env = Go2Env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        add_camera=True,
        use_adaptive_curriculum=use_adaptive_curriculum,
    )
    
    # -------------------------------
    # Initialize runner and load checkpoint
    # -------------------------------
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    iter = 0
    
    # -------------------------------
    # Generate motion commands using interpolation
    # -------------------------------
    motion_commands = interpolate_commands(key_commands, config.steps_per_transition, config.transition_break_steps)
    
    max_iter = len(motion_commands)
    reset_jump_toggle_iter = 0
    images_buffer = []
    commands_buffer = []

    with torch.no_grad():
        while iter < max_iter:
            # -------------------------------
            # Get current command
            # -------------------------------
            lin_x, lin_y, ang_z, base_height, jump_height = motion_commands[iter]
            toggle_jump = True
            
            if iter % 30 == 0:
                print(f"Iter: {iter}, lin_x={lin_x:.2f}, lin_y={lin_y:.2f}")
   
            # -------------------------------
            # Apply command
            # -------------------------------
            actions = policy(obs)
            env.commands = torch.tensor([[lin_x, lin_y, ang_z, base_height, toggle_jump*jump_height]], dtype=torch.float).to("cuda:0").repeat(num_envs, 1)
            obs, _, rews, dones, infos = env.step(actions, is_train=False) # step the simulation

            # -------------------------------
            # Handle jump toggle
            # -------------------------------
            
            if toggle_jump and reset_jump_toggle_iter == 0:
                reset_jump_toggle_iter = iter + config.jump_step
            if iter == reset_jump_toggle_iter and toggle_jump:
                toggle_jump = False
                reset_jump_toggle_iter = 0
            
            # -------------------------------
            # Render the camera
            # -------------------------------
            if env.cam_0 is not None:
                rgb, _, _, _ = env.cam_0.render(
                    rgb=True,
                    depth=False,
                    segmentation=False,
                )
                images_buffer.append(rgb)
                commands_buffer.append([lin_x, lin_y, ang_z, base_height, jump_height])
            
            # -------------------------------
            # Check for termination
            # -------------------------------
            if dones.any():
                print("Environment reset, continuing evaluation...")
            
            iter += 1

    # -------------------------------
    # Save the images and commands
    # -------------------------------
    images_buffer = np.array(images_buffer)
    commands_buffer = np.array(commands_buffer)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("videos", exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    data_dir = os.path.join("eval_data", exp_name, f"ckpt_{ckpt}")
    os.makedirs(data_dir, exist_ok=True)
    pickle.dump(images_buffer, open(os.path.join(data_dir, "images_buffer.pkl"), "wb"))
    pickle.dump(commands_buffer, open(os.path.join(data_dir, "commands_buffer.pkl"), "wb"))

    # -------------------------------
    # Render video with checkpoint-specific filename
    # -------------------------------
    base_output_path = getattr(config, "output_video_path", "output")
    output_video_path = os.path.join(output_dir, f"{os.path.splitext(base_output_path)[0]}_{ckpt}.mp4")
    
    create_video_with_overlay(images_buffer, commands_buffer, output_video_path, fps=30)
    print(f"Video saved to {output_video_path}")
    
    # Clean up
    del env
    del runner
    del policy
    torch.cuda.empty_cache()

def main():
    # -------------------------------
    # Parse arguments
    # -------------------------------
    parser = argparse.ArgumentParser(description='Evaluate Go2 policy on multiple checkpoints.')
    parser.add_argument("-e", "--exp_name", type=str, default="my_experiment",
                       help="Name of the experiment to evaluate")
    parser.add_argument("--ckpt", type=int, default=None,
                       help="Specific checkpoint to evaluate (if not provided, evaluates every 5000th checkpoint)")
    parser.add_argument("--interval", type=int, default=5000,
                       help="Interval between checkpoints to evaluate (default: 5000)")
    parser.add_argument("--adaptive_curriculum", action="store_true", 
                       help="Enable adaptive curriculum learning with explicit stages")
    args = parser.parse_args()
    
    log_dir = f"logs/{args.exp_name}"
    
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Experiment directory not found: {log_dir}")
    
    # Process either a specific checkpoint or all checkpoints at the specified interval
    if args.ckpt is not None:
        # Process only the specified checkpoint
        if not os.path.exists(os.path.join(log_dir, f"model_{args.ckpt}.pt")):
            raise FileNotFoundError(f"Checkpoint {args.ckpt} not found in {log_dir}")
        checkpoints = [args.ckpt]
    else:
        # Find all checkpoints at the specified interval
        checkpoints = get_checkpoints(log_dir, args.interval)
        if not checkpoints:
            print(f"No checkpoints found in {log_dir} at interval {args.interval}")
            return
        print(f"Found {len(checkpoints)} checkpoints to process: {checkpoints}")
    
    # -------------------------------
    # Initialize Genesis
    # -------------------------------
    gs.init(
        logger_verbose_time=False,
        logging_level="warning",
    )

    # -------------------------------
    # Process each checkpoint
    # -------------------------------
    for ckpt in checkpoints:
        try:
            process_checkpoint(args.exp_name, ckpt, args.adaptive_curriculum)
        except Exception as e:
            print(f"Error processing checkpoint {ckpt}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
