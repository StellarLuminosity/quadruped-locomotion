"""
go2_eval_teleop.py

Evaluate a trained Go2 quadruped policy with a scripted sequence of
keyboard commands (headless-friendly) and optionally record images &
command data to immediately create an annotated video.

Key features
------------
1. Replays a key-sequence (e.g. "wwasdj...") or a default pattern.
2. Feeds those commands into the Go2 environment + trained policy.
3. Records RGB frames and command vectors.
4. Generates an MP4 with joystick / height / angular-velocity overlays.

Example usage
-------------
python src/go2_eval_teleop.py -e my_experiment --ckpt 900 --keys "wwasdj"
"""

# --------------------------- Imports ---------------------------

import argparse
import os
import pickle
import cv2  # For video rendering
import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import math
import genesis as gs
import config

# ---------------------- Command Sequence Helpers ----------------------

def process_keypress_sequence(keypresses, steps_per_key=config.steps_per_key):
    """
    Generate robot commands from a sequence of keypresses
    
    Args:
        keypresses: List of characters representing keyboard keys (e.g., ['w', 'a', 's', 'd', 'j'])
        steps_per_key: Number of simulation steps to apply each key command
        
    Returns:
        List of command vectors [lin_x, lin_y, ang_z, base_height, jump_height]
    """
    lin_x = 0.0
    lin_y = 0.0
    ang_z = 0.0
    base_height = 0.3
    jump_height = 0.7
    toggle_jump = False
    commands = []
    
    # Process each keypress
    for key in keypresses:
        # Apply the same logic as the original on_press function
        if key == 'w':
            lin_x += 0.1
        elif key == 's':
            lin_x -= 0.1
        elif key == 'a':
            lin_y += 0.1
        elif key == 'd':
            lin_y -= 0.1
        elif key == 'q':
            ang_z += 0.1
        elif key == 'e':
            ang_z -= 0.1
        elif key == 'r':
            base_height += 0.1
        elif key == 'f':
            base_height -= 0.1
        elif key == 'j':
            toggle_jump = True
        elif key == 'u':
            jump_height += 0.1
        elif key == 'm':
            jump_height -= 0.1
            
        # Apply clipping to ensure values are in valid ranges
        lin_x = np.clip(lin_x, -1.0, 2.0)
        lin_y = np.clip(lin_y, -0.5, 0.5)
        ang_z = np.clip(ang_z, -0.6, 0.6)
        base_height = np.clip(base_height, 0.1, 0.5)
        jump_height = np.clip(jump_height, 0.5, 1.5)
        
        # Create the command vector with the jump flag
        jump_val = jump_height if toggle_jump else 0.0
        cmd = [lin_x, lin_y, ang_z, base_height, jump_val]
        
        # Add the command multiple times based on steps_per_key
        if key != 'j':
            commands.extend([cmd] * steps_per_key)
        
        # Reset jump toggle after one command
        toggle_jump = False
        
        # Print the current command for debugging
        print(f"Key '{key}' → Command: [{lin_x:.2f}, {lin_y:.2f}, {ang_z:.2f}, {base_height:.2f}, {jump_val:.2f}]")
    
    # Always add a final stop command
    if commands:
        commands.append([0.0, 0.0, 0.0, base_height, 0.0])
        
    return commands

# ------------------- Video rendering code helpers -------------------

def _safe_max(values):
    m = max(abs(v) for v in values)
    return m if m > 1e-6 else 1.0

def _normalize_command_ranges(commands_buffer):
    """Return max absolute values for each command dimension for scaling overlays."""
    max_lin_x = _safe_max(cmd[0] for cmd in commands_buffer)
    max_lin_y = _safe_max(cmd[1] for cmd in commands_buffer)
    max_ang_z = _safe_max(cmd[2] for cmd in commands_buffer)
    max_base_h = _safe_max(cmd[3] for cmd in commands_buffer)
    max_jump_h = _safe_max(cmd[4] for cmd in commands_buffer)
    return max_lin_x, max_lin_y, max_ang_z, max_base_h, max_jump_h

def _draw_joystick(img, lin_x, lin_y, max_lin_x, max_lin_y, radius=100, x_offset=10, y_offset=10):
    # Draw gradient disc
    for i in range(radius):
        r = radius - i
        col = int(55 + 200 * (0.5 + 0.5 * i / radius))
        cv2.circle(img, (x_offset + radius, y_offset + radius), r, (col, col, col), -1)
    
    # Ensure max values are not zero to avoid division by zero
    safe_max_x = max(max_lin_x, 1e-6)
    safe_max_y = max(max_lin_y, 1e-6)
    
    # Normalize values to ensure they stay within -1 to 1 range
    norm_lin_y = np.clip(lin_y / safe_max_y, -1.0, 1.0)
    norm_lin_x = np.clip(lin_x / safe_max_x, -1.0, 1.0)
    
    # Thumb position - ensure it stays within the circle
    jx = int(x_offset + radius + norm_lin_y * radius)
    jy = int(y_offset + radius - norm_lin_x * radius)
    
    # Add shadow effect
    cv2.circle(img, (jx + 2, jy + 2), int(radius * 0.12), (0, 0, 0), -1)
    return img

def _draw_height_bar(img, base_h, max_base_h, target_h=None, x_offset=220, y_offset=10, bar_h=200, bar_w=20):
    """Draw vertical bar showing current and target base height."""
    # Background
    cv2.rectangle(img, (x_offset, y_offset), (x_offset + bar_w, y_offset + bar_h), (200, 200, 200), -1)
    # Current height indicator (green)
    cur_pos = int(y_offset + bar_h - (base_h / max_base_h) * bar_h)
    cv2.rectangle(img, (x_offset, cur_pos), (x_offset + bar_w, y_offset + bar_h), (0, 255, 0), -1)
    # Target height line (red)
    if target_h is not None:
        target_pos = int(y_offset + bar_h - (target_h / max_base_h) * bar_h)
        cv2.line(img, (x_offset, target_pos), (x_offset + bar_w, target_pos), (0, 0, 255), 2)
    return img

def _draw_ang_vel_bar(img, ang_z, max_ang_z, x_offset=10, y_offset=220, bar_w=200, bar_h=20):
    cv2.rectangle(img, (x_offset, y_offset), (x_offset + bar_w, y_offset + bar_h), (200, 200, 200), -1)
    safe_max = max(max_ang_z, 1e-6)
    norm = (ang_z / safe_max + 1) / 2  # 0..1
    cur_pos = int(x_offset + norm * bar_w)
    cv2.rectangle(img, (x_offset, y_offset), (cur_pos, y_offset + bar_h), (0, 255, 0), -1)
    return img

def create_video_with_overlay(images_buffer, commands_buffer, output_path, fps=30):
    h, w, _ = images_buffer[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    max_lin_x, max_lin_y, max_ang_z, max_base_h, _ = _normalize_command_ranges(commands_buffer)

    radius = 100
    for img, cmd in zip(images_buffer, commands_buffer):
        lin_x, lin_y, ang_z, base_h, _ = cmd
        x_offset = w // 2 - 100
        y_offset = h - 250
        canvas = _draw_joystick(img.copy(), lin_x, lin_y, max_lin_x, max_lin_y, radius, x_offset, y_offset)
        canvas = _draw_height_bar(canvas, base_h, max_base_h, target_h=max_base_h*0.5, x_offset=x_offset + radius*2 + 10, y_offset=y_offset)
        canvas = _draw_ang_vel_bar(canvas, ang_z, max_ang_z, x_offset, y_offset + radius*2 + 20)
        out.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    out.release()

# ---------------------------- Main Routine ----------------------------

def main():
    # -------------------------------
    # Parse arguments
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="my_experiment")
    parser.add_argument("--ckpt", type=int, default=900)
    parser.add_argument("--keys", type=str, help="Sequence of keyboard commands, e.g. 'wwasdjww'")
    args = parser.parse_args()

    # -------------------------------
    # Initialize Genesis
    # -------------------------------

    gs.init(
        logger_verbose_time = False,
        logging_level="warning",
    )

    # -------------------------------
    # Load experiment configuration
    # -------------------------------

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # -------------------------------
    # Initialize environment
    # -------------------------------

    env_cfg["termination_if_roll_greater_than"] =  50  # degree
    env_cfg["termination_if_pitch_greater_than"] = 50  # degree
    num_envs = config.num_envs # number of robots
    env = Go2Env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        add_camera=True,
    )
    
    # -------------------------------
    # Initialize runner
    # -------------------------------
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    iter = 0
    
    # -------------------------------
    # Generate motion commands
    # -------------------------------
    keypress_sequence = list(args.keys) if args.keys else config.key_commands
    motion_commands = process_keypress_sequence(keypress_sequence)
    
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
                iter = 0
            
            iter += 1

    # -------------------------------
    # Save the images and commands
    # -------------------------------
    images_buffer = np.array(images_buffer)
    commands_buffer = np.array(commands_buffer)
    pickle.dump(images_buffer, open("images_buffer.pkl", "wb"))
    pickle.dump(commands_buffer, open("commands_buffer.pkl", "wb"))

    # -------------------------------
    # Render video
    # -------------------------------
    output_video_path = getattr(config, "output_video_path", "output.mp4")
    create_video_with_overlay(images_buffer, commands_buffer, output_video_path, fps=30)
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    main()
