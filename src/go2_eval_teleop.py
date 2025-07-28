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

# Global variables to store predefined command velocities
key_commands = [
    [1.0, 0.0, 0.0, 0.3, 0.7],    # forward
    [0.0, 1.0, 0.0, 0.3, 0.7],    # left
    [0.0, -1.0, 0.0, 0.3, 0.7],   # right
    [-1.0, 0.0, 0.0, 0.3, 0.7],   # backward
    [0.0, 0.0, 0.0, 0.3, 0.7],    # stop
]

def interpolate_commands(commands, steps_per_transition):
    """
    Interpolate between command vectors to create smooth transitions
    
    Args:
        commands: List of command vectors [lin_x, lin_y, ang_z, base_height, jump_height]
        steps_per_transition: Number of steps to interpolate between each command
        
    Returns:
        List of interpolated command vectors
    """
    result = []
    for i in range(len(commands) - 1):
        start = np.array(commands[i])
        end = np.array(commands[i + 1])
        for alpha in np.linspace(0, 1, steps_per_transition):
            interp = (1 - alpha) * start + alpha * end
            result.append(interp.tolist())
    return result

# ------------------- Video rendering code helpers -------------------

def _safe_max(values):
    m = max(abs(v) for v in values)
    return m if m > 1e-6 else 1.0

def normalize_commands(commands_buffer):
    """Return max absolute values for each command dimension for scaling overlays."""
    max_lin_x = _safe_max(cmd[0] for cmd in commands_buffer)
    max_lin_y = _safe_max(cmd[1] for cmd in commands_buffer)
    max_ang_z = _safe_max(cmd[2] for cmd in commands_buffer)
    max_base_h = _safe_max(cmd[3] for cmd in commands_buffer)
    max_jump_h = _safe_max(cmd[4] for cmd in commands_buffer)
    return max_lin_x, max_lin_y, max_ang_z, max_base_h, max_jump_h

def draw_joystick(image, lin_x, lin_y, ang_z, base_height, jump_height, max_lin_x, max_lin_y, radius=100, x_offset=10, y_offset=10):
    # Draw the joystick base with gradient directly on the image
    for i in range(radius):
        r = radius - i
        # color = (255, int(255 * (0.5 + 0.5 * i / radius)), int(255 * (0.5 + 0.5 * i / radius)))
        color = (int(55+200 * (0.5 + 0.5 * i / radius)), int(55+200 * (0.5 + 0.5 * i / radius)), int(55+200 * (0.5 + 0.5 * i / radius)))
        cv2.circle(image, (x_offset + radius, y_offset + radius), r, color, -1)

    # Draw the joystick position with shadow directly on the image
    joystick_x = int(x_offset + radius + (lin_y / max_lin_y) * radius)
    joystick_y = int(y_offset + radius - (lin_x / max_lin_x) * radius)
    cv2.circle(image, (joystick_x + 2, joystick_y + 2), int(radius * 0.12), (0, 0, 0), -1)  # Shadow
    # cv2.circle(image, (joystick_x, joystick_y), int(radius * 0.1), (0, 0, 255), -1)

    return image


def draw_target_height_bar(image, base_height, max_base_height, target_height=1.0, x_offset=220, y_offset=10):
    base_height = max(0, base_height)  # Ensure base_height is non-negative
    # Create a bar to represent the target height
    bar_width = 20
    bar_height = 200
    bar_x = x_offset  # Place the bar to the right of the joystick
    bar_y = y_offset  # Align with the top of the joystick

    # Draw the background of the bar
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)

    # Draw the current height indicator
    current_height_pos = int(bar_y + bar_height - (base_height / max_base_height) * bar_height)
    cv2.rectangle(image, (bar_x, current_height_pos), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)

    # Draw the target height line
    target_height_pos = int(bar_y + bar_height - (target_height / target_height) * bar_height)
    cv2.line(image, (bar_x, target_height_pos), (bar_x + bar_width, target_height_pos), (0, 0, 255), 2)

    return image

def draw_angular_velocity_bar(image, ang_z, max_ang_z, x_offset=10, y_offset=220):
    # Create a bar to represent the angular velocity
    bar_width = 200
    bar_height = 20
    bar_x = x_offset  # Use the provided x_offset
    bar_y = y_offset  # Use the provided y_offset

    # Draw the background of the bar
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)

    # Draw the current angular velocity indicator
    current_ang_pos = int(bar_x + (ang_z / max_ang_z + 1) / 2 * bar_width)  # Normalize ang_z to [0, 1]
    cv2.rectangle(image, (bar_x, bar_y), (current_ang_pos, bar_y + bar_height), (0, 255, 0), -1)

    return image

def create_video_with_overlay(images_buffer, commands_buffer, output_path, fps=30):
    # Get the dimensions of the images
    height, width, _ = images_buffer[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    max_lin_x, max_lin_y, max_ang_z, max_base_height, max_jump_height = normalize_commands(commands_buffer)

    for i in range(len(images_buffer)):
        image = images_buffer[i]
        # Invert the image channels (RGB to BGR) for OpenCV
        
        lin_x, lin_y, ang_z, base_height, jump_height = commands_buffer[i]

        
        # Overlay the joystick on the image (top-left corner)
        x_offset = images_buffer[0].shape[1] // 2 - 100
        y_offset = images_buffer[0].shape[0]  - 250
        radius = 100
        
        # Draw the joystick overlay
        image = draw_joystick(image, lin_x, lin_y, ang_z, base_height, jump_height, max_lin_x, max_lin_y, radius=radius, x_offset=x_offset, y_offset=y_offset)


        # Draw the target height bar
        image = draw_target_height_bar(image, base_height, max_base_height, x_offset=x_offset + radius*2 + 10, y_offset=y_offset)

        # Draw the angular velocity bar with adjusted position
        image = draw_angular_velocity_bar(image, ang_z, max_ang_z, x_offset=x_offset, y_offset=y_offset + radius*2 + 20)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        out.write(image)

    # Release the VideoWriter
    out.release()

# ---------------------------- Main Routine ----------------------------

def main():
    # -------------------------------
    # Parse arguments
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="my_experiment")
    parser.add_argument("--ckpt", type=int, default=900)
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
    # Generate motion commands using interpolation
    # -------------------------------
    motion_commands = interpolate_commands(key_commands, config.steps_per_transition)
    
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
