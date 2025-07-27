import argparse
import os
import pickle
import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import math
import genesis as gs

# Command vector format: [ lin_x, lin_y, ang_z, base_height, jump_height ]

# Default hard-coded command sequence if no key sequence is provided
default_key_commands = [
    [1.0, 0.0, 0.0, 0.3, 0.7],    # forward
    [0.0, 1.0, 0.0, 0.3, 0.7],    # left
    [0.0, -1.0, 0.0, 0.3, 0.7],   # right
    [-1.0, 0.0, 0.0, 0.3, 0.7],   # backward
    [0.0, 0.0, 0.0, 0.3, 0.7],    # stop
]
steps_per_transition = 60
steps_per_key = 20  # How many steps to apply each keypress

def interpolate_commands(commands, steps_per_transition):
    """Smoothly interpolate between command waypoints"""
    result = []
    for i in range(len(commands) - 1):
        start = np.array(commands[i])
        end = np.array(commands[i + 1])
        for alpha in np.linspace(0, 1, steps_per_transition):
            interp = (1 - alpha) * start + alpha * end
            result.append(interp.tolist())
    return result


def process_keypress_sequence(keypresses, steps_per_key=20):
    """
    Generate robot commands from a sequence of keypresses
    
    Args:
        keypresses: List of characters representing keyboard keys (e.g., ['w', 'a', 's', 'd', 'j'])
        steps_per_key: Number of simulation steps to apply each key command
        
    Returns:
        List of command vectors [lin_x, lin_y, ang_z, base_height, jump_height]
    """
    # Start with default values
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
        commands.extend([cmd] * steps_per_key)
        
        # Reset jump toggle after one command
        toggle_jump = False
        
        # Print the current command for debugging
        print(f"Key '{key}' → Command: [{lin_x:.2f}, {lin_y:.2f}, {ang_z:.2f}, {base_height:.2f}, {jump_val:.2f}]")
    
    # Always add a final stop command
    if commands:
        commands.append([0.0, 0.0, 0.0, base_height, 0.0])
        
    return commands

# lin_x = 0.0
# lin_y = 0.0
# ang_z = 0.0
# base_height = 0.3
# jump_height = 0.7
# toggle_jump = True # False
# stop = False

# def on_press(key):
#     global lin_x, lin_y, ang_z, base_height, toggle_jump, jump_height, stop
#     try:
#         if key.char == 'w':
#             lin_x += 0.1
#         elif key.char == 's':
#             lin_x -= 0.1
#         elif key.char == 'a':
#             lin_y += 0.1
#         elif key.char == 'd':
#             lin_y -= 0.1
#         elif key.char == 'q':
#             ang_z += 0.1
#         elif key.char == 'e':
#             ang_z -= 0.1
#         elif key.char == 'r':
#             base_height += 0.1
#         elif key.char == 'f':
#             base_height -= 0.1
#         elif key.char == 'j':
#             toggle_jump = True
#         elif key.char == 'u':
#             jump_height += 0.1
#         elif key.char == 'm':
#             jump_height -= 0.1
#         elif key.char == '8':
#             stop = True
#             
#         
#         lin_x = np.clip(lin_x, -1.0, 2.0)
#         lin_y = np.clip(lin_y, -0.5, 0.5)
#         ang_z = np.clip(ang_z, -0.6, 0.6)
#         base_height = np.clip(base_height, 0.1, 0.5)
#         jump_height = np.clip(jump_height, 0.5, 1.5)
#         
#             
#         # Clear the console
#         os.system('clear')
#         
#         print(f"lin_x: {lin_x:.2f}, lin_y: {lin_y:.2f}, ang_z: {ang_z:.2f}, base_height: {base_height:.2f}, jump: {toggle_jump*jump_height:.2f}")
#     except AttributeError:
#         pass

# def on_release(key):
#     if key == keyboard.Key.esc:
#         # Stop listener
#         return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="my_experiment")
    parser.add_argument("--ckpt", type=int, default=900)
    parser.add_argument("--save-data", action="store_true", help="Save rendered images and commands")
    parser.add_argument("--keys", type=str, help="Sequence of keyboard commands, e.g. 'wwasdjww'")
    parser.add_argument("--keys-file", type=str, help="Path to a file containing keyboard commands, one per line")
    parser.add_argument("--keys-steps", type=int, default=20, help="Steps per key command")
    parser.add_argument("--use-default", action="store_true", help="Use default hardcoded command sequence")
    args = parser.parse_args()

    gs.init(
        logger_verbose_time = False,
        logging_level="warning",
    )

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"genesis/logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env_cfg["termination_if_roll_greater_than"] =  50  # degree
    env_cfg["termination_if_pitch_greater_than"] = 50  # degree
    num_envs = 50 # number of robots
    env = Go2Env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        add_camera=True,
    )
    
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    iter = 0
    
    # Determine which command sequence to use
    if args.keys:
        # Convert string of keys to list of characters
        keypress_sequence = list(args.keys)
        print(f"\nUsing command sequence from --keys argument: {args.keys}")
        motion_commands = process_keypress_sequence(keypress_sequence, args.keys_steps)
    elif args.keys_file:
        # Read keys from file, one character per line
        with open(args.keys_file, 'r') as f:
            keypress_sequence = [line.strip() for line in f if line.strip()]
        print(f"\nLoaded {len(keypress_sequence)} commands from file: {args.keys_file}")
        motion_commands = process_keypress_sequence(keypress_sequence, args.keys_steps)
    elif args.use_default:
        # Use the default hardcoded command sequence
        print("\nUsing default hardcoded command sequence")
        motion_commands = interpolate_commands(default_key_commands, steps_per_transition)
    else:
        # Provide a default example sequence
        example_sequence = ['w', 'w', 'w', 'a', 'a', 'd', 'd', 'j', 's', 's']
        print("\nNo command sequence provided. Using example sequence:")
        print(" ".join(example_sequence))
        motion_commands = process_keypress_sequence(example_sequence, args.keys_steps)
    
    # env.commands = torch.tensor([[lin_x, lin_y, ang_z, base_height, toggle_jump*jump_height]]).to("cuda:0").repeat(num_envs, 1)
    # env.commands = torch.tensor([[lin_x, lin_y, ang_z, base_height, jump_height]], dtype=torch.float).to("cuda:0").repeat(num_envs, 1)

    # Start keyboard listener
    # listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    # listener.start()
    # max_iter = 300
    max_iter = len(motion_commands)

    reset_jump_toggle_iter = 0
    images_buffer = []
    commands_buffer = []
    with torch.no_grad():
        # while not stop:
        while True:
            if iter >= max_iter:
                break
                
            # Get current motion command
            lin_x, lin_y, ang_z, base_height, jump_height = motion_commands[iter]
            toggle_jump = True

            if iter % 30 == 0:
                print(f"Iter: {iter}, lin_x={lin_x:.2f}, lin_y={lin_y:.2f}")
   
            # env.cam_0.set_pose(lookat=env.base_pos.cpu().numpy()[0],)
            # env.cam_0.set_pose(pos=env.base_pos.cpu().numpy()[0] + np.array([0.5, 0.0, 0.5]) * iter / 50, lookat=env.base_pos.cpu().numpy()[0],)
                
            actions = policy(obs)
            # print(f"toggle_jump: {toggle_jump}, jump_height: {jump_height}")
             
            env.commands = torch.tensor([[lin_x, lin_y, ang_z, base_height, toggle_jump*jump_height]], dtype=torch.float).to("cuda:0").repeat(num_envs, 1)
            obs, _, rews, dones, infos = env.step(actions, is_train=False)
            # print(env.base_pos, env.base_lin_vel)
            if toggle_jump and reset_jump_toggle_iter == 0:
                reset_jump_toggle_iter = iter + 3
            if iter == reset_jump_toggle_iter and toggle_jump:
                toggle_jump = False
                reset_jump_toggle_iter = 0
            
            # Render the camera
            if env.cam_0 is not None:
                rgb, _, _, _ = env.cam_0.render(
                    rgb=True,
                    depth=False,
                    segmentation=False,
                )
                if args.save_data:
                    images_buffer.append(rgb)
                    # commands_buffer.append([lin_x, lin_y, ang_z, base_height, toggle_jump*jump_height])
                    commands_buffer.append([lin_x, lin_y, ang_z, base_height, jump_height])
            
            if dones.any():
                iter = 0
            
            iter += 1
          
    if args.save_data:
        # save the images and commands
        images_buffer = np.array(images_buffer)
        commands_buffer = np.array(commands_buffer)
        pickle.dump(images_buffer, open("images_buffer.pkl", "wb"))
        pickle.dump(commands_buffer, open("commands_buffer.pkl", "wb"))

if __name__ == "__main__":
    main()

"""
# Example usage:
# Default example sequence:
# python src/go2_eval_teleop.py -e my_experiment --ckpt 900
#
# Use inline key sequence:
# python src/go2_eval_teleop.py -e my_experiment --ckpt 900 --keys "wwasdjwwddjj"
#
# Load keys from a file:
# python src/go2_eval_teleop.py -e my_experiment --ckpt 900 --keys-file commands.txt
#
# Use default hardcoded sequence:
# python src/go2_eval_teleop.py -e my_experiment --ckpt 900 --use-default
#
# Save video data:
# python src/go2_eval_teleop.py -e my_experiment --ckpt 900 --keys "wwasdjj" --save-data
# 
# Original evaluation command:
# python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""