import os
import numpy as np
import cv2

# -------------------- Sim-to-real noise helpers --------------------

def gs_rand_float(lower, upper, shape, device):
    """Generates random numbers for domain randomization. Adds variability to training (different starting positions, noise, etc.)"""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def gs_rand_gaussian(mean, min, max, n_std, shape, device):
    """Generates gaussian-distributed random values for more realistic noise patterns for robot sensors/actuators"""
    mean_tensor = mean.expand(shape).to(device)
    std_tensor = torch.full(shape, (max - min) / 4.0 * n_std, device=device)
    return torch.clamp(torch.normal(mean_tensor, std_tensor), min, max)


def gs_additive(base, increment):
    """Adds incremental changes to base values for robot sensor/actuator noise"""
    return base + increment

# ------------------- Video rendering helpers -------------------

def _safe_max(values):
    """Return the maximum absolute value from a sequence, with a minimum threshold of 1e-6 to avoid division by zero."""
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

def draw_joystick(image, lin_x, lin_y, ang_z, base_height, max_lin_x, max_lin_y, radius=100, x_offset=10, y_offset=10):
    """Draw a joystick visualization with position indicator on the given image."""
    for i in range(radius):
        r = radius - i
        color = (int(55+200 * (0.5 + 0.5 * i / radius)), int(55+200 * (0.5 + 0.5 * i / radius)), int(55+200 * (0.5 + 0.5 * i / radius)))
        cv2.circle(image, (x_offset + radius, y_offset + radius), r, color, -1)

    joystick_x = int(x_offset + radius + (lin_y / max_lin_y) * radius)
    joystick_y = int(y_offset + radius - (lin_x / max_lin_x) * radius)
    cv2.circle(image, (joystick_x + 2, joystick_y + 2), int(radius * 0.12), (0, 0, 0), -1)
    return image


def draw_target_height_bar(image, base_height, max_base_height, target_height=1.0, x_offset=220, y_offset=10):
    """Draw a vertical bar showing the current and target base height on the image."""
    base_height = max(0, base_height)
    bar_width = 20
    bar_height = 200
    bar_x = x_offset
    bar_y = y_offset

    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
    current_height_pos = int(bar_y + bar_height - (base_height / max_base_height) * bar_height)
    cv2.rectangle(image, (bar_x, current_height_pos), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
    target_height_pos = int(bar_y + bar_height - (target_height / target_height) * bar_height)
    cv2.line(image, (bar_x, target_height_pos), (bar_x + bar_width, target_height_pos), (0, 0, 255), 2)

    return image

def draw_angular_velocity_bar(image, ang_z, max_ang_z, x_offset=10, y_offset=220):
    """Draw a horizontal bar showing the current angular velocity on the image."""
    bar_width = 200
    bar_height = 20
    bar_x = x_offset
    bar_y = y_offset

    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
    current_ang_pos = int(bar_x + (ang_z / max_ang_z + 1) / 2 * bar_width)
    cv2.rectangle(image, (bar_x, bar_y), (current_ang_pos, bar_y + bar_height), (0, 255, 0), -1)

    return image

def create_video_with_overlay(images_buffer, commands_buffer, output_path, fps=30):
    """Create a video with joystick and command overlays from a sequence of images."""
    height, width, _ = images_buffer[0].shape
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    print(f"Creating video at: {os.path.abspath(output_path)}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return

    max_lin_x, max_lin_y, max_ang_z, max_base_height, _ = normalize_commands(commands_buffer)

    for i in range(len(images_buffer)):
        image = images_buffer[i]
        lin_x, lin_y, ang_z, base_height, _ = commands_buffer[i]
        
        x_offset = images_buffer[0].shape[1] // 2 - 100
        y_offset = images_buffer[0].shape[0] - 250
        radius = 100
        
        image = draw_joystick(image, lin_x, lin_y, ang_z, base_height,
                            max_lin_x, max_lin_y, radius=radius, x_offset=x_offset, y_offset=y_offset)
        image = draw_target_height_bar(image, base_height, max_base_height, 
                                     x_offset=x_offset + radius*2 + 10, y_offset=y_offset)
        image = draw_angular_velocity_bar(image, ang_z, max_ang_z, 
                                        x_offset=x_offset, y_offset=y_offset + radius*2 + 20)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        success = out.write(image)
        if not success:
            print(f"Warning: Failed to write frame {i} to video")

    out.release()
    print(f"Video saved to: {os.path.abspath(output_path)}")
    print(f"Video file exists: {os.path.exists(output_path)}")
    print(f"File size: {os.path.getsize(output_path) if os.path.exists(output_path) else 0} bytes")

# ---------------------- Command Sequence Helpers ----------------------

def interpolate_commands(commands, steps_per_transition):
    """Generate smooth transitions between command vectors with the specified number of interpolation steps."""
    result = []
    for i in range(len(commands) - 1):
        start = np.array(commands[i])
        end = np.array(commands[i + 1])
        for alpha in np.linspace(0, 1, steps_per_transition):
            interp = (1 - alpha) * start + alpha * end
            result.append(interp.tolist())
    return result

# ---------------------- Checkpoint Helpers ----------------------

def get_checkpoints(exp_dir, interval=5000):
    """Find and return checkpoint numbers that are multiples of the specified interval."""
    import re
    import glob
    
    model_files = glob.glob(os.path.join(exp_dir, 'model_*.pt'))
    
    checkpoints = []
    for f in model_files:
        match = re.search(r'model_(\d+)\.pt$', f)
        if match:
            ckpt = int(match.group(1))
            if ckpt % interval == 0:
                checkpoints.append(ckpt)
    
    return sorted(checkpoints)

