

# ---------------------- Tele-operation Parameters ----------------------
# Command vector format: [ lin_x, lin_y, ang_z, base_height, jump_height ]
default_key_commands = [
    [1.0, 0.0, 0.0, 0.3, 0.7],    # forward
    [0.0, 1.0, 0.0, 0.3, 0.7],    # left
    [0.0, -1.0, 0.0, 0.3, 0.7],   # right
    [-1.0, 0.0, 0.0, 0.3, 0.7],   # backward
    [0.0, 0.0, 0.0, 0.3, 0.7],    # stop
]
key_commands = ['w', 'a', 's', 'd', 'j', 'f', 'd']
steps_per_key = 20  # Number of steps to apply each keypress
jump_step = 5 # jump every # steps
num_envs = 3
output_video_path = "videos/my_experiment.mp4"
