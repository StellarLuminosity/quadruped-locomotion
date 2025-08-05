# ---------------------- Environment Configurations ----------------------
# Jump phase constants
JUMP_PHASE_PEAK_START = 0.3
JUMP_PHASE_PEAK_END = 0.6 
JUMP_PHASE_LANDING = 0.6 
JUMP_HEIGHT_TOLERANCE = 0.2 
JUMP_SPEED_MULTIPLIER = 0.2 

# Reward computation constants
TRACKING_SIGMA_DEFAULT = 0.25
HEIGHT_REWARD_SIGMA = 0.25
ACTIVE_MASK_THRESHOLD = 0.01 

# Physics constants
GRAVITY = 9.81
DEFAULT_DT = 0.02 

# Curriculum learning constants
CURRICULUM_HISTORY_SIZE = 100 
STABILITY_STAGE = 0
LOCOMOTION_STAGE = 1
AGILITY_STAGE = 2
MASTERY_STAGE = 3

# ---------------------- Training Configurations ----------------------
def get_env_config():
    """Environment configuration for physics simulation and robot setup."""
    return {
        "num_actions": 12,
        "default_joint_angles": {  # in rad
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        # PD Controller Parameters
        "kp": 20.0,  # Position gain
        "kd": 0.5,   # Velocity gain
        # Safety Limits
        "termination_if_roll_greater_than": 10,   # degrees
        "termination_if_pitch_greater_than": 10,  # degrees
        # Initial Pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # Episode Settings
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }

def get_observation_config():
    """Observation space configuration and scaling factors."""
    return {
        "num_obs": 48,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

def get_reward_config():
    """Reward function configuration and scaling weights."""
    return {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "jump_upward_velocity": 1.2,
        "jump_reward_steps": 50,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "jump_height_tracking": 0.5,
            "jump_height_achievement": 10,
            "jump_speed": 1.0,
            "jump_landing": 0.08,
        },
    }

def get_command_config():
    """Command space configuration for robot control."""
    return {
        "num_commands": 5,  # [lin_vel_x, lin_vel_y, ang_vel, height, jump]
        "lin_vel_x_range": [-1.0, 2.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.6, 0.6],
        "height_range": [0.2, 0.4],
        "jump_range": [0.5, 1.5],
    }

def get_ppo_config(exp_name, max_iterations):
    """PPO algorithm configuration for training."""
    return {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

# ---------------------- Evaluation Configurations ----------------------

def get_eval_env_config():
    """Modified environment config for evaluation (more lenient termination)."""
    config = get_env_config()
    config.update({
        "termination_if_roll_greater_than": 50,
        "termination_if_pitch_greater_than": 50,
    })
    return config

def get_eval_reward_config():
    """Evaluation reward config (typically disabled for pure inference)."""
    config = get_reward_config()
    config["reward_scales"] = {}  # Disable rewards during evaluation
    return config

# ---------------------- Teleoperation Configurations ----------------------

# Command vector format: [ lin_x, lin_y, ang_z, base_height, jump_height ]
default_key_commands = [
    [0.0, 0.0, 0.0, 0.3, 0.7], 
    [0.3, 0.0, 0.0, 0.3, 0.7], 
    [0.0, 0.3, 0.0, 0.3, 0.7], 
    [0.0, -0.3, 0.0, 0.3, 0.7], 
    [-0.3, 0.0, 0.0, 0.3, 0.7], 
    [0.0, 0.0, 0.0, 0.3, 0.7], 
]

key_commands = ['w', 'a', 's', 'd', 'j', 'f', 'd']
steps_per_transition = 35
transition_break_steps = 40  # Number of steps to pause at each edge before transitioning
jump_step = 5 # jump every # steps
num_envs = 50
output_video_path = "videos/my_experiment.mp4"
