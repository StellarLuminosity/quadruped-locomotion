
# Quadruped Locomotion

This project implements a quadruped locomotion system using the Genesis physics engine and the RSL-RL framework. The system trains a Go2 quadruped robot to perform various locomotion tasks including walking, turning, and jumping.

![til](sim.gif)

## Features

- **Multi-task Learning**: Single policy handles walking, turning, height control, and jumping
- **Adaptive Curriculum Learning**: Optional progressive learning stages from basic stability to advanced agility
- **Domain Randomization**: Sim-to-real transfer techniques for robust policies
- **Teleoperation Interface**: Interactive control and evaluation of trained policies
- **Video Generation**: Automated creation of evaluation videos with command overlays

## Project Architecture

### Core Components

- **Genesis Physics Engine**: Provides realistic physics simulation for the quadruped robot
- **RSL-RL Framework**: Implements PPO (Proximal Policy Optimization) for reinforcement learning
- **Go2 Robot Model**: Accurate model of the quadruped with 12 actuated joints (3 per leg)
- **PD Controller**: Low-level joint control using Proportional-Derivative feedback

### Pipeline Structure

```
Training Pipeline:
go2_train.py → go2_env.py → Genesis Physics → RSL-RL Training

Evaluation Pipeline:
go2_eval_teleop.py → Trained Model → go2_env.py → Genesis Physics → Video Output
```

### File Structure

- **go2_env.py**: Core environment implementation with physics, rewards, and curriculum
- **go2_train.py**: Training script with PPO algorithm and curriculum learning options
- **go2_eval_teleop.py**: Evaluation script with teleoperation and video generation
- **config.py**: Configuration parameters for environment, training, and evaluation
- **utils.py**: Utility functions for video rendering, command processing, and more

## Key Technical Concepts

### Adaptive Curriculum Learning

The system implements a 4-stage curriculum that progressively increases task complexity:

1. **Stability Stage**: Focus on balance and basic standing
2. **Locomotion Stage**: Learn basic walking and turning
3. **Agility Stage**: Master complex movements and jumping
4. **Mastery Stage**: Optimize for efficiency and robustness

Each stage modifies reward weights to emphasize different aspects of performance.

### PD Controller

The Proportional-Derivative controller manages the low-level joint control:

- **Proportional (P)**: Responds to position error (difference between target and actual joint angles)
- **Derivative (D)**: Responds to velocity error (dampens oscillations)

The controller computes torque as: `torque = kp * position_error + kd * velocity_error`

### Reward System

The reward function balances multiple objectives:
- Tracking commanded velocities (linear and angular)
- Maintaining commanded body height
- Penalizing energy-inefficient movements
- Encouraging successful jumping when commanded
- Ensuring stable landing after jumps
```python
torque = Kp * (desired_position - actual_position) + Kd * (desired_velocity - actual_velocity)
```

#### Command Vector Breakdown:
```python
self.commands[env_idx, 0] = lin_vel_x    # Forward/backward velocity (-1.0 to 2.0 m/s)
self.commands[env_idx, 1] = lin_vel_y    # Left/right velocity (-0.5 to 0.5 m/s)  
self.commands[env_idx, 2] = ang_vel      # Turning velocity (-0.6 to 0.6 rad/s)
self.commands[env_idx, 3] = height       # Base height (0.2 to 0.4 m)
self.commands[env_idx, 4] = jump         # Jump height (0.5 to 1.5 m)
```

### Training Notes
In robotics, there's an important distinction:
- High-level Commands: What you want the robot to do ("walk forward at 1 m/s")
- Low-level Actions: How the robot actually does it (specific joint angles/torques)
The RL agent learns to translate commands → actions.

### Complete training workflow

1. Training Pipeline:
```python
# Step 1: Train the policy
python src/go2_train.py -e my_experiment --num_envs 4096 --max_iterations 10000
```

This creates:
- logs/my_experiment/model_100.pt (checkpoint every 100 iterations)
- logs/my_experiment/model_200.pt
- ...
- logs/my_experiment/cfgs.pkl (configuration backup)

Training Process:
- Initialize 4096 parallel environments
- Collect 24 steps × 4096 envs = 98,304 samples per iteration
- Update policy using PPO for 5 epochs
- Repeat for 10,000 iterations (≈ 1 billion samples total)

2. Evaluation Pipeline:
```python
# Step 2: Evaluate the policy
python src/go2_eval.py -e my_experiment --ckpt 1000

# Step 3: Interactive control
python src/go2_eval_teleop.py -e my_experiment --ckpt 1000
```

## Code Features

### Sim-to-Real Transfer

This code is designed for simulation-to-real transfer with:
- Domain Randomization: Random commands, noise, varying episode lengths
- Realistic Physics: Genesis provides accurate contact dynamics
- Action Latency: Simulates real robot communication delays
- Robust Rewards: Exponential rewards are more forgiving than linear

### Curriculum Learning

The code implements curriculum learning with:
- Easy commands early in training, harder commands later
```python
self._sample_commands(envs_idx)                    # Normal commands
self._sample_commands(random_idxs_1)               # 5% random exploration
self._sample_jump_commands(random_idxs_2)          # 5% jump commands
```

### Multi-Task Learning
The robot learns multiple behaviors simultaneously:
- Locomotion: Walking, running, turning
- Height Control: Crouching, standing tall
- Jumping: Dynamic maneuvers
- Recovery: Getting up after falls

### Observation Design
```python
obs = [
    base_ang_vel,           # 3D - "Am I rotating?"
    projected_gravity,      # 3D - "Which way is up?"
    commands,               # 5D - "What should I do?"
    dof_pos,               # 12D - "Where are my joints?"
    dof_vel,               # 12D - "How fast are my joints moving?"
    actions,               # 12D - "What did I just do?"
    jump_phase_info        # 1D - "Am I jumping?"
]
```

Design Principles:
- Proprioceptive: Robot's internal state (no external sensors)
- Markovian: Contains all info needed for decision-making
- Normalized: All values scaled to similar ranges

The original code had an implicit curriculum that emerged naturally from:
- Large penalties for instability (base height penalty of -50.0) forced the robot to learn balance first
- Command resampling gradually exposed the robot to different velocities and behaviors
- Reward magnitudes naturally prioritized stability over advanced locomotion

The New System:
1. AdaptiveCurriculum Class Design
- Explicitly defines curriculum stages with clear progression

The 4 Stages:
- Stability: Learn basic balance and standing (high base_height penalty)
- Locomotion: Master walking and turning (emphasize velocity tracking)
- Agility: Add jumping and complex movements (introduce jump rewards)
- Mastery: Optimize everything together (original balanced weights)

2. Performance Tracking & Automatic Advancement
- Data-driven: Progression based on actual performance, not arbitrary time
- Adaptive: Each robot learns at its own pace
- Robust: Requires both success rate AND minimum experience

1. Structured Learning: Each stage builds on the previous one
2. Automatic Adaptation: No manual tuning of when to advance
3. Backward Compatible: Can be toggled on/off for comparison
4. Performance-Based: Advancement based on actual learning progress

#### Commands
Every command is a list of 5 values:
```python
self.commands[env_idx, 0] = lin_vel_x    # forward (positive) / backward (negative) linear velocity (m/s) (-1.0 to 2.0 m/s)
self.commands[env_idx, 1] = lin_vel_y    # left (positive) / right (negative) lateral velocity (m/s)(-0.5 to 0.5 m/s)  
self.commands[env_idx, 2] = ang_vel      # Turning velocity (-0.6 to 0.6 rad/s)
self.commands[env_idx, 3] = height       # Base height (0.2 to 0.4 m)
self.commands[env_idx, 4] = jump         # Jump height (0.5 to 1.5 m)
```
This code is modeled and implemented after: **Federico Sarrocco, Leonardo Bertelli (2025)**: [*Making Quadrupeds Learning to Walk: From Zero to Hero*](https://federicosarrocco.com/blog/Making-Quadrupeds-Learning-To-Walk)
