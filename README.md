
# Quadruped Locomotion


This project implements a quadruped locomotion system using the Genesis physics engine and the RSL-RL framework. The system trains a Go2 quadruped robot to perform various locomotion tasks including walking, turning, and jumping.

<p align="center">
  <img src="sim.gif" alt="quadruped simulation" width="500"/>
</p>

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

```
quadruped-locomotion-og/
├── src/
│   ├── go2_train.py         # Main training script
│   ├── go2_eval_teleop.py   # Evaluation script with keyboard teleoperation
│   ├── go2_env.py           # Core environment with physics, rewards, and curriculum
│   ├── config.py            # All configuration parameters
│   └── utils.py             # Helper functions for video, commands, etc.
├── rsl_rl/                  # Git submodule for RSL-RL framework
├── genesis/                 # Git submodule for Genesis physics engine
├── logs/                    # Directory for storing training logs and model checkpoints
├── requirements.txt         # Python package dependencies
├── instructions.md          # Detailed setup and usage instructions
└── README.md                # This file
```

## Training Approach and Results

Exploration is split into 3 categories:
```python
self._sample_commands(envs_idx)                    # Normal commands
self._sample_commands(random_idxs_1)               # 5% random exploration
self._sample_jump_commands(random_idxs_2)          # 5% jump commands
```

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

### Curriculum vs Implicit Learning

In this project, I explored both curriculum learning and implicit learning approaches for training the quadruped robot. After training both models for the same number of steps, the curriculum learning approach demonstrated superior performance in terms of stability and convergence speed.

![Curriculum Learning](curr.gif) ![Implicit Learning](impl.gif)

*Left: Curriculum learning shows more stable gait and better command following. Right: Implicit learning struggles with stability and direction.*

The curriculum learning approach uses a 5-stage progressive training system that gradually increases complexity:
1. **Foundation**: Basic stability and posture
2. **Basic Locomotion**: Forward/backward movement
3. **Advanced Locomotion**: Directional control and turning
4. **Agility**: Higher speeds and dynamic movements
5. **Mastery**: Complex behaviors including jumping


This staged approach allows the agent to master fundamental skills before moving to more complex tasks, resulting in more stable and reliable locomotion compared to the implicit learning approach which trains on all aspects simultaneously. 

## Code Features

### Sim-to-Real Transfer

This code is designed for simulation-to-real transfer with:
- Domain Randomization: Random commands, noise, varying episode lengths
- Realistic Physics: Genesis provides accurate contact dynamics
- Action Latency: Simulates real robot communication delays
- Robust Rewards: Exponential rewards are more forgiving than linear

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

#### Command Vector Breakdown:
```python
self.commands[env_idx, 0] = lin_vel_x    # Forward/backward velocity (-1.0 to 2.0 m/s)
self.commands[env_idx, 1] = lin_vel_y    # Left/right velocity (-0.5 to 0.5 m/s)  
self.commands[env_idx, 2] = ang_vel      # Turning velocity (-0.6 to 0.6 rad/s)
self.commands[env_idx, 3] = height       # Base height (0.2 to 0.4 m)
self.commands[env_idx, 4] = jump         # Jump height (0.5 to 1.5 m)
```

### Commands
Every command is a list of 5 values:
```python
self.commands[env_idx, 0] = lin_vel_x    # forward (positive) / backward (negative) linear velocity (m/s) (-1.0 to 2.0 m/s)
self.commands[env_idx, 1] = lin_vel_y    # left (positive) / right (negative) lateral velocity (m/s)(-0.5 to 0.5 m/s)  
self.commands[env_idx, 2] = ang_vel      # Turning velocity (-0.6 to 0.6 rad/s)
self.commands[env_idx, 3] = height       # Base height (0.2 to 0.4 m)
self.commands[env_idx, 4] = jump         # Jump height (0.5 to 1.5 m)
```
This code is modeled after the example genesis code: [Genesis Locomotion](https://github.com/Genesis-Embodied-AI/Genesis/tree/806d0a8d84512ff1982330a684bad920ec4262fe/examples/locomotion)
