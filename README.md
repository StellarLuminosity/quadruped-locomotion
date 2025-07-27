# Quadruped Locomotion

This project implements a quadruped locomotion system using the Genesis physics engine and the RSL-RL framework.

![Video](./output_video.mp4)

## Components
Genesis provides realistic physics simulation
RSL-RL handles the PPO training infrastructure
Multi-task learning enables walking, jumping, and height control
Careful reward design balances multiple objectives
Robust observation space enables effective policy learning

## Architecture of the project:

### Robotics Pipeline:
Training Pipeline:
go2_train.py → go2_env.py → Genesis Physics → RSL-RL Training

Evaluation Pipeline:
go2_eval.py → Trained Model → go2_env.py → Genesis Physics
go2_eval_teleop.py → Trained Model → go2_env.py → Genesis Physics (with user control)

### File Relationships:
go2_env.py
 - The core environment that all other files depend on
go2_train.py
 - Uses go2_env.py to train the robot
go2_eval.py
 - Uses go2_env.py + trained model for evaluation
go2_eval_teleop.py
 - Uses go2_env.py + trained model for interactive control

### Env Notes

PD controller: stands for Proportional-Derivative controller. It’s a simple but powerful feedback control strategy widely used in robotics to control motors and joints.
- Proportional (P): Reacts to the current error (difference between where you want the joint to be and where it actually is) (higher = correct errors more agressively).
- Derivative (D): Reacts to the rate of change of the error (how quickly the error is changing) (higher = correct errors more slowly).

PD controllers are used to control the motors of the robot to achieve the desired joint angles. The PD controller is a feedback controller that uses the error between the desired joint angle and the actual joint angle to control the motor.

The controller computes the torque (force) to apply to each joint as:
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

# NOtes about Curriculm Learning

Curriculum learning is inspired by how humans learn - we start with simple concepts and gradually progress to more complex ones. In reinforcement learning, this means:
1. Traditional RL: Agent learns all tasks simultaneously, which can be inefficient
2. Curriculum Learning: Agent learns tasks in a structured progression from easy to hard

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


