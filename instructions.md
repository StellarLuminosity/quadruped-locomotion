# Quadruped Locomotion

A reinforcement learning project for quadruped robot locomotion using the Genesis physics engine and RSL-RL framework. This project implements training and evaluation for a Go2 robot with support for adaptive curriculum learning.

## Project Structure

```
quadruped-locomotion/
├── src/
│   ├── go2_train.py       # Training script
│   ├── go2_eval_teleop.py # Evaluation script with teleoperation
│   ├── go2_env.py         # Environment with adaptive curriculum
│   ├── config.py          # Configuration parameters
│   └── utils.py           # Utility functions
├── rsl_rl/                # RSL-RL framework (submodule)
├── genesis/               # Genesis physics engine (submodule)
├── logs/                  # Training logs and checkpoints
├── videos/                # Evaluation videos
└── requirements.txt       # Project dependencies
```

## Adaptive Curriculum Learning

The project implements a 5-stage curriculum learning approach:

1. **Foundation**: Focus on basic stability and posture
2. **Basic Locomotion**: Simple forward/backward movement
3. **Advanced Locomotion**: Directional movement and turning
4. **Agility**: Higher speeds and more dynamic movements
5. **Mastery**: Complex behaviors including jumping

Each stage has:
- Custom reward weights
- Success criteria for advancement
- Minimum episode requirements

To enable curriculum learning, use the `--adaptive_curriculum` flag during both training and evaluation.

## Installation

1. Clone the repository with submodules:
```bash
git clone https://github.com/StellarLuminosity/quadruped-locomotion.git --recursive
cd quadruped-locomotion
```

2. Initialize the submodules:
```bash
git submodule update --init --recursive
```

3. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/Mac
python -m venv venv
source venv/bin/activate
```

4. Install dependencies:
```bash
# Core dependencies
pip install -r requirements.txt

# Install RSL-RL
cd rsl_rl
pip install -e .
cd ..

# Install Genesis
cd genesis
pip install -e .
cd ..
```

## Usage

### Training

Train the Go2 robot with standard settings:
```bash
python src/go2_train.py --exp_name="my_experiment" --max_iterations=1000
```

Enable adaptive curriculum learning for more stable training:
```bash
python src/go2_train.py --exp_name="my_experiment" --max_iterations=1000 --adaptive_curriculum
```

The training script will:
- Create a `logs/my_experiment/` directory with training checkpoints and metrics
- Save model checkpoints at regular intervals (default: every 100 iterations)
- Log training progress and rewards

### Evaluation

Evaluate a specific checkpoint and generate a video:
```bash
python src/go2_eval_teleop.py -e my_experiment --ckpt 5000
```

If evaluating a model trained with curriculum learning:
```bash
python src/go2_eval_teleop.py -e my_experiment --ckpt 5000 --adaptive_curriculum
```

The evaluation script will:
- Load the specified checkpoint
- Run the robot through a sequence of movement commands
- Generate an annotated video showing the robot's performance with command overlays
- Save the video in the `videos/my_experiment/` directory

### Configuration

Key configuration parameters can be modified in `src/config.py`:

- `steps_per_transition`: Number of steps for each command transition (default: 60)
- `transition_break_steps`: Pause duration at each transition point (default: 30)
- `default_key_commands`: Sequence of commands for evaluation
- PD controller gains: `kp` (position gain) and `kd` (velocity gain)
- Reward scales and weights for different behaviors

## Troubleshooting

- **CUDA errors**: Ensure you have compatible NVIDIA drivers installed
- **Import errors**: Verify all dependencies are installed and submodules are initialized
- **Video generation issues**: Check that OpenCV is properly installed
- **Robot falling too frequently**: Try training with the adaptive curriculum enabled
