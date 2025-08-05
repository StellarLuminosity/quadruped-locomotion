# Quadrupeds Locomotion

A reinforcement learning project for quadruped robot locomotion using RSL-RL framework.

## Setup

1. Clone the repository:
```bash
git clone git@github.com:Argo-Robot/quadrupeds_locomotion.git --recursive
cd quadrupeds_locomotion
```
2. Initialize the submodules:
```bash
git submodule update --init --recursive
```

4. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

5. Install dependencies:
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

Train the Go2 robot:
```bash
python src/go2_train.py --exp_name="my_experiment" --max_iterations=1000
```
Pass the `--adaptive_curriculum` flag to enable adaptive curriculum learning.

Generate evaluations and create video:
```bash
python src/go2_eval_teleop.py -e my_experiment --ckpt 900
```
Pass the `--adaptive_curriculum` flag if loading a model trained with curriculum learning.

## Project Structure
```
quadrupeds_locomotion/
├── src/
│   ├── go2_train.py    # Training script
│   └── go2_env.py      # Environment
├── rsl_rl/             # RSL-RL framework
└── requirements.txt     # Project dependencies
```





## Installation

1. Clone the repository with submodules:
```bash
git clone <repository-url> --recursive
cd quadruped-locomotion
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
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
python src/go2_train.py --exp_name="my_experiment" --max_iterations=10000
```

Enable adaptive curriculum learning:
```bash
python src/go2_train.py --exp_name="my_experiment" --max_iterations=10000 --adaptive_curriculum
```

### Evaluation

Evaluate a specific checkpoint and generate a video:
```bash
python src/go2_eval_teleop.py -e my_experiment --ckpt 5000
```

If evaluating a model trained with curriculum learning:
```bash
python src/go2_eval_teleop.py -e my_experiment --ckpt 5000 --adaptive_curriculum
```

The evaluation script will generate an annotated video showing the robot's performance with command overlays in the `videos/my_experiment/` directory.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request
