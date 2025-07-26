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
