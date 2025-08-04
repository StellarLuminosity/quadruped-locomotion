# Training script for Go2 quadruped robot
import os 
import argparse
import pickle
import shutil
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

import config
import genesis as gs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use: 'cpu' or 'cuda:0'")
    parser.add_argument("--adaptive_curriculum", action="store_true", 
                       help="Enable adaptive curriculum learning with explicit stages")
    args = parser.parse_args()

    backend = gs.constants.backend.gpu if args.device.lower() == "cuda:0" else gs.constants.backend.cpu
    gs.init(logging_level="warning", backend=backend)

    log_dir = f"logs/{args.exp_name}"

    env_cfg = config.get_env_config()
    obs_cfg = config.get_observation_config()
    reward_cfg = config.get_reward_config()
    command_cfg = config.get_command_config()
    train_cfg = config.get_ppo_config(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=args.device,
        use_adaptive_curriculum=args.adaptive_curriculum,
    )
    
    if args.adaptive_curriculum:
        print(f"\n Training with adaptive curriculum")
        print(f"   Experiment: {args.exp_name}")
        print(f"   Environments: {args.num_envs}")
        print(f"   Max Iterations: {args.max_iterations}")
    else:
        print(f"\n No adaptive curriculum")
        print(f"   Experiment: {args.exp_name}")
        print(f"   Environments: {args.num_envs}")
        print(f"   Max Iterations: {args.max_iterations}")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

