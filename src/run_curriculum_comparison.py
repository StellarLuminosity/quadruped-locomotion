#!/usr/bin/env python3
"""
Curriculum Learning Comparison Script

This script runs comparative experiments between:
1. Original implicit curriculum learning
2. New adaptive curriculum learning

Educational Purpose:
- Demonstrates how to set up controlled experiments in RL
- Shows the importance of baseline comparisons in research
- Provides framework for analyzing curriculum learning effectiveness
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_experiment(exp_name, use_adaptive_curriculum=False, num_envs=4096, max_iterations=5000):
    """
    Run a single training experiment
    
    Args:
        exp_name: Name for the experiment (will be used as log directory)
        use_adaptive_curriculum: Whether to use adaptive curriculum
        num_envs: Number of parallel environments
        max_iterations: Maximum training iterations
    
    Educational Note:
    - We use fewer iterations (5000) for comparison experiments to save time
    - In real research, you'd want longer training (10000+ iterations)
    - Multiple seeds/runs would be needed for statistical significance
    """
    
    print(f"\n{'='*60}")
    curriculum_type = "ADAPTIVE" if use_adaptive_curriculum else "IMPLICIT"
    print(f"🚀 STARTING {curriculum_type} CURRICULUM EXPERIMENT")
    print(f"   Experiment Name: {exp_name}")
    print(f"   Environments: {num_envs}")
    print(f"   Max Iterations: {max_iterations}")
    print(f"   Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        sys.executable, "src/go2_train.py",
        "-e", exp_name,
        "-B", str(num_envs),
        "--max_iterations", str(max_iterations),
        "--device", "cuda:0"
    ]
    
    if use_adaptive_curriculum:
        cmd.append("--adaptive_curriculum")
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"\n✅ {curriculum_type} EXPERIMENT COMPLETED")
        print(f"   Duration: {(end_time - start_time)/60:.1f} minutes")
        print(f"   Logs saved to: logs/{exp_name}")
        
        return True, end_time - start_time
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {curriculum_type} EXPERIMENT FAILED")
        print(f"   Error: {e}")
        print(f"   Stdout: {e.stdout}")
        print(f"   Stderr: {e.stderr}")
        
        return False, 0

def main():
    """
    Run comparative curriculum learning experiments
    
    Educational Approach:
    1. Run baseline (implicit curriculum) first
    2. Run adaptive curriculum second
    3. Both use same hyperparameters for fair comparison
    4. Results can be compared using TensorBoard or custom analysis
    """
    
    print("🧪 CURRICULUM LEARNING COMPARISON STUDY")
    print("=" * 50)
    print("This script will run two experiments:")
    print("1. Baseline: Original implicit curriculum")
    print("2. Treatment: New adaptive curriculum")
    print("=" * 50)
    
    # Experiment parameters
    num_envs = 2048  # Reduced for faster experiments
    max_iterations = 3000  # Reduced for demonstration
    
    # Get timestamp for unique experiment names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiments = [
        {
            "name": f"baseline_implicit_{timestamp}",
            "adaptive": False,
            "description": "Original implicit curriculum (baseline)"
        },
        {
            "name": f"adaptive_curriculum_{timestamp}",
            "adaptive": True,
            "description": "New adaptive curriculum (treatment)"
        }
    ]
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n🔬 EXPERIMENT {i}/2: {exp['description']}")
        
        success, duration = run_experiment(
            exp_name=exp["name"],
            use_adaptive_curriculum=exp["adaptive"],
            num_envs=num_envs,
            max_iterations=max_iterations
        )
        
        results.append({
            "name": exp["name"],
            "success": success,
            "duration": duration,
            "adaptive": exp["adaptive"]
        })
        
        if not success:
            print(f"⚠️  Experiment {exp['name']} failed. Continuing with next experiment...")
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        curriculum_type = "Adaptive" if result["adaptive"] else "Implicit"
        duration_str = f"{result['duration']/60:.1f} min" if result["success"] else "N/A"
        
        print(f"{curriculum_type:12} | {status:10} | {duration_str:8} | {result['name']}")
    
    print(f"\n📈 NEXT STEPS:")
    print("1. Analyze training curves using TensorBoard:")
    for result in results:
        if result["success"]:
            print(f"   tensorboard --logdir logs/{result['name']}")
    
    print("\n2. Compare final performance using evaluation scripts:")
    for result in results:
        if result["success"]:
            print(f"   python src/go2_eval.py -e {result['name']} --ckpt 1000")
    
    print(f"\n3. Look for curriculum stage progression in adaptive experiment logs")
    print(f"4. Compare sample efficiency (reward vs. iteration curves)")

if __name__ == "__main__":
    main()
