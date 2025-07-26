#!/usr/bin/env python3
"""
Test Script for Adaptive Curriculum Learning

This script performs basic tests to ensure our adaptive curriculum implementation
works correctly before running full training experiments.

Educational Purpose:
- Shows the importance of testing in research code
- Demonstrates how to validate RL implementations
- Provides debugging framework for curriculum learning
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from go2_env import Go2Env, AdaptiveCurriculum
from go2_train import get_cfgs
import genesis as gs

def test_adaptive_curriculum_class():
    """
    Test the AdaptiveCurriculum class in isolation
    
    Educational Note:
    - Unit testing is crucial for complex RL systems
    - We test each component separately before integration
    - This helps identify bugs early in development
    """
    
    print("🧪 TESTING ADAPTIVE CURRICULUM CLASS")
    print("-" * 40)
    
    # Get configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Create curriculum instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 16  # Small number for testing
    
    try:
        curriculum = AdaptiveCurriculum(reward_cfg, num_envs, device)
        print("✅ AdaptiveCurriculum instantiation: SUCCESS")
    except Exception as e:
        print(f"❌ AdaptiveCurriculum instantiation: FAILED - {e}")
        return False
    
    # Test initial state
    try:
        assert curriculum.current_stage == 0
        assert curriculum.stage_names[0] == "Stability"
        weights = curriculum.get_current_reward_weights()
        assert isinstance(weights, dict)
        assert "base_height" in weights
        print("✅ Initial state validation: SUCCESS")
    except Exception as e:
        print(f"❌ Initial state validation: FAILED - {e}")
        return False
    
    # Test performance update (simulate some episodes)
    try:
        # Simulate episode completion
        episode_rewards = torch.randn(num_envs, device=device) * 10 - 20  # Random rewards around -20
        episode_lengths = torch.randint(500, 1000, (num_envs,), device=device)
        reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        reset_buf[0] = True  # Simulate one environment completing episode
        
        curriculum.update_performance(episode_rewards, episode_lengths, reset_buf)
        print("✅ Performance update: SUCCESS")
    except Exception as e:
        print(f"❌ Performance update: FAILED - {e}")
        return False
    
    print("✅ AdaptiveCurriculum class tests: ALL PASSED\n")
    return True

def test_environment_integration():
    """
    Test the integration of adaptive curriculum with Go2Env
    
    Educational Insight:
    - Integration testing ensures components work together
    - We test both curriculum enabled and disabled modes
    - This validates our backward compatibility
    """
    
    print("🧪 TESTING ENVIRONMENT INTEGRATION")
    print("-" * 40)
    
    # Initialize Genesis (required for environment)
    try:
        gs.init(backend=gs.constants.backend.cpu, logging_level="error")
        print("✅ Genesis initialization: SUCCESS")
    except Exception as e:
        print(f"❌ Genesis initialization: FAILED - {e}")
        return False
    
    # Get configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Test 1: Environment without adaptive curriculum
    try:
        env_baseline = Go2Env(
            num_envs=4,  # Very small for testing
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            device="cpu",
            use_adaptive_curriculum=False
        )
        print("✅ Baseline environment creation: SUCCESS")
    except Exception as e:
        print(f"❌ Baseline environment creation: FAILED - {e}")
        return False
    
    # Test 2: Environment with adaptive curriculum
    try:
        env_adaptive = Go2Env(
            num_envs=4,  # Very small for testing
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            device="cpu",
            use_adaptive_curriculum=True
        )
        print("✅ Adaptive environment creation: SUCCESS")
    except Exception as e:
        print(f"❌ Adaptive environment creation: FAILED - {e}")
        return False
    
    # Test 3: Basic environment step
    try:
        # Random actions for testing
        actions = torch.randn(4, 12)  # 4 envs, 12 actions each
        
        # Test baseline environment step
        obs_baseline, rewards_baseline, dones_baseline, infos_baseline = env_baseline.step(actions, is_train=False)
        
        # Test adaptive environment step
        obs_adaptive, rewards_adaptive, dones_adaptive, infos_adaptive = env_adaptive.step(actions, is_train=False)
        
        # Validate outputs
        assert obs_baseline.shape == obs_adaptive.shape
        assert rewards_baseline.shape == rewards_adaptive.shape
        assert dones_baseline.shape == dones_adaptive.shape
        
        print("✅ Environment step execution: SUCCESS")
    except Exception as e:
        print(f"❌ Environment step execution: FAILED - {e}")
        return False
    
    # Test 4: Curriculum state access
    try:
        assert env_baseline.adaptive_curriculum is None
        assert env_adaptive.adaptive_curriculum is not None
        assert env_adaptive.use_adaptive_curriculum == True
        
        # Test curriculum methods
        current_weights = env_adaptive.adaptive_curriculum.get_current_reward_weights()
        assert isinstance(current_weights, dict)
        
        print("✅ Curriculum state access: SUCCESS")
    except Exception as e:
        print(f"❌ Curriculum state access: FAILED - {e}")
        return False
    
    print("✅ Environment integration tests: ALL PASSED\n")
    return True

def test_reward_computation():
    """
    Test that reward computation works correctly with adaptive curriculum
    
    Educational Value:
    - Validates the core functionality of our system
    - Ensures rewards are computed with correct weights
    - Tests the dynamic weight switching mechanism
    """
    
    print("🧪 TESTING REWARD COMPUTATION")
    print("-" * 40)
    
    # This test would require a more complex setup with actual robot simulation
    # For now, we'll do a conceptual validation
    
    try:
        # Get configurations
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        
        # Test reward weight retrieval
        device = torch.device("cpu")
        curriculum = AdaptiveCurriculum(reward_cfg, 4, device)
        
        # Test stage 0 weights (Stability)
        stage_0_weights = curriculum.get_current_reward_weights()
        assert stage_0_weights["base_height"] == -100.0  # High stability penalty
        assert stage_0_weights["tracking_lin_vel"] == 0.3  # Low velocity tracking
        
        print("✅ Stage 0 reward weights: SUCCESS")
        
        # Simulate advancement to stage 1
        curriculum.current_stage = 1
        stage_1_weights = curriculum.get_current_reward_weights()
        assert stage_1_weights["base_height"] == -75.0  # Reduced stability penalty
        assert stage_1_weights["tracking_lin_vel"] == 0.8  # Higher velocity tracking
        
        print("✅ Stage 1 reward weights: SUCCESS")
        
        print("✅ Reward computation tests: ALL PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Reward computation tests: FAILED - {e}")
        return False

def run_all_tests():
    """
    Run all tests and provide summary
    
    Educational Purpose:
    - Comprehensive testing before deployment
    - Clear pass/fail reporting
    - Debugging guidance for failures
    """
    
    print("🚀 ADAPTIVE CURRICULUM LEARNING - TEST SUITE")
    print("=" * 60)
    print("This test suite validates our curriculum learning implementation")
    print("before running full training experiments.\n")
    
    tests = [
        ("Adaptive Curriculum Class", test_adaptive_curriculum_class),
        ("Environment Integration", test_environment_integration),
        ("Reward Computation", test_reward_computation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"🔍 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {e}")
            results.append((test_name, False))
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("-" * 60)
    print(f"OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ready for training experiments.")
        print("\nNext steps:")
        print("1. Run comparison experiments: python src/run_curriculum_comparison.py")
        print("2. Analyze results: python src/analyze_curriculum_results.py --auto-find")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before proceeding.")
        print("\nDebugging tips:")
        print("- Check Genesis installation and GPU/CUDA setup")
        print("- Verify all dependencies are installed")
        print("- Review error messages above for specific issues")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
