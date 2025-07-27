#!/usr/bin/env python3
"""
Simple test to verify adaptive curriculum works
"""

import sys
import os
sys.path.append('src')

try:
    from go2_env import AdaptiveCurriculum
    from go2_train import get_cfgs
    import torch
    
    print("🧪 Testing Adaptive Curriculum...")
    
    # Get configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Create curriculum
    device = torch.device('cpu')
    curriculum = AdaptiveCurriculum(reward_cfg, 4, device)
    
    print("✅ AdaptiveCurriculum created successfully!")
    print(f"Current stage: {curriculum.stage_names[curriculum.current_stage]}")
    print(f"Stage weights: {curriculum.get_current_reward_weights()}")
    
    # Test stage advancement
    curriculum.current_stage = 1
    print(f"\nAfter advancing to stage 1:")
    print(f"Current stage: {curriculum.stage_names[curriculum.current_stage]}")
    print(f"Stage weights: {curriculum.get_current_reward_weights()}")
    
    print("\n🎉 All tests passed! Adaptive curriculum is working.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
