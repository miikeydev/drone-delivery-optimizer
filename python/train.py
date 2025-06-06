#!/usr/bin/env python3
"""
Complete training script for drone delivery with MaskablePPO and comprehensive logging
"""
import os
import sys
import time
import signal
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Try to import MaskablePPO
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    MASKABLE_PPO_AVAILABLE = True
    print("✅ MaskablePPO available - will use action masking")
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    print("⚠️ MaskablePPO not available - falling back to regular PPO")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import DroneDeliveryFullEnv


def make_env(graph_path: str, battery_init: int, payload_init: int, rank: int = 0,
             randomize_battery: bool = False, battery_range: tuple = (60, 100),
             randomize_payload: bool = False, payload_range: tuple = (1, 5),
             use_masking: bool = False, use_curriculum: bool = True):
    """Create environment factory for vectorization"""
    def _init():
        env = DroneDeliveryFullEnv(
            graph_path=graph_path,
            battery_init=battery_init,
            payload_init=payload_init,
            randomize_battery=randomize_battery,
            battery_range=battery_range,
            randomize_payload=randomize_payload,
            payload_range=payload_range,
            use_curriculum=use_curriculum,
            curriculum_threshold=0.25
        )
        
        if hasattr(env, 'seed'):
            env.seed(rank)
        
        if use_masking and MASKABLE_PPO_AVAILABLE:
            def mask_fn(env_instance):
                return env_instance.action_masks()
            env = ActionMasker(env, mask_fn)
        
        env = Monitor(env)
        return env
    
    set_random_seed(rank)
    return _init


class TrainingMetricsCallback(BaseCallback):
    """Simple callback for essential training metrics every 100K steps"""
    
    def __init__(self, save_freq=100000, verbose=1):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.save_freq = save_freq
        
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.success_rates = deque(maxlen=100)
        self.battery_usage = deque(maxlen=1000)
        self.pickup_rates = deque(maxlen=100)
        self.delivery_rates = deque(maxlen=100)
        self.curriculum_levels = deque(maxlen=100)
        
        self.start_time = time.time()
        self.best_recent_reward = -float('inf')
        self.training_interrupted = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\n🛑 Training interrupted by signal {signum}")
        self.training_interrupted = True
        
    def _on_step(self) -> bool:
        if self.training_interrupted:
            return False
            
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                reward = ep_info['r']
                length = ep_info['l']
                
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                
                # Track success metrics
                success = info.get('success', False)
                pickup_done = info.get('pickup_done', False)
                delivery_done = info.get('delivery_done', False)
                battery_used = info.get('battery_used_net', 0)
                curriculum_level = info.get('curriculum_level', 0)
                
                self.success_rates.append(1.0 if success else 0.0)
                self.pickup_rates.append(1.0 if pickup_done else 0.0)
                self.delivery_rates.append(1.0 if delivery_done else 0.0)
                self.battery_usage.append(battery_used)
                self.curriculum_levels.append(curriculum_level)
                
                # Track best reward
                if reward > self.best_recent_reward:
                    self.best_recent_reward = reward
        
        # Log every 100K steps
        if self.num_timesteps % self.save_freq == 0:
            self._log_metrics()
        
        return True
    
    def _log_metrics(self):
        """Log essential metrics every 100K steps"""
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate metrics
        mean_reward = np.mean(self.episode_rewards)
        success_rate = np.mean(self.success_rates) if self.success_rates else 0.0
        pickup_rate = np.mean(self.pickup_rates) if self.pickup_rates else 0.0
        delivery_rate = np.mean(self.delivery_rates) if self.delivery_rates else 0.0
        avg_curriculum = np.mean(self.curriculum_levels) if self.curriculum_levels else 0.0
        avg_length = np.mean(self.episode_lengths)
        avg_battery = np.mean(self.battery_usage) if self.battery_usage else 0.0
        
        # Calculate training speed
        elapsed_time = time.time() - self.start_time
        steps_per_sec = self.num_timesteps / elapsed_time
        
        # Simple table format
        print(f"\n🎯 STEP {self.num_timesteps:,}")
        print(f"| mean_reward             | {mean_reward:.1f}     |")
        print(f"| success_rate            | {success_rate:.1%}   |")
        print(f"| pickup_rate             | {pickup_rate:.1%}   |")
        print(f"| delivery_rate           | {delivery_rate:.1%}   |")
        print(f"| curriculum_level        | {avg_curriculum:.1f}     |")
        print(f"| training_speed          | {steps_per_sec:.0f}/s   |")
        print(f"| avg_episode_length      | {avg_length:.0f}     |")
        print(f"| avg_battery_used        | {avg_battery:.1f}%    |")
        print(f"| best_recent_reward      | {self.best_recent_reward:.1f}     |")
        
        # Determine if best route was success
        best_was_success = any(r >= self.best_recent_reward * 0.9 and s > 0 
                              for r, s in zip(self.episode_rewards, self.success_rates) 
                              if len(self.success_rates) > 0)
        print(f"| best_route              | {'SUCCESS' if best_was_success else 'FAILED'}  |")
        print("=" * 40)


def train_drone_delivery(graph_path: str, battery_init: int = 100, payload_init: int = 1,
                        total_timesteps: int = 1000000, n_envs: int = 8,
                        save_path: str = "models/best_model",
                        randomize_battery: bool = False, battery_range: tuple = (60, 100),
                        randomize_payload: bool = False, payload_range: tuple = (1, 5),
                        use_masking: bool = True, use_curriculum: bool = True):
    """Train PPO/MaskablePPO agent with simple best model saving"""
    
    print("Drone Delivery Training - Simple Best Model Save")
    print("=" * 50)
    algorithm_name = f"{'MaskablePPO' if use_masking and MASKABLE_PPO_AVAILABLE else 'PPO'}"
    
    print(f"Algorithm: {algorithm_name}")
    print(f"Graph: {graph_path}")
    print(f"Timesteps: {total_timesteps:,}, Envs: {n_envs}")
    print("=" * 50)
    
    os.makedirs("models", exist_ok=True)
    
    env = DummyVecEnv([
        make_env(graph_path, battery_init, payload_init, i,
                randomize_battery, battery_range,
                randomize_payload, payload_range, use_masking, use_curriculum)
        for i in range(n_envs)
    ])
    
    from stable_baselines3.common.vec_env import VecNormalize
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0)
    
    # SIMPLE: Eval env for best model callback only
    eval_env_base = DroneDeliveryFullEnv(
        graph_path=graph_path,
        battery_init=battery_init,
        payload_init=payload_init,
        randomize_battery=randomize_battery,
        battery_range=battery_range,
        randomize_payload=randomize_payload,
        payload_range=payload_range,
        use_curriculum=use_curriculum,
        curriculum_threshold=0.25
    )
    
    eval_env = DummyVecEnv([lambda: Monitor(eval_env_base)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, 
                           clip_obs=5.0, training=False)
    
    if use_masking and MASKABLE_PPO_AVAILABLE:
        model = MaskablePPO(
            "MlpPolicy", env,
            learning_rate=2e-4, n_steps=4096, batch_size=256, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.02, vf_coef=0.4, max_grad_norm=0.5, target_kl=0.03,
            verbose=0,  # CHANGED: verbose=0 to reduce spam
            policy_kwargs=dict(net_arch=[512, 256, 128], activation_fn=torch.nn.ReLU)
        )
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=2e-4, n_steps=4096, batch_size=256, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.02, vf_coef=0.4, max_grad_norm=0.5, target_kl=0.03,
            verbose=0,  # CHANGED: verbose=0 to reduce spam
            policy_kwargs=dict(net_arch=[512, 256, 128], activation_fn=torch.nn.ReLU)
        )
    
    # SIMPLE: Only essential callbacks
    metrics_callback = TrainingMetricsCallback(save_freq=100000, verbose=0)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        eval_freq=50000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=0  # CHANGED: verbose=0 to reduce spam
    )
    
    callbacks = [metrics_callback, eval_callback]
    
    try:
        print(f"🎯 Starting training for {total_timesteps:,} timesteps...")
        start_time = time.time()
        
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True) 
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time/3600:.2f} hours")
        
    except KeyboardInterrupt:
        print(f"🛑 Training interrupted by user")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            model.save("./models/best_model.zip")
            env.save("./models/best_model_vecnormalize.pkl")
            print(f"💾 Model saved to ./models/best_model.zip")
        except Exception as e:
            print(f"⚠️ Save error: {e}")
        
        try:
            env.close()
            eval_env.close()
        except:
            pass
    
    return model, "best_model"


def main():
    parser = argparse.ArgumentParser(description="Train PPO/MaskablePPO for drone delivery")
    parser.add_argument("--graph", type=str, required=True, help="Path to graph JSON file")
    parser.add_argument("--battery", type=int, default=100, help="Initial battery level")
    parser.add_argument("--payload", type=int, default=1, help="Initial payload")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--randomize-battery", action="store_true", help="Randomize initial battery")
    parser.add_argument("--battery-range", type=int, nargs=2, default=[60, 100], help="Battery randomization range")
    parser.add_argument("--battery-min", type=int, help="Minimum battery level")
    parser.add_argument("--battery-max", type=int, help="Maximum battery level")
    parser.add_argument("--randomize-payload", action="store_true", help="Randomize initial payload")
    parser.add_argument("--payload-range", type=int, nargs=2, default=[1, 5], help="Payload randomization range")
    parser.add_argument("--payload-min", type=int, help="Minimum payload")
    parser.add_argument("--payload-max", type=int, help="Maximum payload")
    parser.add_argument("--no-masking", action="store_true", help="Disable action masking")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    
    args = parser.parse_args()
    
    battery_range = args.battery_range
    if args.battery_min is not None and args.battery_max is not None:
        battery_range = [args.battery_min, args.battery_max]
    elif args.battery_min is not None or args.battery_max is not None:
        print("❌ Both --battery-min and --battery-max must be specified together")
        return
    
    payload_range = args.payload_range
    if args.payload_min is not None and args.payload_max is not None:
        payload_range = [args.payload_min, args.payload_max]
    elif args.payload_min is not None or args.payload_max is not None:
        print("❌ Both --payload-min and --payload-max must be specified together")
        return
    
    if not os.path.exists(args.graph):
        print(f"❌ Graph file not found: {args.graph}")
        return
    
    graph_path = os.path.abspath(args.graph)
    
    model, run_name = train_drone_delivery(
        graph_path=graph_path,
        battery_init=args.battery,
        payload_init=args.payload,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        save_path="models/best_model",  # FIXED: Simple path
        randomize_battery=args.randomize_battery,
        battery_range=tuple(battery_range),
        randomize_payload=args.randomize_payload,
        payload_range=tuple(payload_range),
        use_masking=not args.no_masking,
        use_curriculum=not args.no_curriculum
    )
    
    print(f"\n🎉 Training completed! Best model saved as models/best_model.zip")
    
    print(f"\n🎉 Training session '{run_name}' completed!")
    print(f"📂 Check ./logs/{run_name}/ for detailed training logs")
    print(f"📂 Check ./models/ for saved models")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
