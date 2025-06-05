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


class ActionMaskerWrapper:
    """Wrapper to provide action masking for MaskablePPO"""
    def __init__(self, env):
        self.env = env
        
    def mask_fn(self, env):
        # Handle wrapped environments by finding the underlying environment
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env') and not hasattr(unwrapped_env, 'action_masks'):
            unwrapped_env = unwrapped_env.env
        
        if hasattr(unwrapped_env, 'action_masks'):
            return unwrapped_env.action_masks()
        else:
            # Fallback to allowing all actions
            return np.ones(unwrapped_env.action_space.n, dtype=bool)
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def make_env(graph_path: str, battery_init: int, payload_init: int, rank: int = 0,
             randomize_battery: bool = False, battery_range: tuple = (60, 100),
             randomize_payload: bool = False, payload_range: tuple = (1, 5),
             use_masking: bool = False):
    """Create environment factory for vectorization"""
    def _init():
        env = DroneDeliveryFullEnv(
            graph_path=graph_path,
            battery_init=battery_init,
            payload_init=payload_init,
            randomize_battery=randomize_battery,
            battery_range=battery_range,
            randomize_payload=randomize_payload,
            payload_range=payload_range
        )
        
        # Set seed on the base environment first
        env.seed(rank)
        
        # Add action masking BEFORE monitoring if requested and available
        if use_masking and MASKABLE_PPO_AVAILABLE:
            wrapper = ActionMaskerWrapper(env)
            env = ActionMasker(env, wrapper.mask_fn)
        
        # Add monitoring AFTER action masking
        env = Monitor(env)
        
        return env
    
    set_random_seed(rank)
    return _init


class TrainingMetricsCallback(BaseCallback):
    """Enhanced callback for comprehensive training metrics and logging"""
    
    def __init__(self, check_freq=1000, log_dir="./logs/", save_freq=10000, verbose=1):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_freq = save_freq
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Metrics tracking
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.success_rates = deque(maxlen=100)
        self.battery_usage = deque(maxlen=1000)
        self.recharge_counts = deque(maxlen=1000)
        self.invalid_actions = deque(maxlen=1000)
        
        # Training progress
        self.start_time = time.time()
        self.best_mean_reward = -float('inf')
        self.training_interrupted = False
        
        # Episode tracking for detailed analysis
        self.recent_episodes = deque(maxlen=50)
        self.action_distribution = {}
        
        # Set up signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\n🛑 Training interrupted by signal {signum}")
        self.training_interrupted = True
        self._save_training_summary()
        
    def _on_step(self) -> bool:
        # Check if training was interrupted
        if self.training_interrupted:
            return False
            
        # Collect episode information
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                
                # Extract additional metrics if available
                if 'success' in info:
                    self.success_rates.append(1.0 if info['success'] else 0.0)
                if 'battery_used' in info:
                    self.battery_usage.append(info['battery_used'])
                if 'recharge_count' in info:
                    self.recharge_counts.append(info['recharge_count'])
                if 'invalid_action_count' in info:
                    self.invalid_actions.append(info['invalid_action_count'])
                
                # Store recent episode for analysis
                self.recent_episodes.append({
                    'reward': ep_info['r'],
                    'length': ep_info['l'],
                    'timestep': self.num_timesteps,
                    'success': info.get('success', False),
                    'battery_used': info.get('battery_used', 0),
                    'recharge_count': info.get('recharge_count', 0),
                    'route_description': info.get('route_description', 'Unknown'),
                    'pickup_done': info.get('pickup_done', False),
                    'delivery_done': info.get('delivery_done', False),
                    'termination_reason': info.get('termination_reason', 'unknown')
                })
        
        # Log metrics every check_freq steps
        if self.num_timesteps % self.check_freq == 0:
            self._log_metrics()
        
        # Save detailed analysis every save_freq steps
        if self.num_timesteps % self.save_freq == 0:
            self._save_detailed_analysis()
        
        return True
    
    def _log_metrics(self):
        """Log current training metrics"""
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate metrics
        mean_reward = np.mean(self.episode_rewards)
        std_reward = np.std(self.episode_rewards)
        mean_length = np.mean(self.episode_lengths)
        success_rate = np.mean(self.success_rates) if self.success_rates else 0.0
        mean_battery = np.mean(self.battery_usage) if self.battery_usage else 0.0
        mean_recharges = np.mean(self.recharge_counts) if self.recharge_counts else 0.0
        mean_invalid = np.mean(self.invalid_actions) if self.invalid_actions else 0.0
        
        # Update best reward
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
        
        # Time information
        elapsed_time = time.time() - self.start_time
        steps_per_sec = self.num_timesteps / elapsed_time
        
        # Log to console
        print(f"\n📊 Training Metrics [Step {self.num_timesteps:,}]")
        print(f"  Reward: {mean_reward:.3f} ± {std_reward:.3f} (best: {self.best_mean_reward:.3f})")
        print(f"  Episode Length: {mean_length:.1f}")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Battery Usage: {mean_battery:.1f}")
        print(f"  Avg Recharges: {mean_recharges:.1f}")
        print(f"  Invalid Actions: {mean_invalid:.1f}")
        print(f"  Training Speed: {steps_per_sec:.1f} steps/sec")
        print(f"  Elapsed Time: {elapsed_time/3600:.1f}h")
        
        # Log to tensorboard
        if hasattr(self.training_env, 'get_attr'):
            self.logger.record('metrics/mean_reward', mean_reward)
            self.logger.record('metrics/std_reward', std_reward)
            self.logger.record('metrics/mean_episode_length', mean_length)
            self.logger.record('metrics/success_rate', success_rate)
            self.logger.record('metrics/mean_battery_usage', mean_battery)
            self.logger.record('metrics/mean_recharges', mean_recharges)
            self.logger.record('metrics/mean_invalid_actions', mean_invalid)
            self.logger.record('metrics/steps_per_second', steps_per_sec)
    
    def _save_detailed_analysis(self):
        """Save detailed training analysis"""
        if not self.recent_episodes:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = os.path.join(self.log_dir, f"training_analysis_{timestamp}.json")
        
        # Analyze recent episodes
        successful_episodes = [ep for ep in self.recent_episodes if ep['success']]
        failed_episodes = [ep for ep in self.recent_episodes if not ep['success']]
        
        # Analyze failure patterns
        failure_analysis = self._analyze_failure_patterns(failed_episodes)
        
        analysis = {
            'timestamp': timestamp,
            'total_timesteps': self.num_timesteps,
            'recent_episodes_count': len(self.recent_episodes),
            'success_rate': len(successful_episodes) / len(self.recent_episodes),
            'successful_episodes': {
                'count': len(successful_episodes),
                'avg_reward': np.mean([ep['reward'] for ep in successful_episodes]) if successful_episodes else 0,
                'avg_length': np.mean([ep['length'] for ep in successful_episodes]) if successful_episodes else 0,
                'avg_battery_used': np.mean([ep['battery_used'] for ep in successful_episodes]) if successful_episodes else 0,
                'best_routes': sorted(successful_episodes, key=lambda x: x['reward'], reverse=True)[:5]
            },
            'failed_episodes': {
                'count': len(failed_episodes),
                'avg_reward': np.mean([ep['reward'] for ep in failed_episodes]) if failed_episodes else 0,
                'avg_length': np.mean([ep['length'] for ep in failed_episodes]) if failed_episodes else 0,
                'failure_patterns': failure_analysis
            },
            'training_metrics': {
                'best_mean_reward': self.best_mean_reward,
                'current_mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'training_stability': np.std(list(self.episode_rewards)[-100:]) if len(self.episode_rewards) >= 100 else float('inf')
            }
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        if self.verbose > 0:
            print(f"💾 Detailed analysis saved to {analysis_file}")
    
    def _analyze_failure_patterns(self, failed_episodes):
        """Analyze common patterns in failed episodes"""
        if not failed_episodes:
            return {}
        
        # Count termination reasons
        termination_counts = {}
        for ep in failed_episodes:
            reason = ep.get('termination_reason', 'unknown')
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
        
        # Analyze episode characteristics
        short_episodes = sum(1 for ep in failed_episodes if ep['length'] < 10)
        high_battery_usage = sum(1 for ep in failed_episodes if ep['battery_used'] > 80)
        many_recharges = sum(1 for ep in failed_episodes if ep['recharge_count'] > 3)
        no_pickup = sum(1 for ep in failed_episodes if not ep.get('pickup_done', False))
        
        return {
            'termination_reasons': termination_counts,
            'patterns': {
                'short_episodes': short_episodes,
                'high_battery_usage': high_battery_usage,
                'many_recharges': many_recharges,
                'no_pickup_achieved': no_pickup,
                'total_failures': len(failed_episodes)
            }
        }
    
    def _save_training_summary(self):
        """Save final training summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.log_dir, f"training_summary_{timestamp}.json")
        
        summary = {
            'training_completed': not self.training_interrupted,
            'total_timesteps': self.num_timesteps,
            'training_time_hours': (time.time() - self.start_time) / 3600,
            'best_mean_reward': self.best_mean_reward,
            'final_metrics': {
                'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
                'success_rate': np.mean(self.success_rates) if self.success_rates else 0,
                'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
            },
            'total_episodes': len(self.episode_rewards)
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Training summary saved to {summary_file}")


class NewBestCallback(BaseCallback):
    """Simple callback to notify when a new best model is saved"""
    def __init__(self, verbose=1):
        super(NewBestCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        return True
    
    def _on_event(self) -> bool:
        if self.verbose > 0:
            print("🏆 New best model saved!")
        return True


def train_drone_delivery(graph_path: str, battery_init: int = 100, payload_init: int = 1,
                        total_timesteps: int = 1000000, n_envs: int = 8,
                        save_path: str = "models/drone_ppo_masked",
                        randomize_battery: bool = False, battery_range: tuple = (60, 100),
                        randomize_payload: bool = False, payload_range: tuple = (1, 5),
                        use_masking: bool = True):
    """Train PPO/MaskablePPO agent with comprehensive logging"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{save_path}_{timestamp}"
    
    print("🚁 Starting Enhanced Drone Delivery Training")
    print("=" * 60)
    print(f"Algorithm: {'MaskablePPO' if use_masking and MASKABLE_PPO_AVAILABLE else 'PPO'}")
    print(f"Graph: {graph_path}")
    print(f"Battery: {battery_init} {'(randomized ' + str(battery_range) + ')' if randomize_battery else ''}")
    print(f"Payload: {payload_init} {'(randomized ' + str(payload_range) + ')' if randomize_payload else ''}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Action masking: {use_masking and MASKABLE_PPO_AVAILABLE}")
    print(f"Run name: {run_name}")
    print("=" * 60)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create vectorized environment
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(graph_path, battery_init, payload_init, i,
                    randomize_battery, battery_range,
                    randomize_payload, payload_range, use_masking) 
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(graph_path, battery_init, payload_init, 0,
                    randomize_battery, battery_range,
                    randomize_payload, payload_range, use_masking)
        ])
    
    # Create evaluation environment
    eval_env = DroneDeliveryFullEnv(
        graph_path=graph_path,
        battery_init=battery_init,
        payload_init=payload_init,
        randomize_battery=randomize_battery,
        battery_range=battery_range,
        randomize_payload=randomize_payload,
        payload_range=payload_range
    )
    eval_env = Monitor(eval_env)
    
    # Choose algorithm and hyperparameters
    if use_masking and MASKABLE_PPO_AVAILABLE:
        # MaskablePPO with optimized hyperparameters
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"./logs/{run_name}/"
        )
    else:
        # Regular PPO
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"./logs/{run_name}/"
        )
    
    # Set up callbacks
    metrics_callback = TrainingMetricsCallback(
        check_freq=1000,
        log_dir=f"./logs/{run_name}/",
        save_freq=10000,
        verbose=1
    )
    
    # Checkpoint callback - save every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix=f"{run_name}_checkpoint"
    )
    
    # Create new best notification callback
    new_best_callback = NewBestCallback(verbose=1)
    
    # Evaluation callback - save best model (fixed callback configuration)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path=f"./logs/{run_name}/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    callbacks = [metrics_callback, checkpoint_callback, eval_callback]
    
    try:
        print(f"\n🎯 Starting training for {total_timesteps:,} timesteps...")
        start_time = time.time()
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/3600:.2f} hours")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Save final model
        final_model_path = f"./models/{run_name}_final.zip"
        model.save(final_model_path)
        print(f"💾 Final model saved to {final_model_path}")
        
        # Save training metadata
        metadata = {
            'algorithm': 'MaskablePPO' if use_masking and MASKABLE_PPO_AVAILABLE else 'PPO',
            'graph_path': graph_path,
            'battery_init': battery_init,
            'payload_init': payload_init,
            'total_timesteps': total_timesteps,
            'n_envs': n_envs,
            'randomize_battery': randomize_battery,
            'battery_range': battery_range,
            'randomize_payload': randomize_payload,
            'payload_range': payload_range,
            'use_masking': use_masking and MASKABLE_PPO_AVAILABLE,
            'training_time_hours': (time.time() - start_time) / 3600 if 'start_time' in locals() else 0,
            'timestamp': timestamp,
            'hyperparameters': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 256,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
            }
        }
        
        metadata_path = f"./models/{run_name}_final_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Training metadata saved to {metadata_path}")
        
        # Cleanup
        env.close()
        eval_env.close()
    
    return model, run_name


def main():
    parser = argparse.ArgumentParser(description="Train PPO/MaskablePPO for drone delivery with enhanced logging")
    parser.add_argument("--graph", type=str, required=True, help="Path to graph JSON file")
    parser.add_argument("--battery", type=int, default=100, help="Initial battery level")
    parser.add_argument("--payload", type=int, default=1, help="Initial payload")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--save-path", type=str, default="models/drone_ppo_enhanced", help="Model save path prefix")
    parser.add_argument("--randomize-battery", action="store_true", help="Randomize initial battery")
    parser.add_argument("--battery-range", type=int, nargs=2, default=[60, 100], help="Battery randomization range")
    parser.add_argument("--battery-min", type=int, help="Minimum battery level (alternative to --battery-range)")
    parser.add_argument("--battery-max", type=int, help="Maximum battery level (alternative to --battery-range)")
    parser.add_argument("--randomize-payload", action="store_true", help="Randomize initial payload")
    parser.add_argument("--payload-range", type=int, nargs=2, default=[1, 5], help="Payload randomization range")
    parser.add_argument("--payload-min", type=int, help="Minimum payload (alternative to --payload-range)")
    parser.add_argument("--payload-max", type=int, help="Maximum payload (alternative to --payload-range)")
    parser.add_argument("--no-masking", action="store_true", help="Disable action masking (use regular PPO)")
    parser.add_argument("--save", type=str, help="Alternative save path (shorter alias)")
    
    args = parser.parse_args()
    
    # Handle alternative argument formats
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
    
    # Handle alternative save path
    save_path = args.save if args.save else args.save_path
    
    # Validate files exist
    if not os.path.exists(args.graph):
        print(f"❌ Graph file not found: {args.graph}")
        return
    
    # Convert to absolute paths
    graph_path = os.path.abspath(args.graph)
    
    # Train model
    model, run_name = train_drone_delivery(
        graph_path=graph_path,
        battery_init=args.battery,
        payload_init=args.payload,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        save_path=save_path,
        randomize_battery=args.randomize_battery,
        battery_range=tuple(battery_range),
        randomize_payload=args.randomize_payload,
        payload_range=tuple(payload_range),
        use_masking=not args.no_masking
    )
    
    print(f"\n🎉 Training session '{run_name}' completed!")
    print(f"📂 Check ./logs/{run_name}/ for detailed training logs")
    print(f"📂 Check ./models/ for saved models")


if __name__ == "__main__":
    main()
