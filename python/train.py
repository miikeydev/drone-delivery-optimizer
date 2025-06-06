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
             use_masking: bool = False, use_curriculum: bool = True,
             use_node_embedding: bool = True, embedding_dim: int = 32):
    """Create environment factory for vectorization - FIXED reset format"""
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
            curriculum_threshold=0.25,
            use_node_embedding=use_node_embedding,
            embedding_dim=embedding_dim
        )
        
        # Set seed BEFORE any wrapping
        if hasattr(env, 'seed'):
            env.seed(rank)
        
        # Apply action masking wrapper if needed
        if use_masking and MASKABLE_PPO_AVAILABLE:
            def mask_fn(env_instance):
                return env_instance.action_masks()
            env = ActionMasker(env, mask_fn)
        
        # Apply monitoring wrapper
        env = Monitor(env)
        
        return env
    
    # Set random seed for this worker
    set_random_seed(rank)
    return _init


class TrainingMetricsCallback(BaseCallback):
    """Enhanced callback for comprehensive training metrics and logging"""
    
    def __init__(self, check_freq=1000, log_dir="./logs/", save_freq=100000, verbose=1):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_freq = save_freq
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Metrics tracking with higher expected rewards
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.success_rates = deque(maxlen=100)
        self.battery_usage = deque(maxlen=1000)
        self.recharge_counts = deque(maxlen=1000)
        
        # Training progress with higher baseline
        self.start_time = time.time()
        self.best_mean_reward = -float('inf')
        self.training_interrupted = False
        
        # Episode tracking for detailed analysis
        self.recent_episodes = deque(maxlen=100)
        
        # Set up signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\n🛑 Training interrupted by signal {signum}")
        self.training_interrupted = True
        
    def _on_step(self) -> bool:
        # Check if training was interrupted
        if self.training_interrupted:
            return False
            
        # Collect episode information SILENTLY
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                
                # Extract additional metrics if available - FIXED to use NET battery
                if 'success' in info:
                    self.success_rates.append(1.0 if info['success'] else 0.0)
                if 'battery_used_net' in info:
                    self.battery_usage.append(info['battery_used_net'])
                if 'recharge_count' in info:
                    self.recharge_counts.append(info['recharge_count'])
                
                # Store episode data silently
                episode_data = {
                    'reward': ep_info['r'],
                    'length': ep_info['l'],
                    'timestep': self.num_timesteps,
                    'success': info.get('success', False),
                    'battery_used': info.get('battery_used_net', 0),
                    'recharge_count': info.get('recharge_count', 0),
                    'pickup_done': info.get('pickup_done', False),
                    'delivery_done': info.get('delivery_done', False),
                    'termination_reason': info.get('termination_reason', 'unknown'),
                    'curriculum_level': info.get('curriculum_level', 0),
                    'path': info.get('episode_path', []),
                    'route_description': info.get('route_description', 'Unknown')
                }
                self.recent_episodes.append(episode_data)
        
        # ONLY log detailed stats every 100k steps
        if self.num_timesteps % self.save_freq == 0:
            self._log_detailed_stats()
        
        return True
    
    def _calculate_node_revisits(self, path):
        """Calculate average node revisits per episode"""
        if not path or len(path) <= 1:
            return 0.0, 0.0, 0.0
        
        node_counts = {}
        for node in path:
            node_counts[node] = node_counts.get(node, 0) + 1
        
        # Count how many nodes were visited multiple times
        revisited_nodes = sum(1 for count in node_counts.values() if count > 1)
        total_revisits = sum(max(0, count - 1) for count in node_counts.values())
        
        return total_revisits, revisited_nodes, len(node_counts)
    
    def _log_detailed_stats(self):
        """Log current training metrics - ONLY every 100k steps"""
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate metrics
        mean_reward = np.mean(self.episode_rewards)
        mean_length = np.mean(self.episode_lengths)
        success_rate = np.mean(self.success_rates) if self.success_rates else 0.0
        mean_battery = np.mean(self.battery_usage) if self.battery_usage else 0.0
        mean_recharges = np.mean(self.recharge_counts) if self.recharge_counts else 0.0
        
        # Mission progress rates
        pickup_success_rate = len([ep for ep in self.recent_episodes if ep.get('pickup_done', False)]) / max(len(self.recent_episodes), 1)
        delivery_success_rate = len([ep for ep in self.recent_episodes if ep.get('delivery_done', False)]) / max(len(self.recent_episodes), 1)
        
        # Curriculum metrics
        curriculum_levels = [ep.get('curriculum_level', 0) for ep in self.recent_episodes if 'curriculum_level' in ep]
        avg_curriculum_level = np.mean(curriculum_levels) if curriculum_levels else 0
        
        # Update best reward
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
        
        # Time information
        elapsed_time = time.time() - self.start_time
        steps_per_sec = self.num_timesteps / elapsed_time
        
        # SIMPLIFIED OUTPUT - just the essential metrics
        print(f"\n🎯 TRAINING CHECKPOINT - Step {self.num_timesteps:,}")
        print(f"| mean_reward             | {mean_reward:.1f}     |")
        print(f"| success_rate            | {success_rate:.1%}   |")
        print(f"| pickup_rate             | {pickup_success_rate:.1%}   |")
        print(f"| delivery_rate           | {delivery_success_rate:.1%}   |")
        print(f"| curriculum_level        | {avg_curriculum_level:.1f}     |")
        print(f"| training_speed          | {steps_per_sec:.0f}/s   |")
        print(f"| avg_episode_length      | {mean_length:.0f}     |")
        print(f"| avg_battery_used        | {mean_battery:.1f}%    |")
        
        # Show best recent episode if available
        if self.recent_episodes:
            best_episode = max(self.recent_episodes, key=lambda x: x['reward'])
            print(f"| best_recent_reward      | {best_episode['reward']:.1f}     |")
            if best_episode.get('success', False):
                print(f"| best_route              | SUCCESS |")
            else:
                print(f"| best_route              | FAILED  |")
        
        print("=" * 40)
    
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


# Add GNN imports after existing imports
try:
    from gnn_policies import GNNPolicy, MaskableGNNPolicy, SimpleGNNPolicy, MaskableSimpleGNNPolicy
    GNN_AVAILABLE = True
    print("✅ GNN policies available - can use Graph Neural Networks")
except ImportError as e:
    GNN_AVAILABLE = False
    print(f"⚠️ GNN policies not available: {e}")
    print("   Install torch_geometric: pip install torch_geometric")


def train_drone_delivery(graph_path: str, battery_init: int = 100, payload_init: int = 1,
                        total_timesteps: int = 1000000, n_envs: int = 8,
                        save_path: str = "models/drone_ppo_masked",
                        randomize_battery: bool = False, battery_range: tuple = (60, 100),
                        randomize_payload: bool = False, payload_range: tuple = (1, 5),
                        use_masking: bool = True, use_curriculum: bool = True,
                        use_node_embedding: bool = True, embedding_dim: int = 32,
                        use_gnn: bool = False, gnn_simple: bool = False):  # NEW GNN params
    """Train PPO/MaskablePPO agent with optional GNN backbone"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{save_path}_{timestamp}"
    
    print("Drone Delivery Training - Enhanced with GNN Option")
    print("=" * 60)
    algorithm_name = ""
    if use_gnn and GNN_AVAILABLE:
        algorithm_name = f"{'MaskablePPO' if use_masking else 'PPO'} + {'Simple' if gnn_simple else 'Full'}GNN"
    else:
        algorithm_name = f"{'MaskablePPO' if use_masking and MASKABLE_PPO_AVAILABLE else 'PPO'}"
    
    print(f"Algorithm: {algorithm_name}")
    print(f"Graph: {graph_path}")
    print(f"Battery: {battery_init}, Payload: {payload_init}")
    print(f"Curriculum: {'Enabled' if use_curriculum else 'Disabled'} (threshold=0.15)")
    print(f"GNN: {'Enabled (' + ('Simple' if gnn_simple else 'Full') + ')' if use_gnn else 'Disabled'}")
    print(f"Timesteps: {total_timesteps:,}, Envs: {n_envs}")
    print(f"Run: {run_name}")
    print("=" * 60)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # FIXED: Use DummyVecEnv for Windows to avoid BrokenPipe issues
    print("🔧 Using DummyVecEnv to avoid Windows BrokenPipe issues...")
    env = DummyVecEnv([
        make_env(graph_path, battery_init, payload_init, i,
                randomize_battery, battery_range,
                randomize_payload, payload_range, use_masking, use_curriculum,
                use_node_embedding, embedding_dim)
        for i in range(n_envs)
    ])
    
    # Apply normalization wrapper
    from stable_baselines3.common.vec_env import VecNormalize
    env = VecNormalize(env,
                       norm_obs=True,
                       norm_reward=True,
                       clip_obs=5.0)
    
    # Create evaluation environment
    eval_env = DroneDeliveryFullEnv(
        graph_path=graph_path,
        battery_init=battery_init,
        payload_init=payload_init,
        randomize_battery=randomize_battery,
        battery_range=battery_range,
        randomize_payload=randomize_payload,
        payload_range=payload_range,
        use_curriculum=use_curriculum,
        curriculum_threshold=0.25,
        use_node_embedding=use_node_embedding,
        embedding_dim=embedding_dim
    )
    eval_env = Monitor(eval_env)
    
    # ENHANCED MODEL CREATION with GNN support
    if use_gnn and GNN_AVAILABLE:
        # GNN-based training with adjusted hyperparameters
        if use_masking and MASKABLE_PPO_AVAILABLE:
            if gnn_simple:
                policy_class = MaskableSimpleGNNPolicy
                print("🧠 Using MaskablePPO + Simple GNN")
            else:
                policy_class = MaskableGNNPolicy
                print("🧠 Using MaskablePPO + Full GNN")
            
            model = MaskablePPO(
                policy_class,
                env,
                learning_rate=1e-4,        # LOWERED for GNN stability
                n_steps=n_envs * 2048,     # INCREASED batch diversity
                batch_size=1024,           # INCREASED for GNN variance reduction
                n_epochs=5,                # REDUCED to prevent overfitting
                gamma=0.99,                # STANDARD discount
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,            # BALANCED exploration
                vf_coef=0.5,              # INCREASED value function weight
                max_grad_norm=0.5,
                target_kl=0.02,           # TIGHTER KL constraint
                verbose=0,
                tensorboard_log=f"./logs/{run_name}/",
                policy_kwargs=dict(
                    gnn_hidden_dim=64 if not gnn_simple else 32,
                    gnn_heads=4 if not gnn_simple else 2,
                    gnn_layers=2 if not gnn_simple else 1,
                    gnn_aggregation="mean",
                    gnn_features_dim=128,
                    net_arch=[256, 128],      # SMALLER traditional network
                    activation_fn=torch.nn.ReLU,
                    ortho_init=False
                )
            )
        else:
            if gnn_simple:
                policy_class = SimpleGNNPolicy
                print("🧠 Using PPO + Simple GNN")
            else:
                policy_class = GNNPolicy
                print("🧠 Using PPO + Full GNN")
            
            model = PPO(
                policy_class,
                env,
                learning_rate=1e-4,
                n_steps=n_envs * 2048,
                batch_size=1024,
                n_epochs=5,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.02,
                verbose=0,
                tensorboard_log=f"./logs/{run_name}/",
                policy_kwargs=dict(
                    gnn_hidden_dim=64 if not gnn_simple else 32,
                    gnn_heads=4 if not gnn_simple else 2,
                    gnn_layers=2 if not gnn_simple else 1,
                    gnn_aggregation="mean",
                    gnn_features_dim=128,
                    net_arch=[256, 128],
                    activation_fn=torch.nn.ReLU,
                    ortho_init=False
                )
            )
        
        print(f"🧠 GNN HYPERPARAMS: lr=1e-4, batch=1024, ent_coef=0.01")
        print(f"🎯 GNN CONFIG: hidden={64 if not gnn_simple else 32}, heads={4 if not gnn_simple else 2}")
        
    else:
        # REBALANCED HYPERPARAMETERS for standard training
        if use_masking and MASKABLE_PPO_AVAILABLE:
            net_arch = [512, 256, 128]
            
            # EXPLORATION BOOST - Higher entropy for better action space exploration
            model = MaskablePPO(
                "MlpPolicy",
                env,
                learning_rate=2e-4,
                n_steps=4096,
                batch_size=256,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.02,        # CHANGED: 0.005 → 0.02 (4x more exploration)
                vf_coef=0.4,
                max_grad_norm=0.5,
                target_kl=0.03,       # NEW: Early stopping for stable exploration
                verbose=0,
                tensorboard_log=f"./logs/{run_name}/",
                policy_kwargs=dict(
                    net_arch=net_arch,
                    activation_fn=torch.nn.ReLU,
                    ortho_init=False
                )
            )
            print(f"🧠 EXPLORATION BOOSTED: ent_coef=0.02, target_kl=0.03")
            print(f"🎯 Hyperparams: lr=2e-4, gamma=0.995, HIGH ENTROPY for K=10 exploration")
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=2e-4,
                n_steps=4096,
                batch_size=256,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.02,        # CHANGED: 0.005 → 0.02 (4x more exploration)
                vf_coef=0.4,
                max_grad_norm=0.5,
                target_kl=0.03,       # NEW: Early stopping for stable exploration
                verbose=0,
                tensorboard_log=f"./logs/{run_name}/",
                policy_kwargs=dict(
                    net_arch=[512, 256, 128],
                    activation_fn=torch.nn.ReLU,
                    ortho_init=False
                )
            )
            print(f"🧠 EXPLORATION BOOSTED: ent_coef=0.02, target_kl=0.03")
    
    # Set up callbacks
    metrics_callback = TrainingMetricsCallback(
        check_freq=1000,
        log_dir=f"./logs/{run_name}/",
        save_freq=100000,
        verbose=0
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix=f"{run_name}_checkpoint",
        verbose=0
    )
    
    new_best_callback = NewBestCallback(verbose=0)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path=f"./logs/{run_name}/",
        eval_freq=100000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=0
    )
    
    callbacks = [metrics_callback, checkpoint_callback, eval_callback]
    
    try:
        print(f"\n🎯 Starting REBALANCED training for {total_timesteps:,} timesteps...")
        start_time = time.time()
        
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
        # FIXED: Safer cleanup
        try:
            # Save final model and normalization stats
            final_model_path = f"./models/{run_name}_final.zip"
            model.save(final_model_path)
            
            # Save VecNormalize stats
            if hasattr(env, 'save'):
                env.save(f"./models/{run_name}_vecnormalize.pkl")
            
            print(f"💾 Final model saved to {final_model_path}")
            
            # Save training metadata
            metadata = {
                'algorithm': 'MaskablePPO' if use_masking and MASKABLE_PPO_AVAILABLE else 'PPO',
                'graph_path': graph_path,
                'battery_init': battery_init,
                'payload_init': payload_init,
                'total_timesteps': total_timesteps,
                'n_envs': n_envs,
                'rebalanced_rewards': {
                    'move_penalty': -0.001,
                    'recharge_penalty': 0.2,
                    'pickup_reward': 5.0,
                    'delivery_reward': 30.0
                },
                'rebalanced_hyperparameters': {
                    'learning_rate': 2e-4,
                    'n_steps': 4096,
                    'gamma': 0.995,
                    'ent_coef': 0.005,
                    'vf_coef': 0.4,
                    'net_arch': [512, 256, 128]
                },
                'curriculum_threshold': 0.25,
                'normalization': True,
                'timestamp': timestamp
            }
            
            metadata_path = f"./models/{run_name}_final_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"📋 Training metadata saved to {metadata_path}")
        
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup error: {cleanup_error}")
        
        # FIXED: Safe environment closing
        try:
            env.close()
        except:
            pass
        
        try:
            eval_env.close()
        except:
            pass
    
    return model, run_name


def main():
    parser = argparse.ArgumentParser(description="Train PPO/MaskablePPO with optional GNN backbone")
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
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--no-embedding", action="store_true", help="Disable node embedding (may cause aliasing)")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Node embedding dimension")
    parser.add_argument("--save", type=str, help="Alternative save path (shorter alias)")
    
    # NEW GNN arguments
    parser.add_argument("--use-gnn", action="store_true", help="Use Graph Neural Network backbone")
    parser.add_argument("--gnn-simple", action="store_true", help="Use simplified GNN (faster training)")
    parser.add_argument("--gnn-test", action="store_true", help="Quick GNN smoke test (500k steps)")
    parser.add_argument("--gnn-default", action="store_true", help="Use GNN by default (recommended)")
    
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
    
    # FORCE GNN usage by default if available and not explicitly disabled
    if not args.use_gnn and not args.gnn_test and GNN_AVAILABLE:
        if args.gnn_default:
            args.use_gnn = True
            args.gnn_simple = True  # Use simple GNN for faster training
            print("🧠 AUTO-ENABLED: Simple GNN (faster training, better performance)")
    
    # Adjust parameters for GNN test
    if args.gnn_test:
        args.use_gnn = True
        args.gnn_simple = True
        args.timesteps = 500000
        args.no_curriculum = True
        args.embedding_dim = 0  # Don't use tabulaire embedding with GNN
        print("🧪 GNN Smoke Test Mode: 500k steps, simple GNN, no curriculum")
    
    # Train model with optional GNN
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
        use_masking=not args.no_masking,
        use_curriculum=not args.no_curriculum,
        use_node_embedding=not args.no_embedding,
        embedding_dim=args.embedding_dim,
        use_gnn=args.use_gnn,              # NEW
        gnn_simple=args.gnn_simple         # NEW
    )
    
    print(f"\n🎉 Training session '{run_name}' completed!")
    print(f"📂 Check ./logs/{run_name}/ for detailed training logs")
    print(f"📂 Check ./models/ for saved models")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # ADDED: Windows BrokenPipe fix
    main()
