import os
import sys
import argparse
import signal
import json
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import DroneDeliveryFullEnv


def make_env(graph_path: str, battery_init: int, payload_init: int, rank: int = 0,
             randomize_battery: bool = False, battery_range: tuple = (60, 100),
             randomize_payload: bool = False, payload_range: tuple = (1, 5)):
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
        # Seed the environment
        env.seed(rank)
        env = Monitor(env)
        return env
    return _init


class ProgressCallback(BaseCallback):
    """Enhanced callback for training progress with Ctrl-C handling"""
    
    def __init__(self, check_freq=1000, log_dir="./logs/", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.interrupted = False
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\n🛑 Training interrupted! Saving model...")
        self.interrupted = True
        
    def _on_step(self) -> bool:
        # Check for interruption
        if self.interrupted:
            print("💾 Saving model due to interruption...")
            return False  # Stop training
            
        if self.num_timesteps % self.check_freq == 0:
            print(f"📈 Step {self.num_timesteps}: Training in progress...")
            
        return True


def train_ppo_full(graph_path: str, battery_init: int = 100, payload_init: int = 1,
                   total_timesteps: int = 1000000, n_envs: int = 8,
                   save_path: str = "models/drone_ppo_full",
                   randomize_battery: bool = False, battery_range: tuple = (60, 100),
                   randomize_payload: bool = False, payload_range: tuple = (1, 5)):
    """Train PPO agent on full drone delivery environment"""
    
    print("🚁 Starting Full PPO Training")
    print("=" * 50)
    print(f"Graph: {graph_path}")
    print(f"Battery init: {battery_init} {'(randomized ' + str(battery_range) + ')' if randomize_battery else ''}")
    print(f"Payload init: {payload_init} {'(randomized ' + str(payload_range) + ')' if randomize_payload else ''}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print("=" * 50)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create vectorized environment
    env = SubprocVecEnv([
        make_env(graph_path, battery_init, payload_init, i,
                randomize_battery, battery_range,
                randomize_payload, payload_range) 
        for i in range(n_envs)
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
    
    # PPO hyperparameters optimized for this problem
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
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    # Set up callbacks
    progress_callback = ProgressCallback(check_freq=1000)
    
    # Checkpoint callback - save every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix="drone_ppo_full_checkpoint"
    )
    
    # Evaluation callback - save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    callbacks = [progress_callback, checkpoint_callback, eval_callback]
    
    try:
        # Train the model
        print("🎯 Starting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=f"PPO_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always save the model
        final_path = f"{save_path}_final"
        print(f"💾 Saving final model to {final_path}")
        try:
            model.save(final_path)
            print("✅ Model saved successfully!")
              # Save training metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "total_timesteps": total_timesteps,
                "battery_init": battery_init,
                "payload_init": payload_init,
                "randomize_battery": randomize_battery,
                "battery_range": battery_range,
                "randomize_payload": randomize_payload,
                "payload_range": payload_range,
                "n_envs": n_envs,
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "n_steps": 2048,
                    "batch_size": 256,
                    "gamma": 0.99,
                    "ent_coef": 0.01
                }
            }
            
            with open(f"{final_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
        
        finally:
            env.close()
            eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for full drone delivery")
    parser.add_argument("--graph", type=str, required=True, help="Path to graph JSON file")
    parser.add_argument("--battery", type=int, default=100, help="Initial battery level (80-100)")
    parser.add_argument("--payload", type=int, default=1, help="Initial payload (1-3)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--save", type=str, default="models/drone_ppo_full", help="Save path prefix")
    
    # Randomization parameters
    parser.add_argument("--randomize-battery", action="store_true", help="Randomize battery level each episode")
    parser.add_argument("--battery-min", type=int, default=60, help="Minimum battery level (default: 60)")
    parser.add_argument("--battery-max", type=int, default=100, help="Maximum battery level (default: 100)")
    parser.add_argument("--randomize-payload", action="store_true", help="Randomize payload each episode")
    parser.add_argument("--payload-min", type=int, default=1, help="Minimum payload (default: 1)")
    parser.add_argument("--payload-max", type=int, default=5, help="Maximum payload (default: 5)")
    
    args = parser.parse_args()
      # Validate arguments
    if not os.path.exists(args.graph):
        print(f"❌ Graph file not found: {args.graph}")
        sys.exit(1)
        
    if not (60 <= args.battery <= 100):
        print(f"❌ Battery must be between 60-100, got {args.battery}")
        sys.exit(1)
        
    if not (1 <= args.payload <= 5):
        print(f"❌ Payload must be between 1-5, got {args.payload}")
        sys.exit(1)
    
    # Validate randomization ranges
    if args.randomize_battery:
        if not (60 <= args.battery_min <= args.battery_max <= 100):
            print(f"❌ Invalid battery range: {args.battery_min}-{args.battery_max}")
            sys.exit(1)
    
    if args.randomize_payload:
        if not (1 <= args.payload_min <= args.payload_max <= 5):
            print(f"❌ Invalid payload range: {args.payload_min}-{args.payload_max}")
            sys.exit(1)
    
    # Convert to absolute path
    graph_path = os.path.abspath(args.graph)
    
    # Start training
    model = train_ppo_full(
        graph_path=graph_path,
        battery_init=args.battery,
        payload_init=args.payload,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        save_path=args.save,
        randomize_battery=args.randomize_battery,
        battery_range=(args.battery_min, args.battery_max),
        randomize_payload=args.randomize_payload,
        payload_range=(args.payload_min, args.payload_max)
    )
    
    print("\n🎉 Training completed!")


if __name__ == "__main__":
    main()
