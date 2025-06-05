import os
import sys
import argparse
import json
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import DroneDeliveryFullEnv


def evaluate_ppo_model(model_path: str, graph_path: str, 
                      start_node: int = None, pickup_node: int = None, delivery_node: int = None,
                      battery_init: int = 100, payload_init: int = 1,
                      n_episodes: int = 5, render: bool = False):
    """
    Evaluate trained PPO model and return route and statistics
    
    Args:
        model_path: Path to trained PPO model
        graph_path: Path to graph JSON file
        start_node: Specific start hub (optional)
        pickup_node: Specific pickup location (optional)
        delivery_node: Specific delivery location (optional)
        battery_init: Initial battery level
        payload_init: Initial payload
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes
    
    Returns:
        dict: Results containing routes, statistics, and success metrics
    """
    
    print("🔍 Evaluating PPO Model")
    print("=" * 40)
    print(f"Model: {model_path}")
    print(f"Graph: {graph_path}")
    print(f"Episodes: {n_episodes}")
    
    # Load model
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return {"error": f"Failed to load model: {e}"}
    
    # Create environment
    try:
        env = DroneDeliveryFullEnv(
            graph_path=graph_path,
            battery_init=battery_init,
            payload_init=payload_init
        )
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return {"error": f"Failed to create environment: {e}"}
    
    # Override targets if specified
    if pickup_node is not None:
        env.pickups = [pickup_node]
    if delivery_node is not None:
        env.deliveries = [delivery_node]
    
    results = {
        "episodes": [],
        "statistics": {
            "success_rate": 0.0,
            "avg_total_cost": 0.0,
            "avg_steps": 0.0,
            "avg_recharges": 0.0,
            "avg_battery_final": 0.0
        },
        "best_route": None,
        "parameters": {
            "battery_init": battery_init,
            "payload_init": payload_init,
            "n_episodes": n_episodes
        }
    }
    
    successful_episodes = 0
    total_costs = []
    total_steps = []
    total_recharges = []
    final_batteries = []
    best_cost = float('inf')
    
    for episode in range(n_episodes):
        print(f"\n📍 Episode {episode + 1}/{n_episodes}")
        
        obs, info = env.reset()
        episode_data = {
            "episode": episode + 1,
            "route": [],
            "actions": [],
            "rewards": [],
            "costs": [],
            "battery_history": [env.battery],
            "success": False,
            "total_cost": 0.0,
            "total_steps": 0,
            "total_recharges": 0,
            "final_battery": 0.0,
            "termination_reason": ""
        }
        
        # Track specific nodes for forced routing
        if start_node is not None and start_node in env.hubs:
            # Force specific start hub
            start_action = env.hubs.index(start_node)
        else:
            start_action = None
        
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        
        while not (done or truncated) and step_count < 500:  # Safety limit
            # Get action from model
            if env.at_step_zero and start_action is not None:
                action = start_action
            else:
                action, _states = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            # Record step data
            episode_data["actions"].append(int(action))
            episode_data["rewards"].append(float(reward))
            episode_data["battery_history"].append(env.battery)
            episode_data["route"].append(env.current_node)
            
            if 'edge_cost' in info:
                episode_data["costs"].append(float(info['edge_cost']))
                episode_data["total_cost"] += info['edge_cost']
            
            total_reward += reward
            step_count += 1
            
            if render:
                env.render()
                print(f"  Action: {action}, Reward: {reward:.3f}, Info: {info}")
            
            if done or truncated:
                episode_data["success"] = info.get('success', False)
                episode_data["termination_reason"] = info.get('termination_reason', 'unknown')
                episode_data["total_steps"] = step_count
                episode_data["total_recharges"] = env.recharge_count
                episode_data["final_battery"] = env.battery
                
                if episode_data["success"]:
                    successful_episodes += 1
                    total_costs.append(episode_data["total_cost"])
                    total_steps.append(step_count)
                    total_recharges.append(env.recharge_count)
                    final_batteries.append(env.battery)
                    
                    # Track best route
                    if episode_data["total_cost"] < best_cost:
                        best_cost = episode_data["total_cost"]
                        results["best_route"] = episode_data.copy()
                
                print(f"  Result: {'✅ SUCCESS' if episode_data['success'] else '❌ FAILED'}")
                print(f"  Reason: {episode_data['termination_reason']}")
                print(f"  Steps: {step_count}, Cost: {episode_data['total_cost']:.1f}")
                print(f"  Battery: {env.battery:.1f}, Recharges: {env.recharge_count}")
        
        results["episodes"].append(episode_data)
    
    # Calculate statistics
    if successful_episodes > 0:
        results["statistics"] = {
            "success_rate": successful_episodes / n_episodes,
            "avg_total_cost": np.mean(total_costs),
            "avg_steps": np.mean(total_steps),
            "avg_recharges": np.mean(total_recharges),
            "avg_battery_final": np.mean(final_batteries)
        }
    
    print(f"\n📊 Evaluation Summary:")
    print(f"  Success Rate: {results['statistics']['success_rate']:.1%}")
    if successful_episodes > 0:
        print(f"  Avg Cost: {results['statistics']['avg_total_cost']:.1f}")
        print(f"  Avg Steps: {results['statistics']['avg_steps']:.1f}")
        print(f"  Avg Recharges: {results['statistics']['avg_recharges']:.1f}")
        print(f"  Best Cost: {best_cost:.1f}")
    
    return results


def random_evaluation(graph_path: str, n_episodes: int = 10):
    """Run random baseline evaluation"""
    
    print("🎲 Running Random Baseline Evaluation")
    print("=" * 40)
    
    # Create environment
    env = DroneDeliveryFullEnv(graph_path=graph_path)
    
    results = {
        "episodes": [],
        "statistics": {
            "success_rate": 0.0,
            "avg_total_cost": 0.0,
            "avg_steps": 0.0
        }
    }
    
    successful_episodes = 0
    total_costs = []
    total_steps = []
    
    for episode in range(n_episodes):
        print(f"\n📍 Random Episode {episode + 1}/{n_episodes}")
        
        obs, info = env.reset()
        done = False
        truncated = False
        step_count = 0
        total_cost = 0.0
        
        while not (done or truncated) and step_count < 1000:
            # Random action
            if env.at_step_zero:
                action = env.action_space.sample() % len(env.hubs)  # Random hub
            else:
                action = env.action_space.sample() % len(env.adjacency[env.current_node])  # Random edge
            
            obs, reward, done, truncated, info = env.step(action)
            
            if 'edge_cost' in info:
                total_cost += info['edge_cost']
            
            step_count += 1
        
        success = info.get('success', False)
        if success:
            successful_episodes += 1
            total_costs.append(total_cost)
            total_steps.append(step_count)
        
        episode_data = {
            "episode": episode + 1,
            "success": success,
            "total_cost": total_cost,
            "total_steps": step_count,
            "termination_reason": info.get('termination_reason', 'unknown')
        }
        
        results["episodes"].append(episode_data)
        print(f"  Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"  Steps: {step_count}, Cost: {total_cost:.1f}")
    
    # Calculate statistics
    if successful_episodes > 0:
        results["statistics"] = {
            "success_rate": successful_episodes / n_episodes,
            "avg_total_cost": np.mean(total_costs),
            "avg_steps": np.mean(total_steps)
        }
    
    print(f"\n📊 Random Baseline Summary:")
    print(f"  Success Rate: {results['statistics']['success_rate']:.1%}")
    if successful_episodes > 0:
        print(f"  Avg Cost: {results['statistics']['avg_total_cost']:.1f}")
        print(f"  Avg Steps: {results['statistics']['avg_steps']:.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model for drone delivery")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--graph", type=str, required=True, help="Path to graph JSON file")
    parser.add_argument("--start", type=int, help="Specific start hub node ID")
    parser.add_argument("--pickup", type=int, help="Specific pickup node ID")
    parser.add_argument("--delivery", type=int, help="Specific delivery node ID")
    parser.add_argument("--battery", type=int, default=100, help="Initial battery level")
    parser.add_argument("--payload", type=int, default=1, help="Initial payload")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--random-eval", action="store_true", help="Run random baseline")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Validate files exist
    for file_path in [args.model, args.graph]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
    
    # Convert to absolute paths
    model_path = os.path.abspath(args.model)
    graph_path = os.path.abspath(args.graph)
    
    if args.random_eval:
        # Run random evaluation
        random_results = random_evaluation(graph_path, args.episodes)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"random_baseline": random_results}, f, indent=2)
    else:
        # Run PPO evaluation
        results = evaluate_ppo_model(
            model_path=model_path,
            graph_path=graph_path,
            start_node=args.start,
            pickup_node=args.pickup,
            delivery_node=args.delivery,
            battery_init=args.battery,
            payload_init=args.payload,
            n_episodes=args.episodes,
            render=args.render
        )
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            # Print JSON to stdout for consumption by other tools
            print("\n" + "="*50)
            print("JSON OUTPUT:")
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
