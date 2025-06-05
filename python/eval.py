import os
import sys
import argparse
import json
import numpy as np
import time
from datetime import datetime
from collections import defaultdict
from stable_baselines3 import PPO
import gymnasium as gym

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import DroneDeliveryFullEnv

# Try to import MaskablePPO
try:
    from sb3_contrib import MaskablePPO
    MASKABLE_PPO_AVAILABLE = True
    print("✅ MaskablePPO available")
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    print("⚠️  MaskablePPO not available - install sb3-contrib for action masking support")


def load_model(model_path: str):
    """Load PPO or MaskablePPO model with automatic detection"""
    try:
        # First try to load as MaskablePPO
        if MASKABLE_PPO_AVAILABLE:
            try:
                model = MaskablePPO.load(model_path)
                print("✅ Loaded MaskablePPO model")
                return model, "MaskablePPO"
            except Exception:
                pass
        
        # Fallback to regular PPO
        model = PPO.load(model_path)
        print("✅ Loaded PPO model")
        return model, "PPO"
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")


def validate_action_masking(env, model, model_type: str, n_checks: int = 20):
    """Validate that model respects action masking"""
    print(f"\n🔍 Validating action masking for {model_type} model...")
    
    invalid_actions = 0
    total_checks = 0
    
    for _ in range(n_checks):
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 50:  # Short episodes for validation
            # Get valid actions
            valid_actions = env.action_masks() if hasattr(env, 'action_masks') else None
            
            # Get model prediction
            if model_type == "MaskablePPO" and valid_actions is not None:
                action, _states = model.predict(obs, action_masks=valid_actions, deterministic=True)
            else:
                action, _states = model.predict(obs, deterministic=True)
            
            # Check if action is valid
            if valid_actions is not None:
                total_checks += 1
                if not valid_actions[action]:
                    invalid_actions += 1
                    print(f"⚠️  Invalid action {action} chosen at step {step_count}")
            
            obs, _, done, _, _ = env.step(action)
            step_count += 1
    
    if total_checks > 0:
        invalid_rate = invalid_actions / total_checks
        print(f"📊 Action validation: {invalid_actions}/{total_checks} invalid ({invalid_rate:.1%})")
        return invalid_rate
    else:
        print("📊 No action masking validation available")
        return 0.0


def analyze_failure_patterns(episodes):
    """Analyze failure patterns in episodes"""
    failures = [ep for ep in episodes if not ep['success']]
    if not failures:
        return {"message": "No failures to analyze"}
    
    failure_reasons = defaultdict(int)
    failure_steps = []
    failure_batteries = []
    
    for ep in failures:
        reason = ep.get('termination_reason', 'unknown')
        failure_reasons[reason] += 1
        failure_steps.append(ep['total_steps'])
        failure_batteries.append(ep['final_battery'])
    
    return {
        "total_failures": len(failures),
        "failure_rate": len(failures) / len(episodes),
        "failure_reasons": dict(failure_reasons),
        "avg_failure_steps": np.mean(failure_steps) if failure_steps else 0,
        "avg_failure_battery": np.mean(failure_batteries) if failure_batteries else 0
    }


def generate_route_description(episode_data):
    """Generate human-readable route description"""
    if not episode_data.get('route'):
        return "No route recorded"
    
    route = episode_data['route']
    actions = episode_data.get('actions', [])
    
    description = f"Route: {' → '.join(map(str, route))}"
    description += f" ({len(route)} nodes, {episode_data['total_steps']} steps)"
    
    if episode_data['success']:
        description += f" ✅ SUCCESS (cost: {episode_data['total_cost']:.1f})"
    else:
        reason = episode_data.get('termination_reason', 'unknown')
        description += f" ❌ FAILED ({reason})"
    
    return description


def calculate_efficiency_metrics(episode_data):
    """Calculate efficiency metrics for an episode"""
    if not episode_data['success']:
        return {"efficiency": 0.0, "battery_efficiency": 0.0}
    
    total_cost = episode_data['total_cost']
    total_steps = episode_data['total_steps']
    final_battery = episode_data['final_battery']
    recharges = episode_data['total_recharges']
    
    # Simple efficiency metrics
    step_efficiency = 1.0 / max(total_steps, 1)  # Fewer steps = better
    cost_efficiency = 100.0 / max(total_cost, 1)  # Lower cost = better
    battery_efficiency = final_battery / 100.0  # More battery left = better
    recharge_penalty = max(0, 1.0 - recharges * 0.1)  # Fewer recharges = better
    
    overall_efficiency = (step_efficiency + cost_efficiency + battery_efficiency + recharge_penalty) / 4
    
    return {
        "efficiency": overall_efficiency,
        "step_efficiency": step_efficiency,
        "cost_efficiency": cost_efficiency,
        "battery_efficiency": battery_efficiency,    }


def evaluate_ppo_model(model_path: str, graph_path: str, 
                      start_node: int = None, pickup_node: int = None, delivery_node: int = None,
                      battery_init: int = 100, payload_init: int = 1,
                      n_episodes: int = 5, render: bool = False, validate_masking: bool = True):
    """
    Evaluate trained PPO/MaskablePPO model and return comprehensive route analysis
    
    Args:
        model_path: Path to trained model
        graph_path: Path to graph JSON file
        start_node: Specific start hub (optional)
        pickup_node: Specific pickup location (optional)
        delivery_node: Specific delivery location (optional)
        battery_init: Initial battery level
        payload_init: Initial payload
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes
        validate_masking: Whether to validate action masking
    
    Returns:
        dict: Comprehensive results with routes, statistics, and analysis
    """
    
    print("🔍 Evaluating Model")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Graph: {graph_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Battery: {battery_init}, Payload: {payload_init}")
    
    # Load model with automatic detection
    try:
        model, model_type = load_model(model_path)
        print(f"✅ {model_type} model loaded successfully")
    except Exception as e:
        print(f"❌ {e}")
        return {"error": str(e)}
    
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
    
    # Validate action masking if requested
    masking_validation = {}
    if validate_masking:
        try:
            invalid_rate = validate_action_masking(env, model, model_type)
            masking_validation = {
                "invalid_action_rate": invalid_rate,
                "supports_masking": model_type == "MaskablePPO"
            }
        except Exception as e:
            print(f"⚠️  Action masking validation failed: {e}")
            masking_validation = {"error": str(e)}
    
    # Override targets if specified
    if pickup_node is not None:
        env.pickups = [pickup_node]
    if delivery_node is not None:
        env.deliveries = [delivery_node]
    
    # Initialize results structure
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "model_type": model_type,
            "graph_path": graph_path,
            "parameters": {
                "battery_init": battery_init,
                "payload_init": payload_init,
                "n_episodes": n_episodes,
                "start_node": start_node,
                "pickup_node": pickup_node,
                "delivery_node": delivery_node
            }
        },
        "action_masking_validation": masking_validation,
        "episodes": [],
        "statistics": {},
        "analysis": {},
        "best_episode": None,
        "worst_episode": None
    }
    
    # Episode tracking variables
    successful_episodes = 0
    episode_metrics = {
        "total_costs": [],
        "total_steps": [],
        "total_recharges": [],
        "final_batteries": [],
        "efficiencies": [],
        "route_lengths": []
    }
    
    best_cost = float('inf')
    worst_cost = -float('inf')
    
    print(f"\n🚁 Running {n_episodes} evaluation episodes...")
    start_time = time.time()
    
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
            "termination_reason": "",
            "route_description": "",
            "efficiency_metrics": {}
        }
        
        # Track specific nodes for forced routing
        if start_node is not None and start_node in env.hubs:
            start_action = env.hubs.index(start_node)
        else:
            start_action = None
        
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        
        while not (done or truncated) and step_count < 1000:  # Increased safety limit
            # Get action from model
            if env.at_step_zero and start_action is not None:
                action = start_action
            else:
                # Use action masking for MaskablePPO
                if model_type == "MaskablePPO" and hasattr(env, 'action_masks'):
                    action_masks = env.action_masks()
                    action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
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
                print(f"    Step {step_count}: Action {action}, Reward {reward:.3f}, Battery {env.battery:.1f}")
                if 'route_description' in info:
                    print(f"    {info['route_description']}")
            
            if done or truncated:
                # Finalize episode data
                episode_data["success"] = info.get('success', False)
                episode_data["termination_reason"] = info.get('termination_reason', 'unknown')
                episode_data["total_steps"] = step_count
                episode_data["total_recharges"] = env.recharge_count
                episode_data["final_battery"] = env.battery
                episode_data["route_description"] = generate_route_description(episode_data)
                episode_data["efficiency_metrics"] = calculate_efficiency_metrics(episode_data)
                
                # Track statistics
                if episode_data["success"]:
                    successful_episodes += 1
                    episode_metrics["total_costs"].append(episode_data["total_cost"])
                    episode_metrics["total_steps"].append(step_count)
                    episode_metrics["total_recharges"].append(env.recharge_count)
                    episode_metrics["final_batteries"].append(env.battery)
                    episode_metrics["efficiencies"].append(episode_data["efficiency_metrics"]["efficiency"])
                    episode_metrics["route_lengths"].append(len(episode_data["route"]))
                    
                    # Track best and worst episodes
                    if episode_data["total_cost"] < best_cost:
                        best_cost = episode_data["total_cost"]
                        results["best_episode"] = episode_data.copy()
                    
                    if episode_data["total_cost"] > worst_cost:
                        worst_cost = episode_data["total_cost"]
                        results["worst_episode"] = episode_data.copy()
                
                # Print episode summary
                status = '✅ SUCCESS' if episode_data['success'] else '❌ FAILED'
                print(f"  Result: {status}")
                print(f"  Reason: {episode_data['termination_reason']}")
                print(f"  Steps: {step_count}, Cost: {episode_data['total_cost']:.1f}")
                print(f"  Battery: {env.battery:.1f}, Recharges: {env.recharge_count}")
                if episode_data['success']:
                    eff = episode_data["efficiency_metrics"]["efficiency"]
                    print(f"  Efficiency: {eff:.3f}")
                break
        
        results["episodes"].append(episode_data)
    
    evaluation_time = time.time() - start_time
    
    # Calculate comprehensive statistics
    if successful_episodes > 0:
        stats = {
            "success_rate": successful_episodes / n_episodes,
            "successful_episodes": successful_episodes,
            "total_episodes": n_episodes,
            "avg_total_cost": np.mean(episode_metrics["total_costs"]),
            "std_total_cost": np.std(episode_metrics["total_costs"]),
            "min_total_cost": np.min(episode_metrics["total_costs"]),
            "max_total_cost": np.max(episode_metrics["total_costs"]),
            "avg_steps": np.mean(episode_metrics["total_steps"]),
            "avg_recharges": np.mean(episode_metrics["total_recharges"]),
            "avg_battery_final": np.mean(episode_metrics["final_batteries"]),
            "avg_efficiency": np.mean(episode_metrics["efficiencies"]),
            "avg_route_length": np.mean(episode_metrics["route_lengths"]),
            "evaluation_time_seconds": evaluation_time
        }
    else:
        stats = {
            "success_rate": 0.0,
            "successful_episodes": 0,
            "total_episodes": n_episodes,
            "evaluation_time_seconds": evaluation_time
        }
    
    results["statistics"] = stats
    
    # Add failure analysis
    results["analysis"] = {
        "failure_patterns": analyze_failure_patterns(results["episodes"]),
        "performance_summary": {
            "best_cost": best_cost if best_cost != float('inf') else None,
            "worst_cost": worst_cost if worst_cost != -float('inf') else None,
            "cost_range": (best_cost, worst_cost) if best_cost != float('inf') else None
        }
    }
    
    # Print comprehensive summary
    print(f"\n📊 Evaluation Complete!")
    print("=" * 50)
    print(f"Model Type: {model_type}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Evaluation Time: {evaluation_time:.2f}s")
    
    if masking_validation and 'invalid_action_rate' in masking_validation:
        print(f"Invalid Action Rate: {masking_validation['invalid_action_rate']:.1%}")
    
    if successful_episodes > 0:
        print(f"Average Cost: {stats['avg_total_cost']:.1f} ± {stats['std_total_cost']:.1f}")
        print(f"Cost Range: {stats['min_total_cost']:.1f} - {stats['max_total_cost']:.1f}")
        print(f"Average Steps: {stats['avg_steps']:.1f}")
        print(f"Average Recharges: {stats['avg_recharges']:.1f}")
        print(f"Average Efficiency: {stats['avg_efficiency']:.3f}")
    
    # Print failure analysis
    failure_analysis = results["analysis"]["failure_patterns"]
    if failure_analysis.get("total_failures", 0) > 0:
        print(f"\n❌ Failure Analysis:")
        print(f"  Total Failures: {failure_analysis['total_failures']}")
        print(f"  Failure Rate: {failure_analysis['failure_rate']:.1%}")
        print(f"  Failure Reasons: {failure_analysis['failure_reasons']}")
    
    return results


def random_evaluation(graph_path: str, n_episodes: int = 10, battery_init: int = 100, payload_init: int = 1):
    """Run enhanced random baseline evaluation"""
    
    print("🎲 Running Random Baseline Evaluation")
    print("=" * 50)
    
    # Create environment
    env = DroneDeliveryFullEnv(
        graph_path=graph_path,
        battery_init=battery_init,
        payload_init=payload_init
    )
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_type": "Random",
            "graph_path": graph_path,
            "parameters": {
                "battery_init": battery_init,
                "payload_init": payload_init,
                "n_episodes": n_episodes
            }
        },
        "episodes": [],
        "statistics": {},
        "analysis": {}
    }
    
    successful_episodes = 0
    episode_metrics = {
        "total_costs": [],
        "total_steps": [],
        "total_recharges": [],
        "final_batteries": []
    }
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        print(f"\n📍 Random Episode {episode + 1}/{n_episodes}")
        
        obs, info = env.reset()
        episode_data = {
            "episode": episode + 1,
            "route": [],
            "actions": [],
            "success": False,
            "total_cost": 0.0,
            "total_steps": 0,
            "total_recharges": 0,
            "final_battery": 0.0,
            "termination_reason": ""
        }
        
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated) and step_count < 1000:
            # Random action with action masking if available
            if hasattr(env, 'action_masks'):
                valid_actions = env.action_masks()
                valid_indices = [i for i, valid in enumerate(valid_actions) if valid]
                if valid_indices:
                    action = np.random.choice(valid_indices)
                else:
                    action = env.action_space.sample()
            else:
                # Fallback random action
                if env.at_step_zero:
                    action = env.action_space.sample() % len(env.hubs)
                else:
                    action = env.action_space.sample() % len(env.adjacency[env.current_node])
            
            obs, reward, done, truncated, info = env.step(action)
            
            episode_data["actions"].append(int(action))
            episode_data["route"].append(env.current_node)
            
            if 'edge_cost' in info:
                episode_data["total_cost"] += info['edge_cost']
            
            step_count += 1
        
        # Finalize episode data
        episode_data["success"] = info.get('success', False)
        episode_data["termination_reason"] = info.get('termination_reason', 'unknown')
        episode_data["total_steps"] = step_count
        episode_data["total_recharges"] = env.recharge_count
        episode_data["final_battery"] = env.battery
        
        if episode_data["success"]:
            successful_episodes += 1
            episode_metrics["total_costs"].append(episode_data["total_cost"])
            episode_metrics["total_steps"].append(step_count)
            episode_metrics["total_recharges"].append(env.recharge_count)
            episode_metrics["final_batteries"].append(env.battery)
        
        results["episodes"].append(episode_data)
        
        status = '✅ SUCCESS' if episode_data["success"] else '❌ FAILED'
        print(f"  Result: {status}")
        print(f"  Steps: {step_count}, Cost: {episode_data['total_cost']:.1f}")
    
    evaluation_time = time.time() - start_time
    
    # Calculate statistics
    if successful_episodes > 0:
        stats = {
            "success_rate": successful_episodes / n_episodes,
            "successful_episodes": successful_episodes,
            "total_episodes": n_episodes,
            "avg_total_cost": np.mean(episode_metrics["total_costs"]),
            "avg_steps": np.mean(episode_metrics["total_steps"]),
            "avg_recharges": np.mean(episode_metrics["total_recharges"]),
            "avg_battery_final": np.mean(episode_metrics["final_batteries"]),
            "evaluation_time_seconds": evaluation_time
        }
    else:
        stats = {
            "success_rate": 0.0,
            "successful_episodes": 0,
            "total_episodes": n_episodes,
            "evaluation_time_seconds": evaluation_time
        }
    
    results["statistics"] = stats
    results["analysis"] = {
        "failure_patterns": analyze_failure_patterns(results["episodes"])
    }
    
    print(f"\n📊 Random Baseline Summary:")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Evaluation Time: {evaluation_time:.2f}s")
    if successful_episodes > 0:
        print(f"Average Cost: {stats['avg_total_cost']:.1f}")
        print(f"Average Steps: {stats['avg_steps']:.1f}")
        print(f"Average Recharges: {stats['avg_recharges']:.1f}")
    
    return results


def compare_models(model_paths: list, graph_path: str, n_episodes: int = 10):
    """Compare multiple models on the same evaluation tasks"""
    print("🔄 Comparing Multiple Models")
    print("=" * 50)
    
    comparison_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "graph_path": graph_path,
            "n_episodes": n_episodes,
            "models": model_paths
        },
        "model_results": {},
        "comparison": {}
    }
    
    model_stats = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n🤖 Evaluating {model_name}...")
        
        results = evaluate_ppo_model(
            model_path=model_path,
            graph_path=graph_path,
            n_episodes=n_episodes,
            render=False,
            validate_masking=True
        )
        
        comparison_results["model_results"][model_name] = results
        model_stats[model_name] = results["statistics"]
    
    # Create comparison summary
    if model_stats:
        comparison_results["comparison"] = {
            "best_success_rate": max(stats["success_rate"] for stats in model_stats.values()),
            "best_avg_cost": min(stats.get("avg_total_cost", float('inf')) for stats in model_stats.values() if stats.get("avg_total_cost")),
            "model_ranking": sorted(model_stats.items(), key=lambda x: x[1]["success_rate"], reverse=True)
        }
    
    print(f"\n📊 Model Comparison Summary:")
    for model_name, stats in model_stats.items():
        print(f"  {model_name}: {stats['success_rate']:.1%} success rate")
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Enhanced evaluation for drone delivery models")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--models", nargs='+', help="Multiple model paths for comparison")
    parser.add_argument("--graph", type=str, required=True, help="Path to graph JSON file")
    parser.add_argument("--start", type=int, help="Specific start hub node ID")
    parser.add_argument("--pickup", type=int, help="Specific pickup node ID")
    parser.add_argument("--delivery", type=int, help="Specific delivery node ID")
    parser.add_argument("--battery", type=int, default=100, help="Initial battery level")
    parser.add_argument("--payload", type=int, default=1, help="Initial payload")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--random-eval", action="store_true", help="Run random baseline")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")
    parser.add_argument("--no-masking-validation", action="store_true", help="Skip action masking validation")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Validate files exist
    if args.model:
        if not os.path.exists(args.model):
            print(f"❌ Model file not found: {args.model}")
            sys.exit(1)
    
    if args.models:
        for model_path in args.models:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                sys.exit(1)
    
    if not os.path.exists(args.graph):
        print(f"❌ Graph file not found: {args.graph}")
        sys.exit(1)
    
    # Convert to absolute paths
    graph_path = os.path.abspath(args.graph)
    
    if args.random_eval:
        # Run random evaluation
        results = random_evaluation(
            graph_path, 
            args.episodes, 
            args.battery, 
            args.payload
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"random_baseline": results}, f, indent=2)
    
    elif args.compare and args.models:
        # Compare multiple models
        model_paths = [os.path.abspath(path) for path in args.models]
        results = compare_models(model_paths, graph_path, args.episodes)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    
    elif args.model:
        # Single model evaluation
        model_path = os.path.abspath(args.model)
        results = evaluate_ppo_model(
            model_path=model_path,
            graph_path=graph_path,
            start_node=args.start,
            pickup_node=args.pickup,
            delivery_node=args.delivery,
            battery_init=args.battery,
            payload_init=args.payload,
            n_episodes=args.episodes,
            render=args.render,
            validate_masking=not args.no_masking_validation
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            # Print JSON to stdout for consumption by other tools
            print("\n" + "="*50)
            print("JSON OUTPUT:")
            print(json.dumps(results, indent=2))
    
    else:
        print("❌ Must specify --model, --models with --compare, or --random-eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
