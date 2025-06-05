#!/usr/bin/env python3
"""
Clean live inference for PPO drone delivery
"""

import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sb3_contrib import MaskablePPO
    MASKABLE_AVAILABLE = True
except ImportError:
    MASKABLE_AVAILABLE = False

from stable_baselines3 import PPO
from env import DroneDeliveryFullEnv


def load_model(model_path: str):
    """Load model with automatic detection"""
    if MASKABLE_AVAILABLE:
        try:
            model = MaskablePPO.load(model_path)
            return model, "MaskablePPO"
        except:
            pass
    
    # Fallback to regular PPO
    model = PPO.load(model_path)
    return model, "PPO"


def find_node_by_name(nodes: list, node_name: str) -> int:
    """Find node index by name"""
    for i, node in enumerate(nodes):
        if node['id'] == node_name:
            return i
    raise ValueError(f"Node '{node_name}' not found")


def run_clean_inference(model_path: str, graph_path: str, 
                       pickup_name: str, delivery_name: str,
                       battery: int = 100, payload: int = 1) -> dict:
    """Run clean inference with normalization support"""
    
    # Load model
    model, model_type = load_model(model_path)
    
    # Load graph
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    
    # Find target nodes
    pickup_idx = find_node_by_name(graph_data['nodes'], pickup_name)
    delivery_idx = find_node_by_name(graph_data['nodes'], delivery_name)
    
    # Create environment with rebalanced parameters
    env = DroneDeliveryFullEnv(
        graph_path=graph_path,
        battery_init=battery,
        payload_init=payload,
        max_steps=150,
        k_neighbors=10,
        use_curriculum=False,
        curriculum_threshold=0.25  # CHANGED: Match training threshold
    )
    
    # Try to load normalization stats if available
    vecnormalize_path = model_path.replace('.zip', '_vecnormalize.pkl')
    if os.path.exists(vecnormalize_path):
        print(f"Loading normalization stats from {vecnormalize_path}")
        try:
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load(vecnormalize_path, env)
            env.training = False
            env.norm_reward = False
            print("✅ Normalization stats loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load normalization stats: {e}")
            env = env.envs[0] if hasattr(env, 'envs') else env
    
    # SIMPLIFIED LOGS - just essential info
    print(f"Environment: {getattr(env, 'n_nodes', 'wrapped')} nodes")
    print(f"Rebalanced rewards: move=-0.001, pickup=5.0, delivery=30.0")
    
    # Force specific targets BEFORE reset
    if hasattr(env, 'pickup_target'):
        env.pickup_target = pickup_idx
        env.delivery_target = delivery_idx
        env.pickups = [pickup_idx]
        env.deliveries = [delivery_idx]
    
    # Reset environment
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle VecNormalize wrapper
    
    # Run episode
    route = [env.current_node if hasattr(env, 'current_node') else 0]
    actions = []
    battery_history = [battery]
    step_count = 0
    action_types = []
    
    done = False
    truncated = False
    
    while not (done or truncated) and step_count < 200:
        # Get action from model
        if model_type == "MaskablePPO":
            if hasattr(env, 'action_masks'):
                action_mask = env.action_masks()
            else:
                action_mask = env.env_method('action_masks')[0] if hasattr(env, 'env_method') else None
            
            if action_mask is not None:
                action, _states = model.predict(obs, action_masks=action_mask, deterministic=True)
            else:
                action, _states = model.predict(obs, deterministic=True)
        else:
            action, _states = model.predict(obs, deterministic=True)
        
        # Execute action
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, done, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            truncated = False
        
        # Handle VecEnv wrapper
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            obs = obs[0]
        if isinstance(reward, (list, np.ndarray)):
            reward = reward[0] if len(reward) > 0 else 0
        if isinstance(done, (list, np.ndarray)):
            done = done[0] if len(done) > 0 else False
        if isinstance(info, list):
            info = info[0] if len(info) > 0 else {}
        
        # Track step info
        actions.append(int(action))
        current_node = getattr(env, 'current_node', None)
        if hasattr(env, 'envs') and env.envs:
            current_node = getattr(env.envs[0], 'current_node', None)
        
        route.append(current_node if current_node is not None else route[-1])
        
        current_battery = getattr(env, 'battery', battery)
        if hasattr(env, 'envs') and env.envs:
            current_battery = getattr(env.envs[0], 'battery', battery)
        
        battery_history.append(current_battery)
        action_types.append(info.get('action_type', 'unknown'))
        step_count += 1
        
        # SIMPLIFIED STEP LOGS - only essential info
        print(f"Step {step_count}: Action={action}, Node={current_node}, "
              f"Battery={current_battery:.1f}, Reward={reward:.2f}")
        
        # REMOVED: Progress reward logging - too verbose
        # if 'progress_reward' in info:
        #     print(f"       Progress: {info['progress_reward']:.3f}")
        
        if done or truncated:
            print(f"Episode ended: {info.get('termination_reason', 'unknown')}")
            break
    
    # Generate route names
    route_names = []
    for node_idx in route:
        if node_idx < len(graph_data['nodes']):
            route_names.append(graph_data['nodes'][node_idx]['id'])
        else:
            route_names.append(f"Node_{node_idx}")
    
    # Determine success
    success = (env.pickup_done and env.delivery_done and 
               env.pickup_target == pickup_idx and 
               env.delivery_target == delivery_idx)
    
    # Calculate total distance
    total_distance = 0
    if len(route) > 1:
        for i in range(1, len(route)):
            prev_node = route[i-1]
            curr_node = route[i]
            # Find edge distance
            for edge in graph_data['edges']:
                if ((edge.get('source', edge.get('u')) == prev_node and 
                     edge.get('target', edge.get('v')) == curr_node) or
                    (edge.get('source', edge.get('u')) == curr_node and 
                     edge.get('target', edge.get('v')) == prev_node)):
                    total_distance += edge.get('distance', edge.get('dist', 0))
                    break
    
    # Return clean result with enhanced debugging info
    return {
        'success': info.get('success', False),
        'steps': step_count,
        'route_indices': route,
        'route_names': [graph_data['nodes'][i]['id'] if i < len(graph_data['nodes']) else f"Node_{i}" for i in route],
        'actions': actions,
        'action_types': action_types,
        'battery_history': battery_history,
        'battery_used': battery - battery_history[-1],
        'battery_final': battery_history[-1],
        'total_distance': 0,  # Calculate if needed
        'pickup_done': info.get('pickup_done', False),
        'delivery_done': info.get('delivery_done', False),
        'pickup_target': pickup_name,
        'delivery_target': delivery_name,
        'termination_reason': info.get('termination_reason', 'completed'),
        'model_type': model_type,
        'total_reward': info.get('total_reward', 0),
        'rebalanced': True
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run PPO inference for drone delivery")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--graph", type=str, required=True, help="Path to graph JSON")
    parser.add_argument("--pickup", type=str, required=True, help="Pickup node name")
    parser.add_argument("--delivery", type=str, required=True, help="Delivery node name")
    parser.add_argument("--battery", type=int, default=100, help="Initial battery")
    parser.add_argument("--payload", type=int, default=1, help="Payload weight")
    
    args = parser.parse_args()
    
    try:
        result = run_clean_inference(
            model_path=args.model,
            graph_path=args.graph,
            pickup_name=args.pickup,
            delivery_name=args.delivery,
            battery=args.battery,
            payload=args.payload
        )
        
        print("\n" + "="*60)
        print("INFERENCE RESULTS")
        print("="*60)
        print(f"Success: {result['success']}")
        print(f"Steps: {result['steps']}")
        print(f"Battery used: {result['battery_used']:.1f}%")
        print(f"Total distance: {result['total_distance']:.2f}km")
        print(f"Pickup completed: {result['pickup_done']}")
        print(f"Delivery completed: {result['delivery_done']}")
        print(f"Route: {' → '.join(result['route_names'])}")
        print(f"Model: {result['model_type']}")
        
        # Save results
        output_file = "inference_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # ADDED: Windows BrokenPipe fix
    main()
