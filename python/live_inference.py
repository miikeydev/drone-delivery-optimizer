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
    """Run clean inference - just the essentials"""
    
    # Load model
    model, model_type = load_model(model_path)
    
    # Load graph
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    
    # Find target nodes
    pickup_idx = find_node_by_name(graph_data['nodes'], pickup_name)
    delivery_idx = find_node_by_name(graph_data['nodes'], delivery_name)
    
    # Create environment with same parameters as training but NO curriculum for inference
    env = DroneDeliveryFullEnv(
        graph_path=graph_path,
        battery_init=battery,
        payload_init=payload,
        max_steps=150,
        k_neighbors=6,
        use_curriculum=False
    )
    
    # Verify observation space dimensions match model expectations
    print(f"Environment: {env.n_nodes} nodes, action space: {env.action_space.n}")
    print(f"Actions: 0-{env.k_neighbors-1}=move, {env.STAY_ACTION}=stay/recharge")
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"FIXED: dense progress rewards + neighbor shuffle + enhanced [512,256,128] architecture")
    
    # Force specific targets BEFORE reset
    env.pickup_target = pickup_idx
    env.delivery_target = delivery_idx
    env.pickups = [pickup_idx]
    env.deliveries = [delivery_idx]
    
    # Reset environment
    obs, info = env.reset()
    print(f"Observation shape after reset: {obs.shape}")
    
    # Verify targets are correct
    assert env.pickup_target == pickup_idx, f"Pickup target mismatch: {env.pickup_target} != {pickup_idx}"
    assert env.delivery_target == delivery_idx, f"Delivery target mismatch: {env.delivery_target} != {delivery_idx}"
    
    # Run episode
    route = [env.current_node]
    actions = []
    battery_history = [env.battery]
    step_count = 0
    action_types = []  # Track action types for debugging
    
    done = False
    truncated = False
    
    while not (done or truncated) and step_count < 200:
        # Get action from model
        if model_type == "MaskablePPO":
            action_mask = env.action_masks()
            action, _states = model.predict(obs, action_masks=action_mask, deterministic=True)
        else:
            action, _states = model.predict(obs, deterministic=True)
        
        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        
        # Track step info
        actions.append(int(action))
        route.append(env.current_node)
        battery_history.append(env.battery)
        action_types.append(info.get('action_type', 'unknown'))
        step_count += 1
        
        print(f"Step {step_count}: Action={action}, Node={env.current_node}, "
              f"Battery={env.battery:.1f}, Reward={reward:.3f}, "
              f"Type={info.get('action_type', 'move')}")
        
        # NEW: Show progress information when available
        if 'progress_reward' in info:
            progress_type = "pickup" if 'progress_to_pickup' in info else "delivery"
            progress_val = info.get(f'progress_to_{progress_type}', 0)
            print(f"       Progress to {progress_type}: {progress_val:.2f}, "
                  f"Reward: {info['progress_reward']:.3f}")
        
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
        'success': success,
        'steps': step_count,
        'route_indices': route,
        'route_names': route_names,
        'actions': actions,
        'action_types': action_types,
        'battery_history': battery_history,
        'battery_used': battery - env.battery,
        'battery_final': env.battery,
        'total_distance': total_distance,
        'pickup_done': env.pickup_done,
        'delivery_done': env.delivery_done,
        'pickup_target': pickup_name,
        'delivery_target': delivery_name,
        'termination_reason': info.get('termination_reason', 'completed'),
        'model_type': model_type,
        'total_reward': getattr(env, 'total_reward', 0)
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
    main()
