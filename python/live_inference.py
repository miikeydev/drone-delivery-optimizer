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
    print(f"Loading model from: {model_path}")
    
    if MASKABLE_AVAILABLE:
        try:
            model = MaskablePPO.load(model_path)
            print(f"Loaded as MaskablePPO")
            return model, "MaskablePPO"
        except Exception as e:
            print(f"MaskablePPO failed: {e}")
    
    try:
        model = PPO.load(model_path)
        print(f"Loaded as PPO")
        return model, "PPO"
    except Exception as e:
        print(f"PPO load failed: {e}")
        raise


def find_node_by_name(nodes: list, node_name: str) -> int:
    """Find node index by name"""
    for i, node in enumerate(nodes):
        if node['id'] == node_name:
            return i
    raise ValueError(f"Node '{node_name}' not found in {len(nodes)} nodes")


def run_clean_inference(model_path: str, graph_path: str, 
                       pickup_name: str, delivery_name: str,
                       battery: int = 100, payload: int = 1) -> dict:
    """Run clean inference with better error handling"""
    
    print(f"Starting inference:")
    print(f"   Model: {model_path}")
    print(f"   Graph: {graph_path}")
    print(f"   Route: {pickup_name} -> {delivery_name}")
    print(f"   Battery: {battery}%, Payload: {payload}")
    
    # Load model
    model, model_type = load_model(model_path)
    
    # Load graph
    print(f"Loading graph data...")
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    
    print(f"Graph loaded: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
    
    # Find target nodes
    pickup_idx = find_node_by_name(graph_data['nodes'], pickup_name)
    delivery_idx = find_node_by_name(graph_data['nodes'], delivery_name)
    
    print(f"Targets: Pickup={pickup_idx}, Delivery={delivery_idx}")
    
    # Create environment
    print(f"Creating environment...")
    env = DroneDeliveryFullEnv(
        graph_path=graph_path,
        battery_init=battery,
        payload_init=payload,
        max_steps=200,
        k_neighbors=10,
        use_curriculum=False,
        curriculum_threshold=0.25
    )
    
    # FIXED: Handle VecNormalize if exists
    vecnormalize_path = model_path.replace('.zip', '_vecnormalize.pkl')
    if os.path.exists(vecnormalize_path):
        print(f"Loading normalization from {vecnormalize_path}")
        try:
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load(vecnormalize_path, env)
            env.training = False
            env.norm_reward = False
            print("VecNormalize loaded")
            is_vectorized = True
        except Exception as e:
            print(f"VecNormalize failed: {e}")
            env = env.envs[0] if hasattr(env, 'envs') else env
            is_vectorized = False
    else:
        print("No VecNormalize file found")
        is_vectorized = False
    
    # FIXED: Set targets properly
    if hasattr(env, 'pickup_target'):
        env.pickup_target = pickup_idx
        env.delivery_target = delivery_idx
    elif hasattr(env, 'envs') and env.envs:
        env.envs[0].pickup_target = pickup_idx
        env.envs[0].delivery_target = delivery_idx
    
    print(f"Resetting environment...")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    # Initialize tracking
    route = []
    actions = []
    battery_history = []
    action_types = []
    step_count = 0

    info = {}
    
    current_node = getattr(env, 'current_node', None)
    if hasattr(env, 'envs') and env.envs:
        current_node = getattr(env.envs[0], 'current_node', None)
    
    current_battery = getattr(env, 'battery', battery)
    if hasattr(env, 'envs') and env.envs:
        current_battery = getattr(env.envs[0], 'battery', battery)
    
    route.append(current_node if current_node is not None else 0)
    battery_history.append(current_battery)
    
    print(f"Starting at node {current_node} with {current_battery}% battery")
    
    done = False
    truncated = False
    
    while not (done or truncated) and step_count < 300:
        try:
            # FIXED: Get action from model - handle vectorized environment properly
            if model_type == "MaskablePPO":
                # Get action mask
                if hasattr(env, 'action_masks'):
                    action_mask = env.action_masks()
                elif hasattr(env, 'env_method'):
                    action_mask = env.env_method('action_masks')[0]
                else:
                    action_mask = None
                
                if action_mask is not None:
                    action, _states = model.predict(obs, action_masks=action_mask, deterministic=True)
                else:
                    action, _states = model.predict(obs, deterministic=True)
            else:
                action, _states = model.predict(obs, deterministic=True)
            
            # FIXED: Convert action to scalar if it's an array
            if isinstance(action, np.ndarray):
                if action.ndim == 0:
                    action = action.item()
                elif action.size == 1:
                    action = action[0]
                else:
                    print(f"Warning: Multi-dimensional action: {action}")
                    action = action[0]
            
            # FIXED: Convert action to numpy array for vectorized env
            if is_vectorized:
                action_array = np.array([action])
            else:
                action_array = action
            
            # Execute action
            step_result = env.step(action_array)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                truncated = False
            
            # FIXED: Handle vectorized outputs
            if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
                obs = obs[0]
            if isinstance(reward, (list, np.ndarray)):
                reward = reward[0] if len(reward) > 0 else 0
            if isinstance(done, (list, np.ndarray)):
                done = done[0] if len(done) > 0 else False
            if isinstance(truncated, (list, np.ndarray)):
                truncated = truncated[0] if len(truncated) > 0 else False
            if isinstance(info, list):
                info = info[0] if len(info) > 0 else {}
            
            # Track step data
            actions.append(int(action))
            
            # Get current state
            current_node = getattr(env, 'current_node', None)
            if hasattr(env, 'envs') and env.envs:
                current_node = getattr(env.envs[0], 'current_node', None)
            
            current_battery = getattr(env, 'battery', battery)
            if hasattr(env, 'envs') and env.envs:
                current_battery = getattr(env.envs[0], 'battery', battery)
            
            route.append(current_node if current_node is not None else route[-1])
            battery_history.append(current_battery)
            action_types.append(info.get('action_type', 'unknown'))
            step_count += 1
            
            print(f"Step {step_count}: Action={action}, Node={current_node}, "
                  f"Battery={current_battery:.1f}%, Reward={reward:.2f}")
            
            if done or truncated:
                reason = info.get('termination_reason', 'unknown')
                print(f"Episode ended: {reason}")
                break
                
        except Exception as e:
            print(f"Step error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # FIXED: Build route names properly
    route_names = []
    for node_idx in route:
        if isinstance(node_idx, (int, np.integer)) and node_idx < len(graph_data['nodes']):
            route_names.append(graph_data['nodes'][int(node_idx)]['id'])
        else:
            route_names.append(f"Node_{node_idx}")
    
    # FIXED: Get final state with fallback
    pickup_done = info.get('pickup_done', False)
    delivery_done = info.get('delivery_done', False)
    success = delivery_done
    
    result = {
        'success': success,
        'steps': step_count,
        'route_indices': [int(x) for x in route],
        'route_names': route_names,
        'actions': [int(x) for x in actions],
        'battery_history': [float(x) for x in battery_history],
        'battery_used': float(battery - battery_history[-1]),
        'battery_final': float(battery_history[-1]),
        'pickup_done': pickup_done,
        'delivery_done': delivery_done,
        'pickup_target': pickup_name,
        'delivery_target': delivery_name,
        'termination_reason': info.get('termination_reason', 'completed'),
        'model_type': model_type,
        'total_reward': float(info.get('total_reward', 0)),
        'total_distance': 0.0
    }
    
    print(f"Final results:")
    print(f"   Success: {success}")
    print(f"   Steps: {step_count}")
    print(f"   Pickup: {pickup_done}, Delivery: {delivery_done}")
    print(f"   Battery: {battery_history[-1]:.1f}% remaining")
    print(f"   Route: {' -> '.join(route_names[:5])}{'...' if len(route_names) > 5 else ''}")
    
    # REMIS: Sauvegarde pour l'affichage web
    output_file = "inference_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


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
    
    print("DRONE DELIVERY PPO INFERENCE")
    print("=" * 50)
    
    try:
        result = run_clean_inference(
            model_path=args.model,
            graph_path=args.graph,
            pickup_name=args.pickup,
            delivery_name=args.delivery,
            battery=args.battery,
            payload=args.payload
        )
        
        print("\nFINAL RESULTS")
        print("=" * 50)
        print(f"Success: {result['success']}")
        print(f"Steps: {result['steps']}")
        print(f"Battery used: {result['battery_used']:.1f}%")
        print(f"Pickup completed: {result['pickup_done']}")
        print(f"Delivery completed: {result['delivery_done']}")
        print(f"Route: {' -> '.join(result['route_names'])}")
        print(f"Model: {result['model_type']}")
        print(f"Total reward: {result['total_reward']:.2f}")
        

        print(f"\nInference completed without file save")
            
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Value error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
