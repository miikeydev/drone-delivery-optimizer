#!/usr/bin/env python3
"""
Live inference script for PPO drone delivery model
Connects the trained model with the web interface graph
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sb3_contrib import MaskablePPO
    MASKABLE_PPO_AVAILABLE = True
    print("✅ MaskablePPO available")
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    print("⚠️ MaskablePPO not available")

from stable_baselines3 import PPO
from env import DroneDeliveryFullEnv


def load_model(model_path: str):
    """Load PPO or MaskablePPO model with automatic detection"""
    print(f"🔄 Loading model from: {model_path}")
    try:
        if MASKABLE_PPO_AVAILABLE:
            try:
                print("🔄 Attempting to load as MaskablePPO...")
                model = MaskablePPO.load(model_path)
                print("✅ Successfully loaded as MaskablePPO")
                return model, "MaskablePPO"
            except Exception as e:
                print(f"⚠️ Failed to load as MaskablePPO: {e}")
                print("🔄 Falling back to regular PPO...")
        
        model = PPO.load(model_path)
        print("✅ Successfully loaded as PPO")
        return model, "PPO"
    except Exception as e:
        print(f"❌ Failed to load model completely: {e}")
        raise Exception(f"Failed to load model: {e}")


def find_node_by_name(nodes: List[Dict], node_name: str) -> Optional[int]:
    """Find node index by name (e.g., 'Pickup 1', 'Delivery 5')"""
    print(f"🔍 Looking for node: '{node_name}'")
    for i, node in enumerate(nodes):
        if node.get('id') == node_name:
            print(f"✅ Found '{node_name}' at index {i}")
            return i
    
    # Si pas trouvé, montrer les options disponibles
    print(f"❌ Node '{node_name}' not found!")
    print("Available nodes:")
    for i, node in enumerate(nodes[:10]):  # Montre les 10 premiers
        print(f"  {i}: {node.get('id', 'Unknown')} ({node.get('type', 'Unknown')})")
    if len(nodes) > 10:
        print(f"  ... and {len(nodes) - 10} more nodes")
    
    return None


def run_inference(model_path: str, graph_path: str, 
                 pickup_node: str, delivery_node: str,
                 battery_capacity: int = 100, max_payload: int = 1,
                 render_steps: bool = True) -> Dict:
    """
    Run live inference with the trained model
    
    Args:
        model_path: Path to trained model (.zip file)
        graph_path: Path to graph JSON file (from web interface)
        pickup_node: Name of pickup node (e.g., "Pickup 1")
        delivery_node: Name of delivery node (e.g., "Delivery 5")
        battery_capacity: Initial battery level
        max_payload: Maximum payload capacity
        render_steps: Whether to print step-by-step details
    
    Returns:
        Dict with inference results including route, actions, costs, etc.
    """
    
    print(f"🚁 Starting PPO Live Inference")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Graph: {graph_path}")
    print(f"Route: {pickup_node} → {delivery_node}")
    print(f"Battery: {battery_capacity}%, Payload: {max_payload}")
    print(f"Render steps: {render_steps}")
    
    # Load model
    try:
        model, model_type = load_model(model_path)
        print(f"✅ {model_type} model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return {
            "status": "error",
            "message": f"Failed to load model: {e}",
            "route": [],
            "actions": [],
            "costs": [],
            "stats": {}
        }
    
    # Load graph data
    try:
        print(f"🔄 Loading graph from: {graph_path}")
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        print(f"✅ Loaded graph with {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
        
        # Analyser les types de nœuds
        node_types = {}
        for node in graph_data['nodes']:
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        print(f"📊 Node types: {node_types}")
        
    except Exception as e:
        print(f"❌ Graph loading failed: {e}")
        return {
            "status": "error", 
            "message": f"Failed to load graph: {e}",
            "route": [],
            "actions": [],
            "costs": [],
            "stats": {}
        }
    
    # Create environment with same parameters as training
    try:
        print(f"🔄 Creating environment...")
        env = DroneDeliveryFullEnv(
            graph_path=graph_path,
            battery_init=battery_capacity,
            payload_init=max_payload,
            max_steps=200
        )
        print("✅ Environment created")
        print(f"📊 Environment info:")
        print(f"  - Observation space: {env.observation_space.shape}")
        print(f"  - Action space: {env.action_space.n}")
        print(f"  - Hubs: {len(env.hubs)} {env.hubs[:5]}{'...' if len(env.hubs) > 5 else ''}")
        print(f"  - Pickups: {len(env.pickups)} {env.pickups[:5]}{'...' if len(env.pickups) > 5 else ''}")
        print(f"  - Deliveries: {len(env.deliveries)} {env.deliveries[:5]}{'...' if len(env.deliveries) > 5 else ''}")
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return {
            "status": "error",
            "message": f"Failed to create environment: {e}",
            "route": [],
            "actions": [],
            "costs": [],
            "stats": {}
        }
    
    # Find target nodes by name
    pickup_idx = find_node_by_name(graph_data['nodes'], pickup_node)
    delivery_idx = find_node_by_name(graph_data['nodes'], delivery_node)
    
    if pickup_idx is None:
        return {
            "status": "error",
            "message": f"Pickup node '{pickup_node}' not found in graph",
            "route": [],
            "actions": [],
            "costs": [],
            "stats": {}
        }
    
    if delivery_idx is None:
        return {
            "status": "error", 
            "message": f"Delivery node '{delivery_node}' not found in graph",
            "route": [],
            "actions": [],
            "costs": [],
            "stats": {}
        }
    
    print(f"🎯 Target mapping:")
    print(f"  - Pickup: '{pickup_node}' → node {pickup_idx}")
    print(f"  - Delivery: '{delivery_node}' → node {delivery_idx}")
    
    # Override environment targets to match user selection
    print(f"🔄 Setting environment targets...")
    original_pickups = env.pickups.copy()
    original_deliveries = env.deliveries.copy()
    env.pickups = [pickup_idx]
    env.deliveries = [delivery_idx]
    print(f"  - Original pickups: {original_pickups}")
    print(f"  - New pickups: {env.pickups}")
    print(f"  - Original deliveries: {original_deliveries}")
    print(f"  - New deliveries: {env.deliveries}")
    
    # Run inference
    print(f"\n🚀 Starting inference episode...")
    obs, info = env.reset()
    print(f"📊 Reset info: {info}")
    print(f"📊 Initial observation shape: {obs.shape}")
    
    # Track inference data
    route = []
    actions = []
    costs = []
    rewards = []
    battery_history = [env.battery]
    step_count = 0
    total_cost = 0.0
    
    done = False
    truncated = False
    
    print(f"📊 Initial state:")
    print(f"  - Current node: {env.current_node}")
    print(f"  - Battery: {env.battery}")
    print(f"  - Payload: {env.payload}")
    print(f"  - At step zero: {env.at_step_zero}")
    print(f"  - Pickup target: {env.pickup_target}")
    print(f"  - Delivery target: {env.delivery_target}")
    
    while not (done or truncated) and step_count < 200:
        print(f"\n--- Step {step_count + 1} ---")
        
        # Check action masks
        if hasattr(env, 'action_masks'):
            action_mask = env.action_masks()
            valid_actions = np.where(action_mask)[0]
            print(f"🎭 Valid actions: {valid_actions}")
        else:
            print("⚠️ No action masking available")
            valid_actions = list(range(env.action_space.n))
        
        # Get action from model
        try:
            if model_type == "MaskablePPO" and hasattr(env, 'action_masks'):
                action_mask = env.action_masks()
                action, _states = model.predict(obs, action_masks=action_mask, deterministic=True)
                print(f"🤖 MaskablePPO chose action: {action}")
            else:
                action, _states = model.predict(obs, deterministic=True)
                print(f"🤖 PPO chose action: {action}")
            
            action = int(action)
            print(f"✅ Action selected: {action}")
            
        except Exception as e:
            print(f"❌ Model prediction failed: {e}")
            # Fallback à une action valide
            if valid_actions:
                action = valid_actions[0]
                print(f"🔄 Fallback to action: {action}")
            else:
                print(f"💀 No valid actions available!")
                break
        
        # Vérifier que l'action est valide
        if hasattr(env, 'action_masks'):
            if not env.action_masks()[action]:
                print(f"⚠️ WARNING: Model chose invalid action {action}!")
        
        # Execute action
        try:
            print(f"🏃 Executing action {action}...")
            obs, reward, done, truncated, info = env.step(action)
            print(f"✅ Step executed successfully")
            
        except Exception as e:
            print(f"❌ Step execution failed: {e}")
            break
        
        # Track data
        actions.append(action)
        rewards.append(float(reward))
        battery_history.append(env.battery)
        
        if info.get('distance'):
            costs.append(float(info['distance']))
            total_cost += float(info['distance'])
        
        # Track route (current node)
        if env.current_node is not None:
            route.append(env.current_node)
        
        step_count += 1
        
        # Log step details
        action_type = info.get('action_type', 'unknown')
        current_node_name = graph_data['nodes'][env.current_node]['id'] if env.current_node is not None else 'None'
        
        print(f"📊 Step results:")
        print(f"  - Action: {action} ({action_type})")
        print(f"  - Current node: {env.current_node} ({current_node_name})")
        print(f"  - Reward: {reward:.3f}")
        print(f"  - Battery: {env.battery:.1f}")
        print(f"  - Done: {done}, Truncated: {truncated}")
        print(f"  - Pickup done: {env.pickup_done}")
        print(f"  - Delivery done: {env.delivery_done}")
        
        if 'distance' in info:
            print(f"  - Distance: {info['distance']:.2f}")
        if 'termination_reason' in info:
            print(f"  - Termination: {info['termination_reason']}")
        
        if done or truncated:
            reason = info.get('termination_reason', 'unknown')
            print(f"🏁 Episode finished: {reason}")
            break
            
        # Safety check
        if step_count >= 200:
            print(f"⏰ Max steps reached!")
            break
    
    # Generate route description with node names
    route_names = []
    for node_idx in route:
        if node_idx < len(graph_data['nodes']):
            node_name = graph_data['nodes'][node_idx]['id']
            route_names.append(node_name)
    
    # Determine success - correction ici !
    success = done and info.get('termination_reason') == 'successful_delivery'
    
    # Calculate statistics
    stats = {
        "success": success,
        "total_steps": step_count,
        "total_cost": total_cost,
        "battery_used": battery_capacity - env.battery,
        "battery_remaining": env.battery,
        "recharges": env.recharge_count,
        "pickup_completed": env.pickup_done,
        "delivery_completed": env.delivery_done,
        "termination_reason": info.get('termination_reason', 'unknown')
    }
    
    result = {
        "status": "success" if success else "failed",
        "message": f"Mission {'completed' if success else 'failed'}: {info.get('termination_reason', 'unknown')}",
        "route": route,  # Node indices
        "route_names": route_names,  # Node names for display
        "actions": actions,
        "rewards": rewards,
        "costs": costs,
        "battery_history": battery_history,
        "stats": stats,
        "graph_info": {
            "total_nodes": len(graph_data['nodes']),
            "pickup_node": pickup_node,
            "delivery_node": delivery_node,
            "pickup_idx": pickup_idx,
            "delivery_idx": delivery_idx
        }
    }
    
    print(f"\n📊 Final Inference Results:")
    print(f"Success: {success}")
    print(f"Steps: {step_count}")
    print(f"Route indices: {route}")
    print(f"Route names: {' → '.join(route_names) if route_names else 'No route'}")
    print(f"Battery used: {stats['battery_used']:.1f}%")
    print(f"Total cost: {total_cost:.2f}")
    print(f"Pickup completed: {env.pickup_done}")
    print(f"Delivery completed: {env.delivery_done}")
    print(f"Termination reason: {stats['termination_reason']}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Live PPO inference for drone delivery")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--graph", type=str, required=True, help="Path to graph JSON file")
    parser.add_argument("--pickup", type=str, required=True, help="Pickup node name (e.g., 'Pickup 1')")
    parser.add_argument("--delivery", type=str, required=True, help="Delivery node name (e.g., 'Delivery 5')")
    parser.add_argument("--battery", type=int, default=100, help="Initial battery level")
    parser.add_argument("--payload", type=int, default=1, help="Maximum payload")
    parser.add_argument("--quiet", action="store_true", help="Disable step-by-step rendering")
    
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
        
    if not os.path.exists(args.graph):
        print(f"❌ Graph file not found: {args.graph}")
        sys.exit(1)
    
    # Run inference
    result = run_inference(
        model_path=args.model,
        graph_path=args.graph,
        pickup_node=args.pickup,
        delivery_node=args.delivery,
        battery_capacity=args.battery,
        max_payload=args.payload,
        render_steps=not args.quiet
    )
    
    # Output JSON result for web interface
    print("\n" + "="*50)
    print("JSON OUTPUT:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
