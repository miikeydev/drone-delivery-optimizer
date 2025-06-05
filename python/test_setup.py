"""
Quick test script to validate the PPO environment setup
"""
import os
import sys
import json
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_graph_loading():
    """Test graph JSON loading"""
    # Get the correct path relative to the python script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(os.path.dirname(current_dir), 'data', 'graph.json')
    
    print(f"Looking for graph at: {graph_path}")
    
    if not os.path.exists(graph_path):
        print("❌ Graph file not found. Please generate the network in the frontend first.")
        return False
    
    try:
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        print(f"✅ Graph loaded: {len(nodes)} nodes, {len(edges)} edges")
        
        # Check node types
        node_types = {}
        for node in nodes:
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print(f"   Node types: {node_types}")
        
        # Check max edge cost
        if edges:
            max_cost = max(edge.get('cost', 0) for edge in edges)
            print(f"   Max edge cost: {max_cost:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        return False


def test_environment():
    """Test environment creation"""
    try:
        from env import DroneDeliveryFullEnv
        
        # Get the correct path relative to the python script location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        graph_path = os.path.join(os.path.dirname(current_dir), 'data', 'graph.json')
        
        env = DroneDeliveryFullEnv(graph_path)
        print(f"✅ Environment created successfully")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.n}")
          # Test reset
        obs, info = env.reset()  # gymnasium format returns (obs, info)
        print(f"   Initial observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(3):
            action_mask = env._get_action_mask()
            valid_actions = np.where(action_mask)[0]
            
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = env.step(action)  # gymnasium format returns 5 values
                print(f"   Step {i+1}: Action={action}, Reward={reward:.3f}, Done={done}")
            else:
                print(f"   Step {i+1}: No valid actions available")
                break
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Please run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


def test_model_loading():
    """Test if we can load stable-baselines3"""
    try:
        from stable_baselines3 import PPO
        print("✅ Stable-baselines3 available")
        return True
    except ImportError:
        print("❌ Stable-baselines3 not available")
        print("   Please run: pip install stable-baselines3")
        return False


def main():
    print("🚁 Testing PPO Drone Delivery Environment")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Graph loading
    print("\n1. Testing graph loading...")
    if test_graph_loading():
        tests_passed += 1
    
    # Test 2: Environment creation
    print("\n2. Testing environment...")
    if test_environment():
        tests_passed += 1
    
    # Test 3: Model library
    print("\n3. Testing PPO library...")
    if test_model_loading():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! You can now train the PPO model.")
        print("\nNext steps:")
        print("1. python train_simpleppo.py --graph ../data/graph.json")
        print("2. python eval_simpleppo.py --graph ../data/graph.json --random-eval")
    else:
        print("❌ Some tests failed. Please fix the issues above.")


if __name__ == "__main__":
    main()
