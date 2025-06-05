import gymnasium as gym
import numpy as np
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from gymnasium import spaces


class DroneDeliveryFullEnv(gym.Env):
    """
    Full-featured PPO environment for drone delivery with all constraints.
    Implements the complete specification from the road-map:
    - UI parameters (battery_init, payload_init)
    - Step 0: hub selection with teleport action
    - Full observation space with distances, adjacency, recharge count
    - Battery consumption formula with payload effect
    - Recharge mechanics with penalties
    - Complete reward system with pickup/delivery bonuses
    - Proper termination conditions
    """
    def __init__(self, graph_path: str, battery_init: int = 100, payload_init: int = 1, 
                 max_battery: int = 100, k_neighbors: int = 10, max_steps: int = 200,
                 randomize_battery: bool = False, battery_range: tuple = (60, 100),
                 randomize_payload: bool = False, payload_range: tuple = (1, 5)):
        super(DroneDeliveryFullEnv, self).__init__()
        
        # Load graph data
        with open(graph_path, 'r') as f:
            self.graph_data = json.load(f)
        
        self.nodes = self.graph_data['nodes']
        self.edges = self.graph_data['edges']
        self.n_nodes = len(self.nodes)
        self.k_neighbors = k_neighbors
        self.max_battery = max_battery
        self.battery_init = battery_init
        self.payload_init = payload_init
        self.max_steps = max_steps
        
        # Randomization parameters
        self.randomize_battery = randomize_battery
        self.battery_range = battery_range
        self.randomize_payload = randomize_payload
        self.payload_range = payload_range
        
        # Battery consumption parameters
        self.k_norm = 10.8  # normalization factor
        self.alpha = 0.2    # payload effect factor
        
        # Recharge parameters
        self.recharge_step = 20
        self.recharge_penalty = 0.05  # μ = 0.05
        
        # Build adjacency structure for fast lookup
        self.adjacency = self._build_adjacency()
        
        # Find node indices by type
        self.hubs = [i for i, node in enumerate(self.nodes) if node['type'] == 'hubs']
        self.pickups = [i for i, node in enumerate(self.nodes) if node['type'] == 'pickup']
        self.deliveries = [i for i, node in enumerate(self.nodes) if node['type'] == 'delivery']
        self.charging_stations = [i for i, node in enumerate(self.nodes) if node['type'] == 'charging']
        
        # Node type mappings
        self.type_to_id = {'hubs': 0, 'pickup': 1, 'delivery': 2, 'charging': 3}
        
        # State space calculation:
        # one-hot position (N) + battery (1) + payload (1) + flags (2) + 
        # distances (3) + recharge_count (1) + local_adjacency (k*4)
        state_dim = self.n_nodes + 1 + 1 + 2 + 3 + 1 + (self.k_neighbors * 4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(state_dim,), dtype=np.float32)
        
        # Action space: varies by step
        # Step 0: choose hub (len(hubs))
        # Step 1+: choose edge (k_neighbors)
        self.action_space = spaces.Discrete(max(len(self.hubs), self.k_neighbors))
        
        # Environment state
        self.current_node = None
        self.battery = self.battery_init
        self.payload = self.payload_init
        self.pickup_done = False
        self.delivery_done = False
        self.pickup_target = None
        self.delivery_target = None
        self.recharge_count = 0
        self.step_count = 0
        self.at_step_zero = True
        
        print(f"Initialized full environment with {self.n_nodes} nodes, {len(self.edges)} edges")
        print(f"Hubs: {len(self.hubs)}, Pickups: {len(self.pickups)}, Deliveries: {len(self.deliveries)}")
        print(f"State dimension: {state_dim}")
        print(f"Battery init: {battery_init}, Payload init: {payload_init}")
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the environment's random number generator."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self._np_random = np.random.RandomState(seed)
        else:
            seed = np.random.randint(0, 2**32 - 1)
            random.seed(seed)
            np.random.seed(seed)
            self._np_random = np.random.RandomState(seed)
        
        return [seed]
    
    def _build_adjacency(self) -> Dict[int, List[Dict]]:
        """Build adjacency list from edges"""
        adj = {i: [] for i in range(self.n_nodes)}
        
        for edge in self.edges:
            adj[edge['u']].append({
                'target': edge['v'],
                'distance': edge['dist'],
                'cost': edge['cost']
            })
        
        # Pad adjacency lists to k_neighbors length
        for node_id in adj:
            while len(adj[node_id]) < self.k_neighbors:
                adj[node_id].append({
                    'target': -1,  # Invalid target
                    'distance': 0,
                    'cost': 0
                })
        
        return adj
    
    def _get_node_type(self, node_idx: int) -> str:
        """Get type of node"""
        return self.nodes[node_idx]['type']
    
    def _get_node_type_id(self, node_idx: int) -> int:
        """Get numeric type ID of node"""
        return self.type_to_id[self._get_node_type(node_idx)]
    
    def _calculate_distance(self, node1_idx: int, node2_idx: int) -> float:
        """Calculate Euclidean distance between two nodes"""
        lat1, lng1 = self.nodes[node1_idx]['lat'], self.nodes[node1_idx]['lng']
        lat2, lng2 = self.nodes[node2_idx]['lat'], self.nodes[node2_idx]['lng']
        return ((lat1 - lat2)**2 + (lng1 - lng2)**2)**0.5
    
    def _distance_to_closest_type(self, current_node: int, node_type: str) -> float:
        """Calculate distance to closest node of given type"""
        if node_type == 'pickup':
            target_nodes = self.pickups
        elif node_type == 'delivery':
            target_nodes = self.deliveries
        elif node_type == 'hubs':
            target_nodes = self.hubs
        else:
            return 100.0
        
        if not target_nodes:
            return 100.0
        
        min_dist = float('inf')
        for target_idx in target_nodes:
            dist = self._calculate_distance(current_node, target_idx)
            min_dist = min(min_dist, dist)
        
        return min_dist / 100.0  # Normalize by 100km
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector according to specification"""
        obs = []
        
        # 1. One-hot position (N dimensions)
        position_onehot = np.zeros(self.n_nodes)
        if self.current_node is not None:
            position_onehot[self.current_node] = 1.0
        obs.extend(position_onehot)
        
        # 2. Battery level normalized [0,1]
        obs.append(self.battery / self.max_battery)
        
        # 3. Payload normalized [0,1]
        obs.append(self.payload / 3.0)  # assuming max payload is 3
        
        # 4. Flags (2 dimensions)
        obs.append(1.0 if self.pickup_done else 0.0)
        obs.append(1.0 if self.delivery_done else 0.0)
        
        # 5. Guide distances (3 dimensions)
        if self.current_node is not None:
            d_pickup = self._distance_to_closest_type(self.current_node, 'pickup')
            d_delivery = self._distance_to_closest_type(self.current_node, 'delivery')
            d_hub = self._distance_to_closest_type(self.current_node, 'hubs')
        else:
            d_pickup = d_delivery = d_hub = 0.0
        
        obs.extend([d_pickup, d_delivery, d_hub])
        
        # 6. Recharge count normalized
        obs.append(self.recharge_count / 10.0)
        
        # 7. Local adjacency (k*4 dimensions)
        if self.current_node is not None:
            adj_info = self.adjacency[self.current_node]
        else:
            adj_info = [{'target': -1, 'distance': 0, 'cost': 0}] * self.k_neighbors
        
        for edge_info in adj_info:
            if edge_info['target'] == -1:
                # Invalid edge
                obs.extend([0.0, 0.0, 0.0, 0.0])
            else:
                target_idx = edge_info['target'] / self.n_nodes  # normalized target
                cost_norm = edge_info['cost'] / 160.0  # normalize by max cost
                dist_norm = edge_info['distance'] / 160.0  # normalize by max distance
                type_id = self._get_node_type_id(edge_info['target']) / 3.0  # normalize type
                obs.extend([target_idx, cost_norm, dist_norm, type_id])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_action_mask(self) -> np.ndarray:
        """Get action mask for valid actions"""
        if self.at_step_zero:
            # Step 0: only hubs are valid
            mask = np.zeros(self.action_space.n, dtype=bool)
            mask[:len(self.hubs)] = True
            return mask        
        else:
            # Step 1+: only valid outgoing edges
            mask = np.zeros(self.action_space.n, dtype=bool)
            if self.current_node is not None:
                adj_info = self.adjacency[self.current_node]
                for i, edge_info in enumerate(adj_info):
                    if edge_info['target'] != -1:
                        mask[i] = True
            return mask
    
    def _calculate_battery_consumption(self, edge_cost: float) -> float:
        """Calculate battery consumption for an edge according to formula"""
        return (edge_cost / self.k_norm) * (1 + self.alpha * (self.payload - 1))
    
    def _select_random_targets(self):
        """Select random pickup and delivery targets"""
        if self.pickups and self.deliveries:
            self.pickup_target = random.choice(self.pickups)
            self.delivery_target = random.choice(self.deliveries)
        else:
            # Fallback if no pickup/delivery nodes
            self.pickup_target = random.choice(range(self.n_nodes))
            self.delivery_target = random.choice(range(self.n_nodes))
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        if seed is not None:
            self.seed(seed)
        
        self.current_node = None  # Will be set at step 0
        
        # Randomize battery if enabled
        if self.randomize_battery:
            self.battery = random.randint(self.battery_range[0], self.battery_range[1])
        else:
            self.battery = self.battery_init
        
        # Randomize payload if enabled
        if self.randomize_payload:
            self.payload = random.randint(self.payload_range[0], self.payload_range[1])
        else:
            self.payload = self.payload_init
        
        self.pickup_done = False
        self.delivery_done = False
        self.recharge_count = 0
        self.step_count = 0
        self.at_step_zero = True
        
        # Select random targets
        self._select_random_targets()
        
        info = {
            'battery_init': self.battery,
            'payload_init': self.payload,
            'pickup_target': self.pickup_target,
            'delivery_target': self.delivery_target,
            'randomized': {
                'battery': self.randomize_battery,
                'payload': self.randomize_payload
            }
        }
        
        return self._get_observation(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        reward = 0.0
        done = False
        truncated = False
        info = {}
        
        # Action validation
        mask = self._get_action_mask()
        if not mask[action]:
            # Invalid action
            reward = -1.0
            done = True
            info['termination_reason'] = 'invalid_action'
            return self._get_observation(), reward, done, truncated, info
        
        if self.at_step_zero:
            # Step 0: teleport to chosen hub
            hub_idx = action
            self.current_node = self.hubs[hub_idx]
            self.at_step_zero = False            
            info['action_type'] = 'teleport_hub'
            info['hub_chosen'] = self.current_node
        else:
            # Step 1+: move along edge
            prev_node = self.current_node
            adj_info = self.adjacency[self.current_node]
            edge_info = adj_info[action]
            
            if edge_info['target'] == -1:
                # Should not happen due to masking, but safety check
                reward = -1.0
                done = True
                info['termination_reason'] = 'invalid_edge'
                return self._get_observation(), reward, done, truncated, info
            
            self.current_node = edge_info['target']
            edge_cost = edge_info['cost']
            
            # Step reward: negative normalized cost
            reward += -edge_cost / 160.0
              # Battery consumption
            battery_consumption = self._calculate_battery_consumption(edge_cost)
            self.battery -= battery_consumption
            
            # Check battery depletion
            if self.battery <= 0:
                reward = -1.0
                done = True
                info['termination_reason'] = 'battery_depleted'
                return self._get_observation(), reward, done, truncated, info
            
            info['action_type'] = 'move'
            info['edge_cost'] = edge_cost
            info['battery_consumption'] = battery_consumption
        
        # Check current node for special actions
        current_type = self._get_node_type(self.current_node)
        
        # Pickup logic
        if current_type == 'pickup' and not self.pickup_done:
            self.pickup_done = True
            reward += 0.5
            info['pickup_completed'] = True
        
        # Delivery logic
        if current_type == 'delivery' and self.pickup_done and not self.delivery_done:
            self.delivery_done = True
            self.payload = 0  # Drop off cargo
            reward += 0.5
            info['delivery_completed'] = True
        
        # Recharge logic
        if current_type == 'charging':
            old_battery = self.battery
            self.battery = min(self.battery + self.recharge_step, self.max_battery)
            if self.battery > old_battery:
                self.recharge_count += 1
                reward -= self.recharge_penalty
                info['recharged'] = True
                info['recharge_amount'] = self.battery - old_battery
        
        # Mission completion check
        if self.pickup_done and self.delivery_done and current_type == 'hubs':
            reward += 1.0
            done = True
            info['termination_reason'] = 'mission_completed'
            info['success'] = True
        
        # Step limit check
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            info['termination_reason'] = 'max_steps_reached'
            info['step_count'] = self.step_count
            info['battery'] = self.battery
            info['payload'] = self.payload
            info['pickup_done'] = self.pickup_done
            info['delivery_done'] = self.delivery_done
            info['recharge_count'] = self.recharge_count
        
        return self._get_observation(), reward, done, truncated, info
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step {self.step_count}: Node {self.current_node} ({self._get_node_type(self.current_node) if self.current_node is not None else 'None'})")
            print(f"  Battery: {self.battery:.1f}/{self.max_battery}")
            print(f"  Payload: {self.payload}")
            print(f"  Pickup: {self.pickup_done}, Delivery: {self.delivery_done}")
            print(f"  Recharges: {self.recharge_count}")
    
    def get_action_meanings(self):
        """Return human-readable action meanings"""
        if self.at_step_zero:
            return [f"Teleport to hub {i}" for i in range(len(self.hubs))]
        else:
            meanings = []
            if self.current_node is not None:
                adj_info = self.adjacency[self.current_node]
                for i, edge_info in enumerate(adj_info):
                    if edge_info['target'] != -1:
                        target_type = self._get_node_type(edge_info['target'])
                        meanings.append(f"Move to {target_type} node {edge_info['target']} (cost: {edge_info['cost']:.1f})")
                    else:
                        meanings.append("Invalid action")
            return meanings


def test_full_environment():
    """Test the full environment"""
    import os
    
    # Get absolute path to graph file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(script_dir, "..", "data", "graph.json")
    
    print("🚁 Testing Full PPO Drone Delivery Environment")
    print("=" * 55)
    
    try:
        # Test with different parameters
        env = DroneDeliveryFullEnv(graph_path, battery_init=90, payload_init=2)
        
        print("✅ Environment created successfully")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.n}")
        
        # Test reset
        obs = env.reset()
        print(f"   Initial observation shape: {obs.shape}")
        print(f"   At step zero: {env.at_step_zero}")
        
        # Test step 0 (hub selection)
        print("\n📍 Testing step 0 (hub selection):")
        action_mask = env._get_action_mask()
        valid_actions = np.where(action_mask)[0]
        print(f"   Valid actions (hubs): {len(valid_actions)}")
        
        action = valid_actions[0]  # Choose first valid hub
        obs, reward, done, info = env.step(action)
        print(f"   Step 0: Action={action}, Reward={reward:.3f}, Done={done}")
        print(f"   Info: {info}")
        
        # Test a few more steps
        for step in range(1, 4):
            action_mask = env._get_action_mask()
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                obs, reward, done, info = env.step(action)
                print(f"   Step {step}: Action={action}, Reward={reward:.3f}, Done={done}")
                if done:
                    print(f"   Termination: {info.get('termination_reason', 'unknown')}")
                    break
            else:
                print(f"   Step {step}: No valid actions available")
                break
        
        print("\n🎉 Full environment test completed successfully!")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_full_environment()
