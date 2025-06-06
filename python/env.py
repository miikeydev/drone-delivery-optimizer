# Complete PPO environment for drone delivery with action masking support
import gymnasium as gym
import numpy as np
import json
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from gymnasium import spaces
import torch
import torch.nn as nn


class DroneDeliveryFullEnv(gym.Env):
    """
    Complete PPO environment for drone delivery with action masking support.
    """
    def __init__(self, graph_path: str, battery_init: int = 100, payload_init: int = 1, 
             max_battery: int = 100, k_neighbors: int = 10, max_steps: int = 100,
             randomize_battery: bool = False, battery_range: tuple = (60, 100),
             randomize_payload: bool = False, payload_range: tuple = (1, 5),
             use_curriculum: bool = True, curriculum_threshold: float = 0.25):
        super(DroneDeliveryFullEnv, self).__init__()
        
        # Load graph data
        with open(graph_path, 'r') as f:
            self.graph_data = json.load(f)
        
        self.nodes = self.graph_data['nodes']
        self.edges = self.graph_data['edges']
        self.n_nodes = len(self.nodes)
        self.max_battery = max_battery
        self.battery_init = battery_init
        self.payload_init = payload_init
        self.max_steps = max_steps
        
        # Build adjacency structure for fast lookup
        self.adjacency = self._build_adjacency()
        
        self.k_neighbors = 10
        self.max_degree = 10
        
        print(f"✅ FIXED K=10 structure: every node → exactly 10 neighbors")
        print(f"Graph structure: enforced k_neighbors={self.k_neighbors}")
        
        # Randomization parameters
        self.randomize_battery = randomize_battery
        self.battery_range = battery_range
        self.randomize_payload = randomize_payload
        self.payload_range = payload_range
        
        # FIXED: Battery consumption parameters - much more aggressive
        self.k_norm = 1.2  
        self.alpha = 0.4   
        
        # REBALANCED REWARDS - stronger signals
        self.move_penalty = 0.0           # CHANGED: No penalty for movement
        self.recharge_penalty = 0.1       # CHANGED: Small penalty for recharge
        self.pickup_reward = 1.0          # CHANGED: +1 for pickup
        self.delivery_reward = 2.0        # CHANGED: +2 for delivery
        self.battery_empty_penalty = -1.0  # CHANGED: -1 for battery empty
        
        # Recharge parameters
        self.recharge_step = 30
        self.recharge_good_threshold = 0.20
        self.MAX_RECHARGES = 5
        
        # Find node indices by type
        self.hubs = [i for i, node in enumerate(self.nodes) if node['type'] == 'hubs']
        self.pickups = [i for i, node in enumerate(self.nodes) if node['type'] == 'pickup']
        self.deliveries = [i for i, node in enumerate(self.nodes) if node['type'] == 'delivery']
        self.charging_stations = [i for i, node in enumerate(self.nodes) if node['type'] == 'charging']
        
        # Node type mappings
        self.type_to_id = {'hubs': 0, 'pickup': 1, 'delivery': 2, 'charging': 3}
        
        # UPDATED: Observation space without node embedding
        state_dim = 1 + 1 + 2 + 1 + 1 + (10 * 8)  # FIXED: 86 dims total (no embedding)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(state_dim,), dtype=np.float32)
        
        # FIXED: Action space is always 11 (10 moves + 1 stay)
        self.STAY_ACTION = 10  # Action index 10 = stay
        self.action_space = spaces.Discrete(11)  # 0-9=move, 10=stay
        
        # Cache for shortest paths (computed once per episode)
        self.dist_to_pickup = None
        self.dist_to_delivery = None
        
        # Environment state
        self.current_node = None
        self.battery = self.battery_init
        self.payload = 0
        self.pickup_done = False
        self.delivery_done = False
        self.pickup_target = None
        self.delivery_target = None
        self.recharge_count = 0
        self.step_count = 0
        self.stayed_last_step = False  # NEW: Prevent stay spam
        
        # Episode tracking for logging
        self.episode_path = []
        self.episode_actions = []
        self.episode_rewards = []
        self.total_reward = 0
        
        # Curriculum learning parameters
        self.use_curriculum = use_curriculum
        self.curriculum_threshold = 0.25  # CHANGED: 0.15 -> 0.25
        self.curriculum_level = 0
        self.max_curriculum_level = 4
        self.episode_successes = []
        self.curriculum_window = 50
        
        # NEW: Curriculum parameters for battery and payload
        self.curriculum_battery_levels = [
            (40, 60),    # Level 0: Constrained battery (immediate pressure)
            (35, 55),    # Level 1: Tighter constraints
            (30, 50),    # Level 2: Low battery
            (25, 45),    # Level 3: Very low battery
            (20, 40)     # Level 4: Extreme battery challenge
        ]
        
        self.curriculum_payload_levels = [
            (1, 1),      # Level 0: Light payload (easy)
            (1, 2),      # Level 1: Light to medium
            (1, 3),      # Level 2: Light to heavy
            (2, 4),      # Level 3: Medium to heavy
            (2, 5)       # Level 4: Heavy payload (hard)
        ]
        
        # Calculate pickup-delivery distances for curriculum
        self._calculate_pickup_delivery_distances()
        
        print(f"Initialized environment: {self.n_nodes} nodes, k_neighbors={self.k_neighbors}")
        print(f"Action space: {self.action_space.n} (0-9=move to neighbor, 10=stay)")
        print(f"Observation space: {state_dim} dims (K=10 fixed structure)")
        print(f"MAX_RECHARGES={self.MAX_RECHARGES}, recharge_step={self.recharge_step}")
        print(f"Enhanced: K=10 regularity + explicit direction indicators")

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the environment's random number generator."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self._np_random = np.random.RandomState(seed)
        else:
            seed = np.random.randint(0, 2**32 - 1)
        return [seed]
    
    def _build_adjacency(self) -> Dict[int, List[Tuple[int, float]]]:
        """Build K=10 outgoing adjacency (directed) for consistent observation structure."""
        # First, collect all undirected edges
        undirected = {i: [] for i in range(self.n_nodes)}
        for edge in self.edges:
            # Handle both frontend format (u, v, dist) and training format (source, target, distance)
            if 'source' in edge and 'target' in edge:
                source = edge['source']
                target = edge['target']
                distance = edge.get('distance', edge.get('dist', 0))
            elif 'u' in edge and 'v' in edge:
                source = edge['u']
                target = edge['v']
                distance = edge.get('dist', edge.get('distance', 0))
            else:
                print(f"⚠️ Warning: Unknown edge format: {edge}")
                continue
                
            # Add both directions to undirected graph first
            undirected[source].append((target, distance))
            undirected[target].append((source, distance))
        
        # FIXED: Now create directed K=10 outgoing adjacency
        directed_adjacency = {i: [] for i in range(self.n_nodes)}
        
        for node_id in range(self.n_nodes):
            neighbors = undirected[node_id]
            neighbors_sorted = sorted(neighbors, key=lambda x: x[1])
            
            if len(neighbors_sorted) >= 10:
                selected_neighbors = neighbors_sorted[:10]
            else:
                selected_neighbors = neighbors_sorted[:]
                if neighbors_sorted:
                    closest = neighbors_sorted[0]
                    while len(selected_neighbors) < 10:
                        selected_neighbors.append(closest)
                else:
                    selected_neighbors = [(node_id, 0.0)] * 10
            
            directed_adjacency[node_id] = selected_neighbors
            assert len(directed_adjacency[node_id]) == 10
        
        return directed_adjacency
    
    def _get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        """Get exactly K=10 neighbors for consistent structure."""
        if node_id not in self.adjacency:
            return [(node_id, 0.0)] * 10  # Fallback: 10 self-loops
        
        neighbors = self.adjacency[node_id]
        
        # ASSERT: Should always be exactly 10 due to _build_adjacency
        assert len(neighbors) == 10, f"Expected 10 neighbors, got {len(neighbors)} for node {node_id}"
        
        return neighbors  # Already sorted by distance in _build_adjacency

    def _calculate_distances(self) -> Tuple[float, float, float]:
        """Calculate distances to closest pickup, delivery, and charging station."""
        if self.current_node is None:
            return 1.0, 1.0, 1.0  # Max normalized distance
        
        # BFS to find shortest paths
        def bfs_distance(targets: List[int]) -> float:
            if not targets:
                return 1.0
            
            queue = [(self.current_node, 0)]
            visited = {self.current_node}
            
            while queue:
                node, dist = queue.pop(0)
                if node in targets:
                    return min(1.0, dist / 100.0)  # Normalize to [0,1]
                
                for neighbor, edge_dist in self.adjacency.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + edge_dist))
            
            return 1.0  # If no path found
        
        # Calculate distances based on current state
        if not self.pickup_done:
            pickup_dist = bfs_distance(self.pickups)
        else:
            pickup_dist = 0.0  # Already picked up
            
        if self.pickup_done and not self.delivery_done:
            delivery_dist = bfs_distance(self.deliveries)
        else:
            delivery_dist = 1.0 if not self.delivery_done else 0.0
            
        charging_dist = bfs_distance(self.charging_stations)
        
        return pickup_dist, delivery_dist, charging_dist

    def _dijkstra_from_target(self, target_node: int) -> Dict[int, float]:
        """Calculate shortest distances from target to all nodes using Dijkstra."""
        if target_node is None:
            return {i: float('inf') for i in range(self.n_nodes)}
        
        import heapq
        
        # Initialize distances
        distances = {i: float('inf') for i in range(self.n_nodes)}
        distances[target_node] = 0.0
        
        # Priority queue: (distance, node)
        pq = [(0.0, target_node)]
        visited = set()
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            # Check all neighbors
            for neighbor, edge_weight in self.adjacency.get(current_node, []):
                if neighbor not in visited:
                    new_dist = current_dist + edge_weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))
        
        return distances

    def _get_observation(self) -> np.ndarray:
        """Get observation with K=10 fixed structure."""
        
        # Normalized battery and payload
        battery_norm = self.battery / self.max_battery
        payload_norm = self.payload / 5.0  # FIXED: Max payload is 5, not 10
        
        # Task flags
        pickup_flag = 1.0 if self.pickup_done else 0.0
        delivery_flag = 1.0 if self.delivery_done else 0.0
        
        # Recharge count (normalized)
        recharge_norm = min(1.0, self.recharge_count / 5.0)
        
        # Global delivery distance signal
        delivery_dist_global = 0.0
        if self.pickup_done and not self.delivery_done and self.dist_to_delivery and self.current_node is not None:
            cur_to_delivery = self.dist_to_delivery.get(self.current_node, float('inf'))
            if cur_to_delivery != float('inf'):
                delivery_dist_global = min(1.0, cur_to_delivery / 100.0)
        
        # FIXED: K=10 local adjacency (always 80 dims = 10 neighbors * 8 features)
        local_adj = np.zeros(80, dtype=np.float32)  # FIXED: 10*8 = 80 always
        if self.current_node is not None:
            neighbors = self._get_neighbors(self.current_node)
            
            # Get current distances to targets
            cur_to_pickup = self.dist_to_pickup.get(self.current_node, float('inf')) if self.dist_to_pickup else float('inf')
            cur_to_delivery = self.dist_to_delivery.get(self.current_node, float('inf')) if self.dist_to_delivery else float('inf')
            
            # FIXED: Process exactly 10 neighbors
            for i in range(10):
                neighbor_id, edge_distance = neighbors[i]
                base_idx = i * 8
                
                # [0-3] Node type (one-hot encoding)
                node_type = self.nodes[neighbor_id]['type']
                type_id = self.type_to_id.get(node_type, 0)
                local_adj[base_idx + type_id] = 1.0
                
                # [4] Pickup target indicator
                if neighbor_id == self.pickup_target and not self.pickup_done:
                    local_adj[base_idx + 4] = 1.0
                    
                # [5] Delivery target indicator  
                if neighbor_id == self.delivery_target and self.pickup_done and not self.delivery_done:
                    local_adj[base_idx + 5] = 1.0
                
                # [6] Direction indicator (+1=closer, -1=farther, 0=neutral)
                direction_indicator = 0.0
                
                if not self.pickup_done and self.dist_to_pickup:
                    nbr_to_pickup = self.dist_to_pickup.get(neighbor_id, float('inf'))
                    if cur_to_pickup != float('inf') and nbr_to_pickup != float('inf'):
                        if nbr_to_pickup < cur_to_pickup:
                            direction_indicator = 1.0
                        elif nbr_to_pickup > cur_to_pickup:
                            direction_indicator = -1.0
                        
                elif self.pickup_done and not self.delivery_done and self.dist_to_delivery:
                    nbr_to_delivery = self.dist_to_delivery.get(neighbor_id, float('inf'))
                    if cur_to_delivery != float('inf') and nbr_to_delivery != float('inf'):
                        if nbr_to_delivery < cur_to_delivery:
                            direction_indicator = 1.0
                        elif nbr_to_delivery > cur_to_delivery:
                            direction_indicator = -1.0
                
                local_adj[base_idx + 6] = direction_indicator
                
                # [7] FIXED: Edge duplicate flag (1.0 if this is a duplicate destination)
                unique_destinations = set()
                for j in range(i):
                    other_neighbor_id, _ = neighbors[j]
                    unique_destinations.add(other_neighbor_id)
                
                is_duplicate = 1.0 if neighbor_id in unique_destinations else 0.0
                local_adj[base_idx + 7] = is_duplicate

        # Fixed observation structure: 6 global + 80 local
        local_features = np.concatenate([
            [battery_norm],             # 1 dimension
            [payload_norm],             # 1 dimension (payload/5.0)
            [pickup_flag, delivery_flag], # 2 dimensions
            [recharge_norm],            # 1 dimension
            [delivery_dist_global],     # 1 dimension
            local_adj                   # 80 dimensions (K=10 * 8 features)
        ])
        
        return local_features.astype(np.float32)
    
    def _closest_hub(self, target_id: int) -> int:
        """Return the hub closest to target_id using BFS."""
        from collections import deque
        
        visited = {target_id}
        queue = deque([(target_id, 0)])
        
        while queue:
            node, dist = queue.popleft()
            if node in self.hubs:
                return node
            
            for neighbor, _ in self.adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        # Fallback (should never be reached in connected graph)
        return random.choice(self.hubs) if self.hubs else 0
    
    def _get_action_mask(self) -> np.ndarray:
        """Get mask for valid actions with K=10 structure."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        if self.current_node is not None:
            # Movement actions (0-9 for K=10 neighbors)
            neighbors = self._get_neighbors(self.current_node)
            
            # FIXED: Allow ALL actions with sufficient battery, including duplicates
            for i in range(10):  # FIXED: exactly 10 neighbors always
                neighbor_id, distance = neighbors[i]
                battery_cost = self._calculate_battery_cost(distance)
                
                # FIXED: Only check battery cost, allow duplicates
                if self.battery >= battery_cost:
                    mask[i] = True
        
            # STAY action (index 10)
            current_node_type = self.nodes[self.current_node]['type']
            mask[10] = (  # FIXED: action 10 = stay
                current_node_type == 'charging' and
                not self.stayed_last_step and
                self.recharge_count < self.MAX_RECHARGES
            )
    
        return mask
    
    def action_masks(self) -> np.ndarray:
        """Get action mask (for MaskablePPO compatibility)."""
        return self._get_action_mask()
    
    def _calculate_battery_cost(self, distance: float) -> float:
        """Calculate battery cost for moving a certain distance."""
        return distance / self.k_norm * (1 + self.alpha * self.payload)
    
    def _calculate_pickup_delivery_distances(self):
        """Calculate distances between all pickup-delivery pairs for curriculum."""
        self.pickup_delivery_pairs = []
        
        for pickup_idx in self.pickups:
            for delivery_idx in self.deliveries:
                distance = self._bfs_distance_between_nodes(pickup_idx, delivery_idx)
                self.pickup_delivery_pairs.append({
                    'pickup': pickup_idx,
                    'delivery': delivery_idx,
                    'distance': distance
                })
        
        self.pickup_delivery_pairs.sort(key=lambda x: x['distance'])
        
        total_pairs = len(self.pickup_delivery_pairs)
        pairs_per_level = max(1, total_pairs // (self.max_curriculum_level + 1))
        
        self.curriculum_pairs = []
        for level in range(self.max_curriculum_level + 1):
            start_idx = 0
            end_idx = min(total_pairs, (level + 1) * pairs_per_level)
            if level == self.max_curriculum_level:
                end_idx = total_pairs
            
            level_pairs = self.pickup_delivery_pairs[start_idx:end_idx]
            self.curriculum_pairs.append(level_pairs)
            
            # SIMPLIFIED: Only print curriculum setup once
            if level == 0:
                print(f"Curriculum Level {level}: {len(level_pairs)} pairs, "
                      f"distance range: {level_pairs[0]['distance']:.1f}-{level_pairs[-1]['distance']:.1f}")
            elif level == self.max_curriculum_level:
                print(f"Curriculum Level {level}: {len(level_pairs)} pairs, "
                      f"distance range: {level_pairs[0]['distance']:.1f}-inf")

    def _bfs_distance_between_nodes(self, start_node: int, target_node: int) -> float:
        """Calculate shortest distance between two nodes using BFS."""
        if start_node == target_node:
            return 0.0
        
        from collections import deque
        
        queue = deque([(start_node, 0.0)])
        visited = {start_node}
        
        while queue:
            current_node, current_distance = queue.popleft()
            
            # Check all neighbors
            for neighbor_id, edge_distance in self.adjacency.get(current_node, []):
                if neighbor_id == target_node:
                    return current_distance + edge_distance
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, current_distance + edge_distance))
        
        # If no path found, return a large distance
        return float('inf')

    def _select_targets(self):
        """Select pickup and delivery targets based on curriculum level."""
        if not self.use_curriculum or not hasattr(self, 'curriculum_pairs'):
            # Random selection if curriculum disabled or not ready
            self.pickup_target = random.choice(self.pickups) if self.pickups else None
            self.delivery_target = random.choice(self.deliveries) if self.deliveries else None
            self.current_pair_distance = 0
            return
        
        # Get pairs for current curriculum level
        if self.curriculum_level < len(self.curriculum_pairs):
            available_pairs = self.curriculum_pairs[self.curriculum_level]
        else:
            available_pairs = self.curriculum_pairs[-1]  # Use hardest level
        
        if not available_pairs:
            # Fallback to random
            self.pickup_target = random.choice(self.pickups) if self.pickups else None
            self.delivery_target = random.choice(self.deliveries) if self.deliveries else None
            self.current_pair_distance = 0
            return
        
        # Select random pair from current level
        selected_pair = random.choice(available_pairs)
        self.pickup_target = selected_pair['pickup']
        self.delivery_target = selected_pair['delivery']
        self.current_pair_distance = selected_pair['distance']

    def _get_curriculum_battery_range(self) -> tuple:
        """Get battery range for current curriculum level."""
        if not self.use_curriculum or self.curriculum_level >= len(self.curriculum_battery_levels):
            return self.battery_range
        return self.curriculum_battery_levels[self.curriculum_level]

    def _get_curriculum_payload_range(self) -> tuple:
        """Get payload range for current curriculum level."""
        if not self.use_curriculum or self.curriculum_level >= len(self.curriculum_payload_levels):
            return self.payload_range
        return self.curriculum_payload_levels[self.curriculum_level]

    def _update_curriculum(self, success: bool):
        """Update curriculum level based on recent performance - SILENT mode."""
        if not self.use_curriculum:
            return
        
        self.episode_successes.append(success)
        if len(self.episode_successes) > self.curriculum_window:
            self.episode_successes.pop(0)
        
        if len(self.episode_successes) < min(10, self.curriculum_window // 3):
            return
        
        recent_success_rate = sum(self.episode_successes) / len(self.episode_successes)
        
        # ONLY INCREASE difficulty if doing well - NO DOWNGRADE
        if (recent_success_rate >= self.curriculum_threshold and 
            self.curriculum_level < self.max_curriculum_level):
            old_level = self.curriculum_level
            self.curriculum_level += 1
            
            # SILENT curriculum advancement - no print spam
            self.episode_successes = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if self.use_curriculum:
            self.pickup_target = None
            self.delivery_target = None
        
        self._select_targets()
        
        self.dist_to_pickup = self._dijkstra_from_target(self.pickup_target)
        self.dist_to_delivery = self._dijkstra_from_target(self.delivery_target)
        
        if self.pickup_target is not None:
            self.current_node = self._closest_hub(self.pickup_target)
        else:
            self.current_node = random.choice(self.hubs) if self.hubs else 0
        
        self.payload = 0
        self.step_count = 0
        self.pickup_done = False
        self.delivery_done = False
        self.recharge_count = 0
        
        # FIXED: Battery selon curriculum ou randomisation
        if self.randomize_battery or self.use_curriculum:
            if self.use_curriculum:
                curriculum_battery_range = self._get_curriculum_battery_range()
                self.battery = random.randint(*curriculum_battery_range)
            else:
                self.battery = random.randint(*self.battery_range)
        else:
            self.battery = self.battery_init
        
        # FIXED: Payload selon curriculum - UNE valeur fixe, pas random
        if self.use_curriculum:
            curriculum_payload_range = self._get_curriculum_payload_range()
            # Prendre la valeur MAXIMALE du range pour ce niveau (difficulté croissante)
            self.payload_init = curriculum_payload_range[1]
        elif self.randomize_payload:
            self.payload_init = random.randint(*self.payload_range)
        # sinon garde self.payload_init du constructeur
        
        self.last_dist_to_pickup = self.dist_to_pickup.get(self.current_node, float('inf')) if self.dist_to_pickup else float('inf')
        self.last_dist_to_delivery = self.dist_to_delivery.get(self.current_node, float('inf')) if self.dist_to_delivery else float('inf')
        
        self.episode_path = [self.current_node]
        self.episode_actions = []
        self.episode_rewards = []
        self.total_reward = 0
        self.stayed_last_step = False
        
        observation = self._get_observation()
        info = {
            'step': self.step_count,
            'battery': self.battery,
            'payload': self.payload,
            'payload_init': self.payload_init,  # Info sur le payload qui sera chargé
            'pickup_target': self.pickup_target,
            'delivery_target': self.delivery_target,
            'spawn_hub': self.current_node,
            'curriculum_level': self.curriculum_level,
            'pair_distance': getattr(self, 'current_pair_distance', 0),
            'curriculum_success_rate': (sum(self.episode_successes) / len(self.episode_successes) 
                                      if self.episode_successes else 0.0),
            'curriculum_battery_range': self._get_curriculum_battery_range(),
            'curriculum_payload_range': self._get_curriculum_payload_range(),
            'pickup_distance': self.dist_to_pickup.get(self.current_node, float('inf')) if self.dist_to_pickup else float('inf'),
            'delivery_distance': self.dist_to_delivery.get(self.current_node, float('inf')) if self.dist_to_delivery else float('inf')
        }
        
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step with simplified reward structure."""
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        self.episode_actions.append(action)
        
        if self.current_node is None:
            reward = -1.0
            terminated = True
            info['termination_reason'] = 'no_current_node'
            info['action_type'] = 'error'
        
        # Handle STAY action (action=10)
        elif action == 10:
            self.stayed_last_step = True
            current_node_type = self.nodes[self.current_node]['type']
            
            if current_node_type != 'charging':
                reward = -0.5  # Invalid stay action
                info['action_type'] = 'stay_invalid'
            else:
                # CHANGED: Simple recharge penalty
                reward = -self.recharge_penalty  # -0.1
                
                old_battery = self.battery
                self.battery = min(self.max_battery, self.battery + self.recharge_step)
                self.recharge_count += 1
                
                info['action_type'] = 'stay_recharge'
                info['battery_gained'] = self.battery - old_battery
                info['recharge_count'] = self.recharge_count
            
            self.step_count += 1
            
        # MOVEMENT actions (0-9 for K=10 neighbors)
        elif 0 <= action <= 9:
            self.stayed_last_step = False
            neighbors = self._get_neighbors(self.current_node)
            
            next_node, distance = neighbors[action]
            battery_cost = self._calculate_battery_cost(distance)
            
            if self.battery >= battery_cost:
                # Execute movement
                self.battery -= battery_cost
                old_node = self.current_node
                self.current_node = next_node
                self.episode_path.append(next_node)
                
                reward = self.move_penalty  # CHANGED: 0.0 (no penalty for movement)
                
                # REMOVED: All progress rewards - too complex
                
                # CHANGED: Simple target rewards
                if next_node == self.pickup_target and not self.pickup_done:
                    self.pickup_done = True
                    self.payload = self.payload_init
                    reward += self.pickup_reward  # +1.0
                    info['payload_loaded'] = self.payload
                    
                elif next_node == self.delivery_target and self.pickup_done and not self.delivery_done:
                    self.delivery_done = True
                    self.payload = 0
                    reward += self.delivery_reward  # +2.0
                    info['delivery_completed'] = True
                    
                info['action_type'] = 'move'
                info['distance'] = distance
                info['battery_cost'] = battery_cost
                
            else:
                reward = -0.5  # Insufficient battery
                info['action_type'] = 'move_insufficient_battery'
            
            self.step_count += 1
            
        else:
            # Invalid action
            reward = -1.0
            terminated = True
            info['termination_reason'] = 'invalid_action_index'
            info['action_type'] = 'invalid_action'
            self.step_count += 1
        
        # CHANGED: Simplified termination penalties
        if self.battery <= 0:
            terminated = True
            reward += self.battery_empty_penalty  # -1.0
            info['termination_reason'] = 'battery_depleted'
            
        if self.step_count >= self.max_steps:
            truncated = True
            if self.delivery_done:
                # Completed successfully - no penalty
                pass
            elif self.pickup_done:
                reward -= 1.0  # CHANGED: -1 for timeout with pickup done
            else:
                reward -= 2.0  # CHANGED: -2 for timeout without pickup
            info['termination_reason'] = 'max_steps_reached'

        # Track reward and update info
        self.episode_rewards.append(reward)
        self.total_reward += reward

        battery_used_gross = self.battery_init - self.battery
        if self.recharge_count > 0:
            battery_recharged = self.recharge_count * self.recharge_step
            battery_used_net = max(0, battery_used_gross - battery_recharged)
        else:
            battery_used_net = battery_used_gross

        info.update({
            'step': self.step_count,
            'current_node': self.current_node,
            'battery': self.battery,
            'payload': self.payload,
            'pickup_done': self.pickup_done,
            'delivery_done': self.delivery_done,
            'recharge_count': self.recharge_count,
            'recharge_limit': self.MAX_RECHARGES,
            'total_reward': self.total_reward,
            'success': self.delivery_done,
            'battery_used_net': battery_used_net,
            'stayed_last_step': self.stayed_last_step,
            'episode_path': self.episode_path.copy(),
            'route_description': self._get_route_description()
        })

        if terminated or truncated:
            success = self.delivery_done
            self._update_curriculum(success)

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def _get_mission_progress(self) -> str:
        """Get current mission progress description."""
        if self.delivery_done:
            return "COMPLETED"
        elif self.pickup_done:
            return "PICKUP_DONE_NEED_DELIVERY"
        else:
            return "NEED_PICKUP"

    def render(self, mode: str = 'human') -> Optional[str]:
        """Render the current state."""
        if mode == 'human':
            print(f"Step {self.step_count}: Node {self.current_node}, Battery {self.battery}, "
                  f"Pickup: {self.pickup_done}, Delivery: {self.delivery_done}")
        elif mode == 'rgb_array':
            # Could implement visual rendering here
            pass
        return None

    def close(self):
        """Clean up resources."""
        pass

    def get_episode_info(self) -> dict:
        """Get comprehensive episode information for enhanced logging."""
        # Calculate route description
        route_description = self._get_route_description()
        
        # Calculate efficiency metrics
        total_distance = sum(
            self._calculate_edge_distance(self.episode_path[i-1], self.episode_path[i])
            for i in range(1, len(self.episode_path))
        ) if len(self.episode_path) > 1 else 0
        
        result = {
            'path': self.episode_path.copy(),
            'actions': self.episode_actions.copy(),
            'rewards': self.episode_rewards.copy(),
            'total_reward': self.total_reward,
            'steps': self.step_count,
            'battery': self.battery,
            'battery_used': self.battery_init - self.battery,
            'battery_efficiency': (self.battery_init - self.battery) / max(1, self.step_count),
            'pickup_done': self.pickup_done,
            'delivery_done': self.delivery_done,
            'recharge_count': self.recharge_count,
            'pickup_target': self.pickup_target,
            'delivery_target': self.delivery_target,
            'success': self.delivery_done,
            'total_distance': total_distance,
            'distance_efficiency': total_distance / max(1, self.step_count),
            'route_description': route_description,
            'mission_progress': self._get_mission_progress(),
            'reward_breakdown': self._calculate_reward_breakdown(),
            'curriculum_info': {
                'level': self.curriculum_level,
                'pair_distance': getattr(self, 'current_pair_distance', 0),
                'success_rate': (sum(self.episode_successes) / len(self.episode_successes) 
                               if self.episode_successes else 0.0),
                'use_curriculum': self.use_curriculum,
                'total_pairs_available': len(self.pickup_delivery_pairs) if hasattr(self, 'pickup_delivery_pairs') else 0,
                'battery_range': self._get_curriculum_battery_range(),
                'payload_range': self._get_curriculum_payload_range(),
                'difficulty_progression': {
                    'distances': 'easy→hard' if self.curriculum_level > 0 else 'easy',
                    'battery': 'high→low' if self.curriculum_level > 0 else 'high',
                    'payload': 'light→heavy' if self.curriculum_level > 0 else 'light'
                }
            }
        }
        
        return result

    def _calculate_reward_breakdown(self) -> dict:
        """Calculate detailed reward breakdown."""
        breakdown = {
            'movement_penalty': sum(r for r in self.episode_rewards if -0.01 <= r <= 0),
            'pickup_target_reward': 15.0 if self.pickup_done else 0.0,
            'delivery_target_reward': 200.0 if self.delivery_done else 0.0,  # UPDATED to reflect massive reward
            'wrong_station_penalties': 0.0,
            'delivery_without_pickup_penalty': sum(r for r in self.episode_rewards if r == -3.0),
            'recharge_penalty': -0.1 * self.recharge_count,
            'battery_depletion_penalty': 0.0,
            'timeout_penalty': 0.0
        }
        
        # Check final penalties
        if len(self.episode_rewards) > 0:
            final_reward = self.episode_rewards[-1]
            if final_reward <= -15.0:
                breakdown['battery_depletion_penalty'] = final_reward
            elif final_reward <= -10.0:
                breakdown['timeout_penalty'] = final_reward
                
        return breakdown

    def _get_route_description(self) -> str:
        """Get human-readable route description."""
        if not self.episode_path:
            return "No movement"
        
        descriptions = []
        for node_id in self.episode_path:
            node = self.nodes[node_id]
            node_type = node['type']
            
            if node_type == 'hubs':
                descriptions.append(f"Hub{node_id}")
            elif node_type == 'pickup':
                if node_id == self.pickup_target:
                    descriptions.append(f"PICKUP{node_id}*")
                else:
                    descriptions.append(f"pickup{node_id}")
            elif node_type == 'delivery':
                if node_id == self.delivery_target:
                    descriptions.append(f"DELIVERY{node_id}*")
                else:
                    descriptions.append(f"delivery{node_id}")
            elif node_type == 'charging':
                descriptions.append(f"Charge{node_id}")
            else:
                descriptions.append(f"Node{node_id}")
        
        return " → ".join(descriptions)

    def _calculate_edge_distance(self, node1: int, node2: int) -> float:
        """Calculate distance between two connected nodes."""
        if node1 in self.adjacency:
            for neighbor_id, distance in self.adjacency[node1]:
                if neighbor_id == node2:
                    return distance
        return 0.0


# Test function
if __name__ == "__main__":
    print("Testing Enhanced Drone Delivery Environment...")
    
    try:
        graph_path = "data/graph.json"
        env = DroneDeliveryFullEnv(
            graph_path=graph_path,
            battery_init=80,
            payload_init=2,
            k_neighbors=10,
            max_steps=100,
            randomize_payload=True,
            payload_range=(1, 5)
        )
        
        # Test reset
        obs, info = env.reset()
        print(f"Reset successful. Observation shape: {obs.shape}")
        print(f"   Spawn hub: {info['spawn_hub']}")
        print(f"   Pickup target: {info['pickup_target']}")
        print(f"   Delivery target: {info['delivery_target']}")
        print(f"   Initial payload: {env.payload} (should be 0)")
        
        # Test action masking
        action_mask = env._get_action_mask()
        valid_actions = np.where(action_mask)[0]
        print(f"Action masking works. Valid actions: {valid_actions}")
        
        # Test a few steps
        for step in range(1, 6):
            action_mask = env._get_action_mask()
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step {step}: Action={action}, Reward={reward:.3f}, Payload={env.payload}")
                print(f"   Action type: {info.get('action_type', 'move')}")
                print(f"   Current node: {env.current_node}")
                print(f"   Mission progress: {info.get('mission_progress', 'unknown')}")
                if 'payload_loaded' in info:
                    print(f"   Pickup completed! Loaded {info['payload_loaded']} kg")
                if terminated or truncated:
                    print(f"   Termination: {info.get('termination_reason', 'unknown')}")
                    break
            else:
                print(f"Step {step}: No valid actions available")
                break
        
        # Test episode info
        episode_info = env.get_episode_info()
        print(f"Episode info: {episode_info['route_description']}")
        print(f"   Success: {episode_info['success']}, Steps: {episode_info['steps']}")
        print(f"   Mission: {episode_info['mission_progress']}")
        print(f"   Reward breakdown: {episode_info['reward_breakdown']}")
        
        print("\nEnhanced environment test completed successfully!")
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
