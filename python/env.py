# Complete PPO environment for drone delivery with action masking support
import gymnasium as gym
import numpy as np
import json
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from gymnasium import spaces


class DroneDeliveryFullEnv(gym.Env):
    """
    Complete PPO environment for drone delivery with action masking support.
    Implements the full specification with MaskablePPO compatibility:
    - UI parameters (battery_init, payload_init)
    - Direct hub spawning near pickup (no step 0 teleport)
    - Proper payload mechanics (empty → pickup → loaded)
    - Full observation space with distances, adjacency, recharge count
    - Battery consumption formula with payload effect
    - Recharge mechanics with penalties
    - Complete reward system with pickup/delivery bonuses
    - Action masking for invalid actions
    - MaskablePPO and regular PPO compatibility
    """
    def __init__(self, graph_path: str, battery_init: int = 100, payload_init: int = 1, 
             max_battery: int = 100, k_neighbors: int = 6, max_steps: int = 150,  # REDUCED from 200 to 150
             randomize_battery: bool = False, battery_range: tuple = (60, 100),
             randomize_payload: bool = False, payload_range: tuple = (1, 5),
             use_curriculum: bool = True, curriculum_threshold: float = 0.5):
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
        self.payload_init = payload_init  # charge capacity when loaded
        self.max_steps = max_steps
        
        # Randomization parameters
        self.randomize_battery = randomize_battery
        self.battery_range = battery_range
        self.randomize_payload = randomize_payload
        self.payload_range = payload_range
        
        # Battery consumption parameters
        self.k_norm = 10.8  # normalization factor
        self.alpha = 0.2    # payload effect factor
        
        # Recharge parameters - TOUGHER
        self.recharge_step = 30  # INCREASED from 20 to 30
        self.recharge_penalty = 1.0  # INCREASED from 0.05 to 1.0
        
        # NEW: HARD LIMIT on recharges - ADD THIS LINE
        self.MAX_RECHARGES = 5  # MECHANICAL LIMIT - no more than 5 recharges per episode
        
        # Build adjacency structure for fast lookup
        self.adjacency = self._build_adjacency()
        
        # Find node indices by type
        self.hubs = [i for i, node in enumerate(self.nodes) if node['type'] == 'hubs']
        self.pickups = [i for i, node in enumerate(self.nodes) if node['type'] == 'pickup']
        self.deliveries = [i for i, node in enumerate(self.nodes) if node['type'] == 'delivery']
        self.charging_stations = [i for i, node in enumerate(self.nodes) if node['type'] == 'charging']
        
        # Node type mappings
        self.type_to_id = {'hubs': 0, 'pickup': 1, 'delivery': 2, 'charging': 3}
        
        # NEW: Enhanced state space with global delivery signal
        # one-hot position (N) + battery (1) + payload (1) + flags (2) + 
        # recharge_count (1) + delivery_dist_global (1) + local_adjacency_enhanced (k*8)
        state_dim = self.n_nodes + 1 + 1 + 2 + 1 + 1 + (self.k_neighbors * 8)  # CHANGED from k*7 to k*8
        self.observation_space = spaces.Box(low=-1, high=1, shape=(state_dim,), dtype=np.float32)
        
        # NEW: Action space includes STAY action for recharging
        self.STAY_ACTION = self.k_neighbors  # Last action index = stay
        self.action_space = spaces.Discrete(self.k_neighbors + 1)  # +1 for stay action
        
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
        self.curriculum_threshold = curriculum_threshold
        self.curriculum_level = 0
        self.max_curriculum_level = 4
        self.episode_successes = []
        self.curriculum_window = 30  # REDUCED from 50 to 30
        
        # NEW: Curriculum parameters for battery and payload
        self.curriculum_battery_levels = [
            (90, 100),   # Level 0: High battery (easy)
            (80, 100),   # Level 1: Good battery
            (70, 100),   # Level 2: Medium battery
            (60, 100),   # Level 3: Low battery
            (50, 100)    # Level 4: Very low battery (hard)
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
        
        print(f"Initialized environment: {self.n_nodes} nodes, k_neighbors={k_neighbors}")
        print(f"Action space: {self.action_space.n} (0-{self.k_neighbors-1}=move, {self.STAY_ACTION}=stay)")
        print(f"MAX_RECHARGES={self.MAX_RECHARGES}, recharge_step={self.recharge_step}")
        print(f"Enhanced: explicit direction indicators per neighbor + 30% recharge threshold + 150 max steps")

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
        """Build adjacency list from edges with distances."""
        adjacency = {i: [] for i in range(self.n_nodes)}
        for edge in self.edges:
            # Handle both frontend format (u, v, dist) and training format (source, target, distance)
            if 'source' in edge and 'target' in edge:
                # Training format
                source = edge['source']
                target = edge['target']
                distance = edge.get('distance', edge.get('dist', 0))
            elif 'u' in edge and 'v' in edge:
                # Frontend format - CONVERT to expected format
                source = edge['u']
                target = edge['v']
                distance = edge.get('dist', edge.get('distance', 0))
            else:
                print(f"⚠️ Warning: Unknown edge format: {edge}")
                continue
                
            adjacency[source].append((target, distance))
            adjacency[target].append((source, distance))
        return adjacency
    
    def _get_k_nearest_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        """Get k nearest neighbors from current node."""
        if node_id not in self.adjacency:
            return []
        
        # Get all connected nodes
        neighbors = self.adjacency[node_id]
        
        # Sort by distance and take k nearest
        neighbors_sorted = sorted(neighbors, key=lambda x: x[1])
        
        # NEW: Break symmetry by shuffling neighbors with equal distances
        import random
        random.shuffle(neighbors_sorted)  # Breaks ties in distance ordering
        
        return neighbors_sorted[:self.k_neighbors]
    
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
        """Get current observation state with directional deltas + global delivery signal."""
        # One-hot position
        position = np.zeros(self.n_nodes, dtype=np.float32)
        if self.current_node is not None:
            position[self.current_node] = 1.0
        
        # Normalized battery and payload
        battery_norm = self.battery / self.max_battery
        payload_norm = self.payload / 10.0
        
        # Task flags
        pickup_flag = 1.0 if self.pickup_done else 0.0
        delivery_flag = 1.0 if self.delivery_done else 0.0
        
        # Recharge count (normalized)
        recharge_norm = min(1.0, self.recharge_count / 5.0)
        
        # Global delivery distance signal (always available)
        delivery_dist_global = 0.0
        if self.pickup_done and not self.delivery_done and self.dist_to_delivery and self.current_node is not None:
            cur_to_delivery = self.dist_to_delivery.get(self.current_node, float('inf'))
            if cur_to_delivery != float('inf'):
                delivery_dist_global = min(1.0, cur_to_delivery / 100.0)  # Normalize to [0,1]
        
        # NEW: Enhanced local adjacency with EXPLICIT direction indicators (k*8)
        local_adj = np.zeros(self.k_neighbors * 8, dtype=np.float32)  # CHANGED from k*7 to k*8
        if self.current_node is not None:
            neighbors = self._get_k_nearest_neighbors(self.current_node)
            
            # Get current distances to targets
            cur_to_pickup = self.dist_to_pickup.get(self.current_node, float('inf')) if self.dist_to_pickup else float('inf')
            cur_to_delivery = self.dist_to_delivery.get(self.current_node, float('inf')) if self.dist_to_delivery else float('inf')
            
            for i, (neighbor_id, edge_distance) in enumerate(neighbors):
                if i >= self.k_neighbors:
                    break
                
                base_idx = i * 8  # CHANGED from i * 7 to i * 8
                
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
                
                # [6] ENHANCED: Does this move get us CLOSER to objective? (+1=closer, -1=farther, 0=neutral)
                direction_indicator = 0.0
                
                if not self.pickup_done and self.dist_to_pickup:
                    # Phase 1: Focus on pickup target
                    nbr_to_pickup = self.dist_to_pickup.get(neighbor_id, float('inf'))
                    if cur_to_pickup != float('inf') and nbr_to_pickup != float('inf'):
                        if nbr_to_pickup < cur_to_pickup:
                            direction_indicator = 1.0  # Moving CLOSER to pickup
                        elif nbr_to_pickup > cur_to_pickup:
                            direction_indicator = -1.0  # Moving FARTHER from pickup
                        # else: stays 0.0 (same distance)
                        
                elif self.pickup_done and not self.delivery_done and self.dist_to_delivery:
                    # Phase 2: Focus on delivery target
                    nbr_to_delivery = self.dist_to_delivery.get(neighbor_id, float('inf'))
                    if cur_to_delivery != float('inf') and nbr_to_delivery != float('inf'):
                        if nbr_to_delivery < cur_to_delivery:
                            direction_indicator = 1.0  # Moving CLOSER to delivery
                        elif nbr_to_delivery > cur_to_delivery:
                            direction_indicator = -1.0  # Moving FARTHER from delivery
                        # else: stays 0.0 (same distance)
                
                local_adj[base_idx + 6] = direction_indicator
                
                # [7] NEW: Distance delta magnitude (how much closer/farther)
                delta_magnitude = 0.0
                
                if not self.pickup_done and self.dist_to_pickup:
                    nbr_to_pickup = self.dist_to_pickup.get(neighbor_id, float('inf'))
                    if cur_to_pickup != float('inf') and nbr_to_pickup != float('inf'):
                        delta_distance = abs(cur_to_pickup - nbr_to_pickup)
                        delta_magnitude = min(1.0, delta_distance / 20.0)  # Normalize to [0,1]
                        
                elif self.pickup_done and not self.delivery_done and self.dist_to_delivery:
                    nbr_to_delivery = self.dist_to_delivery.get(neighbor_id, float('inf'))
                    if cur_to_delivery != float('inf') and nbr_to_delivery != float('inf'):
                        delta_distance = abs(cur_to_delivery - nbr_to_delivery)
                        delta_magnitude = min(1.0, delta_distance / 20.0)  # Normalize to [0,1]
                
                local_adj[base_idx + 7] = delta_magnitude

        # Enhanced observation with explicit direction indicators
        observation = np.concatenate([
            position,                    # N dimensions
            [battery_norm],             # 1 dimension
            [payload_norm],             # 1 dimension
            [pickup_flag, delivery_flag], # 2 dimensions
            [recharge_norm],            # 1 dimension
            [delivery_dist_global],     # 1 dimension
            local_adj                   # k*8 dimensions (CHANGED from k*7)
        ])
        
        return observation.astype(np.float32)
    
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
        
        # Fallback (should never be reached in connected K-NN graph)
        return random.choice(self.hubs) if self.hubs else 0
    
    def _get_action_mask(self) -> np.ndarray:
        """Get mask for valid actions. True = valid, False = invalid."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        if self.current_node is not None:
            # Movement actions (0 to k_neighbors-1)
            neighbors = self._get_k_nearest_neighbors(self.current_node)
            for i, (neighbor_id, distance) in enumerate(neighbors):
                if i < self.k_neighbors:
                    battery_cost = self._calculate_battery_cost(distance)
                    if self.battery >= battery_cost:
                        mask[i] = True
        
            # STAY action (k_neighbors index) - Only block if physically impossible
            current_node_type = self.nodes[self.current_node]['type']
            mask[self.STAY_ACTION] = (
                current_node_type == 'charging' and              # Only on charging stations
                not self.stayed_last_step and                    # Prevent stay spam
                self.recharge_count < self.MAX_RECHARGES         # Respect recharge limit
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
                # Calculate distance using BFS
                distance = self._bfs_distance_between_nodes(pickup_idx, delivery_idx)
                self.pickup_delivery_pairs.append({
                    'pickup': pickup_idx,
                    'delivery': delivery_idx,
                    'distance': distance
                })
        
        # Sort by distance (easy to hard)
        self.pickup_delivery_pairs.sort(key=lambda x: x['distance'])
        
        # Group into curriculum levels
        total_pairs = len(self.pickup_delivery_pairs)
        pairs_per_level = max(1, total_pairs // (self.max_curriculum_level + 1))
        
        self.curriculum_pairs = []
        for level in range(self.max_curriculum_level + 1):
            start_idx = 0
            end_idx = min(total_pairs, (level + 1) * pairs_per_level)
            if level == self.max_curriculum_level:  # Last level gets all remaining
                end_idx = total_pairs
            
            level_pairs = self.pickup_delivery_pairs[start_idx:end_idx]
            self.curriculum_pairs.append(level_pairs)
            
            print(f"Curriculum Level {level}: {len(level_pairs)} pairs, "
                  f"distance range: {level_pairs[0]['distance']:.1f}-{level_pairs[-1]['distance']:.1f}")
    
    def _bfs_distance_between_nodes(self, start: int, end: int) -> float:
        """Calculate shortest distance between two nodes using BFS."""
        if start == end:
            return 0.0
        
        queue = [(start, 0)]
        visited = {start}
        
        while queue:
            node, dist = queue.pop(0)
            
            for neighbor, edge_dist in self.adjacency.get(node, []):
                if neighbor == end:
                    return dist + edge_dist
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + edge_dist))
        
        return float('inf')  # No path found
    
    def _update_curriculum(self, success: bool):
        """Update curriculum level based on recent performance."""
        if not self.use_curriculum:
            return
        
        # Track recent successes
        self.episode_successes.append(success)
        if len(self.episode_successes) > self.curriculum_window:
            self.episode_successes.pop(0)
        
        # NEW: Reduced minimum episodes before update
        if len(self.episode_successes) < min(10, self.curriculum_window // 3):  # REDUCED from 20
            return
        
        # Calculate recent success rate
        recent_success_rate = sum(self.episode_successes) / len(self.episode_successes)
        
        # Increase difficulty if doing well
        if (recent_success_rate >= self.curriculum_threshold and 
            self.curriculum_level < self.max_curriculum_level):
            old_level = self.curriculum_level
            self.curriculum_level += 1
            
            # Get new parameters for logging
            new_battery_range = self.curriculum_battery_levels[self.curriculum_level]
            new_payload_range = self.curriculum_payload_levels[self.curriculum_level]
            
            print(f"Curriculum Level UP! {old_level} → {self.curriculum_level}")
            print(f"   Success rate: {recent_success_rate:.1%}")
            print(f"   New battery range: {new_battery_range}")
            print(f"   New payload range: {new_payload_range}")
            
            # Reset tracking for new level
            self.episode_successes = []
            
        # Decrease difficulty if struggling too much
        elif (recent_success_rate < 0.2 and self.curriculum_level > 0 and  # LOWERED from 0.3
              len(self.episode_successes) >= self.curriculum_window):
            old_level = self.curriculum_level
            self.curriculum_level -= 1
            
            # Get new parameters for logging
            new_battery_range = self.curriculum_battery_levels[self.curriculum_level]
            new_payload_range = self.curriculum_payload_levels[self.curriculum_level]
            
            print(f"Curriculum Level DOWN! {old_level} → {self.curriculum_level}")
            print(f"   Success rate: {recent_success_rate:.1%}")
            print(f"   New battery range: {new_battery_range}")
            print(f"   New payload range: {new_payload_range}")
            
            # Reset tracking for easier level
            self.episode_successes = []

    def _get_curriculum_battery_range(self) -> tuple:
        """Get battery range for current curriculum level."""
        if not self.use_curriculum:
            return self.battery_range
        return self.curriculum_battery_levels[self.curriculum_level]

    def _get_curriculum_payload_range(self) -> tuple:
        """Get payload range for current curriculum level."""
        if not self.use_curriculum:
            return self.payload_range
        return self.curriculum_payload_levels[self.curriculum_level]
    
    def _select_targets_curriculum(self):
        """Select pickup and delivery targets based on curriculum level."""
        if not self.use_curriculum or not self.curriculum_pairs:
            # Fallback to random selection
            self._select_targets_random()
            return
        
        # Get pairs for current curriculum level
        current_pairs = self.curriculum_pairs[self.curriculum_level]
        
        if not current_pairs:
            self._select_targets_random()
            return
        
        # Select random pair from current curriculum level
        selected_pair = random.choice(current_pairs)
        self.pickup_target = selected_pair['pickup']
        self.delivery_target = selected_pair['delivery']
        
        # Store pair info for logging
        self.current_pair_distance = selected_pair['distance']
    
    def _select_targets_random(self):
        """Fallback random target selection."""
        if self.pickup_target is None and self.pickups:
            self.pickup_target = random.choice(self.pickups)
        if self.delivery_target is None and self.deliveries:
            self.delivery_target = random.choice(self.deliveries)
        self.current_pair_distance = self._bfs_distance_between_nodes(
            self.pickup_target, self.delivery_target) if self.pickup_target and self.delivery_target else 0
    
    def _select_targets(self):
        """Select pickup and delivery targets (curriculum or random)."""
        # Only select if targets are not already set (allows manual override)
        if self.pickup_target is None or self.delivery_target is None:
            if self.use_curriculum:
                self._select_targets_curriculum()
            else:
                self._select_targets_random()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset targets to None to force curriculum selection
        if self.use_curriculum:
            self.pickup_target = None
            self.delivery_target = None
        
        # Select targets (curriculum-based if enabled)
        self._select_targets()
        
        # NEW: Pre-compute shortest paths from targets to all nodes (once per episode)
        print(f"🔄 Computing shortest paths for pickup={self.pickup_target}, delivery={self.delivery_target}")
        self.dist_to_pickup = self._dijkstra_from_target(self.pickup_target)
        self.dist_to_delivery = self._dijkstra_from_target(self.delivery_target)
        
        # Spawn in hub closest to pickup target
        if self.pickup_target is not None:
            self.current_node = self._closest_hub(self.pickup_target)
        else:
            self.current_node = random.choice(self.hubs) if self.hubs else 0
        
        # Reset state
        self.payload = 0
        self.step_count = 0
        self.pickup_done = False
        self.delivery_done = False
        self.recharge_count = 0
        
        # NEW: Reset battery with curriculum-adapted randomization
        if self.randomize_battery or self.use_curriculum:
            if self.use_curriculum:
                curriculum_battery_range = self._get_curriculum_battery_range()
                self.battery = random.randint(*curriculum_battery_range)
            else:
                self.battery = random.randint(*self.battery_range)
        else:
            self.battery = self.battery_init
        
        # NEW: Initialize progress tracking for dense rewards
        self.last_dist_to_pickup = self.dist_to_pickup.get(self.current_node, float('inf')) if self.dist_to_pickup else float('inf')
        self.last_dist_to_delivery = self.dist_to_delivery.get(self.current_node, float('inf')) if self.dist_to_delivery else float('inf')
        
        # Reset episode tracking
        self.episode_path = [self.current_node]
        self.episode_actions = []
        self.episode_rewards = []
        self.total_reward = 0
        self.stayed_last_step = False  # NEW: Reset stay flag
        
        observation = self._get_observation()
        info = {
            'step': self.step_count,
            'battery': self.battery,
            'payload': self.payload,
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
        """Execute one step in the environment."""
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
        
        # Handle STAY action first
        elif action == self.STAY_ACTION:
            self.stayed_last_step = True
            current_node_type = self.nodes[self.current_node]['type']
            
            if current_node_type != 'charging':
                # Allow staying but give small penalty - no termination
                reward = -0.5  # REDUCED from -3.0, no termination
                info['action_type'] = 'stay_invalid'
            else:
                # 30% threshold recharge logic
                battery_percentage = self.battery / self.max_battery
                
                if battery_percentage <= 0.30:  # Battery ≤ 30% = good decision, small bonus
                    base_reward = 0.5
                else:                    
                    base_reward = -1.0
                
                # Apply recharge
                old_battery = self.battery
                self.battery = min(self.max_battery, self.battery + self.recharge_step)
                self.recharge_count += 1
                
                reward = base_reward
                info['action_type'] = 'stay_recharge'
                info['battery_gained'] = self.battery - old_battery
                info['battery_percentage_before'] = battery_percentage
                info['recharge_count'] = self.recharge_count
            
            self.step_count += 1
            
        # MOVEMENT actions (0 to k_neighbors-1)
        else:
            self.stayed_last_step = False
            neighbors = self._get_k_nearest_neighbors(self.current_node)
            
            if action < len(neighbors):
                next_node, distance = neighbors[action]
                battery_cost = self._calculate_battery_cost(distance)
                
                if self.battery >= battery_cost:
                    # Execute movement
                    self.battery -= battery_cost
                    old_node = self.current_node
                    self.current_node = next_node
                    self.episode_path.append(next_node)
                    
                    # Base movement penalty
                    reward = -0.01
                    
                    # NEW: DENSE PROGRESS REWARD - This is the key fix!
                    if not self.pickup_done and self.dist_to_pickup:
                        # Phase 1: Reward getting closer to pickup
                        old_dist = self.last_dist_to_pickup
                        new_dist = self.dist_to_pickup.get(self.current_node, float('inf'))
                        if old_dist != float('inf') and new_dist != float('inf'):
                            progress = old_dist - new_dist  # positive = getting closer
                            progress_reward = 0.02 * progress
                            # Extra penalty for moving away to discourage ping-pong
                            if progress < 0:
                                progress_reward *= 1.5  # 1.5x penalty for moving away
                            reward += progress_reward
                            info['progress_to_pickup'] = progress
                            info['progress_reward'] = progress_reward
                        self.last_dist_to_pickup = new_dist
                        
                    elif self.pickup_done and not self.delivery_done and self.dist_to_delivery:
                        # Phase 2: Reward getting closer to delivery
                        old_dist = self.last_dist_to_delivery
                        new_dist = self.dist_to_delivery.get(self.current_node, float('inf'))
                        if old_dist != float('inf') and new_dist != float('inf'):
                            progress = old_dist - new_dist  # positive = getting closer
                            progress_reward = 0.02 * progress
                            # Extra penalty for moving away
                            if progress < 0:
                                progress_reward *= 1.5  # 1.5x penalty for moving away
                            reward += progress_reward
                            info['progress_to_delivery'] = progress
                            info['progress_reward'] = progress_reward
                        self.last_dist_to_delivery = new_dist
                    
                    # Check if we're at pickup/delivery targets
                    if next_node == self.pickup_target and not self.pickup_done:
                        self.pickup_done = True
                        self.payload = self.payload_init if self.randomize_payload else self.payload_init
                        reward += 15.0
                        info['payload_loaded'] = self.payload
                        
                    elif next_node == self.delivery_target and self.pickup_done and not self.delivery_done:
                        self.delivery_done = True
                        self.payload = 0
                        reward += 200.0  # Big success reward
                        info['delivery_completed'] = True
                        
                    # REMOVED: No penalty for visiting delivery without pickup
                    # Players can explore freely
                        
                    info['action_type'] = 'move'
                    info['distance'] = distance
                    info['battery_cost'] = battery_cost
                    
                else:
                    # Not enough battery
                    reward = -2.0
                    info['action_type'] = 'move_insufficient_battery'
                    info['required_battery'] = battery_cost
                    info['current_battery'] = self.battery
            else:
                # Invalid action index
                reward = -2.0
                terminated = True
                info['termination_reason'] = 'invalid_action_index'
                info['action_type'] = 'invalid_move'
            
            self.step_count += 1
        
        # Check termination conditions
        if self.battery <= 0:
            terminated = True
            if self.delivery_done:
                reward -= 3.0
            elif self.pickup_done:
                reward -= 12.0
            else:
                reward -= 20.0
            info['termination_reason'] = 'battery_depleted'
            
        if self.step_count >= self.max_steps:
            truncated = True
            if self.delivery_done:
                reward -= 2.0
            elif self.pickup_done:
                reward -= 15.0
            else:
                reward -= 25.0
            info['termination_reason'] = 'max_steps_reached'

        # Track reward and update info
        self.episode_rewards.append(reward)
        self.total_reward += reward

        # Battery metrics (unchanged)
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
            'stayed_last_step': self.stayed_last_step
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
        # Initialize environment
        graph_path = "../data/graph.json"
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
