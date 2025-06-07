/**
 * Genetic Algorithm for Drone Delivery Path Optimization
 * Finds optimal routes considering battery consumption, payload, and weather conditions
 */

// GA Configuration
const GA_CONFIG = {
  POPULATION_SIZE: 30,
  GENERATIONS: 50,
  MUTATION_RATE: 0.2,
  CROSSOVER_RATE: 0.8,
  ELITE_SIZE: 3,
  TOURNAMENT_SIZE: 3,
  
  // Battery consumption parameters
  K_NORM: 1.2,
  ALPHA: 0.4,
  
  // Constraints
  MAX_ROUTE_LENGTH: 15,
  BATTERY_PENALTY: 1000,
  INVALID_ROUTE_PENALTY: 10000,  // INCREASED: was 5000
  MISSION_FAILURE_PENALTY: 50000  // NEW: Very high penalty for mission failure
};

/**
 * Individual representation of a route
 * Route structure: [startHub, ...intermediateNodes..., pickupNode, ...intermediateNodes..., deliveryNode, ...intermediateNodes..., endHub]
 */
class Individual {
  constructor(route = []) {
    this.route = route;
    this.fitness = null;
    this.batteryHistory = [];
    this.isValid = false;
  }

  copy() {
    const newIndividual = new Individual([...this.route]);
    newIndividual.fitness = this.fitness;
    newIndividual.batteryHistory = [...this.batteryHistory];
    newIndividual.isValid = this.isValid;
    return newIndividual;
  }
}

/**
 * Main Genetic Algorithm class
 */
export class GeneticAlgorithm {
  constructor(nodes, edges, pickupNodeId, deliveryNodeId, batteryCapacity = 100, maxPayload = 3) {
    this.nodes = nodes;
    this.edges = edges;
    this.edgeMap = this.buildEdgeMap(edges);
    this.pickupNodeId = pickupNodeId;
    this.deliveryNodeId = deliveryNodeId;
    this.batteryCapacity = batteryCapacity;
    this.maxPayload = maxPayload;
    
    // Find node indices with extensive debugging
    this.pickupIndex = nodes.findIndex(n => n.id === pickupNodeId);
    this.deliveryIndex = nodes.findIndex(n => n.id === deliveryNodeId);
    this.hubIndices = nodes.map((n, i) => n.type === 'hubs' ? i : -1).filter(i => i >= 0);
    this.chargingIndices = nodes.map((n, i) => n.type === 'charging' ? i : -1).filter(i => i >= 0);
    
    // CRITICAL DEBUG: Verify node mapping integrity
    console.log(`[GA] 🔍 DEBUGGING NODE MAPPING:`);
    console.log(`[GA]   Target pickup: "${pickupNodeId}" → index ${this.pickupIndex}`);
    console.log(`[GA]   Target delivery: "${deliveryNodeId}" → index ${this.deliveryIndex}`);
    
    if (this.pickupIndex >= 0) {
      const actualPickupNode = nodes[this.pickupIndex];
      console.log(`[GA]   Pickup node at index ${this.pickupIndex}: "${actualPickupNode?.id}" (type: ${actualPickupNode?.type})`);
    }
    
    if (this.deliveryIndex >= 0) {
      const actualDeliveryNode = nodes[this.deliveryIndex];
      console.log(`[GA]   Delivery node at index ${this.deliveryIndex}: "${actualDeliveryNode?.id}" (type: ${actualDeliveryNode?.type})`);
    }
    
    // Verify no duplicate IDs
    const idCounts = {};
    nodes.forEach((node, idx) => {
      idCounts[node.id] = (idCounts[node.id] || 0) + 1;
      if (idCounts[node.id] > 1) {
        console.error(`[GA] 🚨 DUPLICATE ID DETECTED: "${node.id}" appears multiple times!`);
      }
    });
    
    // List all pickup and delivery nodes for verification
    const allPickups = nodes.filter(n => n.type === 'pickup').map((n, i) => `"${n.id}":${nodes.indexOf(n)}`);
    const allDeliveries = nodes.filter(n => n.type === 'delivery').map((n, i) => `"${n.id}":${nodes.indexOf(n)}`);
    console.log(`[GA]   All pickup nodes: ${allPickups.join(', ')}`);
    console.log(`[GA]   All delivery nodes: ${allDeliveries.join(', ')}`);
    
    console.log(`[GA] Initialized - Pickup: ${pickupNodeId} (${this.pickupIndex}), Delivery: ${deliveryNodeId} (${this.deliveryIndex})`);
    console.log(`[GA] Hubs: ${this.hubIndices.length}, Charging: ${this.chargingIndices.length}`);
  }

  buildEdgeMap(edges) {
    const map = new Map();
    const adjacencyList = new Map();
    
    // Build both edge map and adjacency list
    edges.forEach(edge => {
      const key = `${edge.source}-${edge.target}`;
      map.set(key, edge);
      
      // Add to adjacency list
      if (!adjacencyList.has(edge.source)) {
        adjacencyList.set(edge.source, []);
      }
      adjacencyList.get(edge.source).push(edge.target);
      
      // Add reverse direction if not exists
      const reverseKey = `${edge.target}-${edge.source}`;
      if (!map.has(reverseKey)) {
        const reverseEdge = {
          source: edge.target,
          target: edge.source,
          distance: edge.distance,
          cost: edge.cost
        };
        map.set(reverseKey, reverseEdge);
        
        // Add to adjacency list
        if (!adjacencyList.has(edge.target)) {
          adjacencyList.set(edge.target, []);
        }
        adjacencyList.get(edge.target).push(edge.source);
      }
    });
    
    this.adjacencyList = adjacencyList;
    return map;
  }

  /**
   * Find a valid path between two nodes using BFS
   */
  findPath(startIndex, endIndex, maxLength = 5) {
    if (startIndex === endIndex) return [startIndex];
    
    const queue = [[startIndex]];
    const visited = new Set();
    
    while (queue.length > 0) {
      const path = queue.shift();
      const currentNode = path[path.length - 1];
      
      if (path.length > maxLength) continue;
      
      if (visited.has(currentNode)) continue;
      visited.add(currentNode);
      
      const neighbors = this.adjacencyList.get(currentNode) || [];
      
      for (const neighbor of neighbors) {
        if (path.includes(neighbor)) continue; // Avoid cycles
        
        const newPath = [...path, neighbor];
        
        if (neighbor === endIndex) {
          return newPath;
        }
        
        if (newPath.length < maxLength) {
          queue.push(newPath);
        }
      }
    }
    
    // If no path found, return direct connection if it exists
    const neighbors = this.adjacencyList.get(startIndex) || [];
    if (neighbors.includes(endIndex)) {
      return [startIndex, endIndex];
    }
    
    return null; // No path found
  }

  /**
   * Generate a random valid route following graph connections
   */
  generateRandomRoute() {
    // Choose start hub close to pickup point
    const startHub = this.selectClosestHub(this.pickupIndex);
    // Choose end hub close to delivery point  
    const endHub = this.selectClosestHub(this.deliveryIndex);
    
    // Build route using only valid graph paths
    const fullRoute = this.buildCompleteValidRoute(startHub, endHub);
    
    return new Individual(fullRoute);
  }

  /**
   * Build a complete valid route that MUST follow graph edges AND ensure sufficient battery
   */
  buildCompleteValidRoute(startHub, endHub) {
    let route = [];
    
    console.log(`[GA] 🎯 Building route: Hub ${this.nodes[startHub]?.id} → ${this.pickupNodeId} (idx:${this.pickupIndex}) → ${this.deliveryNodeId} (idx:${this.deliveryIndex}) → Hub ${this.nodes[endHub]?.id}`);
    
    // CRITICAL: Verify pickup and delivery indices are correct
    if (this.pickupIndex < 0) {
      console.error(`[GA] 🚨 INVALID PICKUP INDEX: ${this.pickupIndex} for ID ${this.pickupNodeId}`);
      return [startHub, endHub]; // Emergency fallback
    }
    
    if (this.deliveryIndex < 0) {
      console.error(`[GA] 🚨 INVALID DELIVERY INDEX: ${this.deliveryIndex} for ID ${this.deliveryNodeId}`);
      return [startHub, endHub]; // Emergency fallback
    }
    
    // Step 1: Build path from start hub to pickup with battery consideration
    const pathToPickup = this.buildBatteryAwarePath(startHub, this.pickupIndex, 0); // No payload initially
    if (pathToPickup && pathToPickup.length > 0) {
      route.push(...pathToPickup);
      console.log(`[GA] ✅ Added battery-aware path to pickup (${pathToPickup.length} nodes): ${pathToPickup.map(i => this.nodes[i]?.id).join(' → ')}`);
    } else {
      // Force direct connection - this ensures pickup is ALWAYS included
      route = [startHub, this.pickupIndex];
      console.log(`[GA] ⚠️  FORCED direct hub→pickup connection as fallback`);
    }
    
    // CRITICAL: Double-check pickup is in the route
    if (!route.includes(this.pickupIndex)) {
      console.warn(`[GA] 🚨 PICKUP MISSING! Force-adding at position 1`);
      route.splice(1, 0, this.pickupIndex);
    }
    
    // Step 2: Build path from pickup to delivery with payload consideration
    console.log(`[GA] 🎯 Now adding delivery ${this.deliveryNodeId} (idx:${this.deliveryIndex}) with payload`);
    
    const pathToDelivery = this.buildBatteryAwarePath(this.pickupIndex, this.deliveryIndex, this.maxPayload);
    if (pathToDelivery && pathToDelivery.length > 1) {
      route.push(...pathToDelivery.slice(1)); // Skip pickup (already included)
      console.log(`[GA] ✅ Added battery-aware path pickup→delivery (${pathToDelivery.length - 1} new nodes): ${pathToDelivery.slice(1).map(i => this.nodes[i]?.id).join(' → ')}`);
    } else {
      // Force direct connection
      route.push(this.deliveryIndex);
      console.log(`[GA] ⚠️  FORCED direct pickup→delivery connection as last resort`);
    }
    
    // CRITICAL: Double-check delivery is in the route BEFORE adding end hub
    if (!route.includes(this.deliveryIndex)) {
      console.error(`[GA] 🚨 DELIVERY MISSING! Force-adding before end hub`);
      route.push(this.deliveryIndex);
    }
    
    // Step 3: Build path from delivery to end hub (no payload)
    const pathToEnd = this.buildBatteryAwarePath(this.deliveryIndex, endHub, 0);
    if (pathToEnd && pathToEnd.length > 1) {
      route.push(...pathToEnd.slice(1));
      console.log(`[GA] ✅ Added battery-aware path delivery→end hub (${pathToEnd.length - 1} new nodes): ${pathToEnd.slice(1).map(i => this.nodes[i]?.id).join(' → ')}`);
    } else {
      // Force direct connection to end hub
      route.push(endHub);
      console.log(`[GA] ⚠️  FORCED direct delivery→hub connection`);
    }
    
    // Final comprehensive verification
    const hasPickupFinal = route.includes(this.pickupIndex);
    const hasDeliveryFinal = route.includes(this.deliveryIndex);
    const pickupPos = route.indexOf(this.pickupIndex);
    const deliveryPos = route.indexOf(this.deliveryIndex);
    
    console.log(`[GA] 🔍 Final route verification:`);
    console.log(`[GA]   Pickup ${this.pickupNodeId}: ${hasPickupFinal} (position ${pickupPos})`);
    console.log(`[GA]   Delivery ${this.deliveryNodeId}: ${hasDeliveryFinal} (position ${deliveryPos})`);
    console.log(`[GA]   Correct order: ${pickupPos < deliveryPos && pickupPos >= 0 && deliveryPos >= 0}`);
    console.log(`[GA]   Route: ${route.map(i => this.nodes[i]?.id).join(' → ')}`);
    
    // Emergency repairs if still missing critical points
    if (!hasPickupFinal) {
      console.error(`[GA] 🚨 EMERGENCY: Pickup still missing, inserting at position 1`);
      route.splice(1, 0, this.pickupIndex);
    }
    
    if (!hasDeliveryFinal) {
      console.error(`[GA] 🚨 EMERGENCY: Delivery still missing, inserting before last node`);
      route.splice(-1, 0, this.deliveryIndex);
    }
    
    // Fix order if wrong
    const finalPickupPos = route.indexOf(this.pickupIndex);
    const finalDeliveryPos = route.indexOf(this.deliveryIndex);
    if (finalPickupPos >= finalDeliveryPos || finalPickupPos < 0 || finalDeliveryPos < 0) {
      console.error(`[GA] 🚨 EMERGENCY: Wrong order (P:${finalPickupPos}, D:${finalDeliveryPos}), rebuilding route`);
      // Rebuild with minimal working route
      route = [startHub, this.pickupIndex, this.deliveryIndex, endHub];
      console.log(`[GA] 🔧 Emergency minimal route: ${route.map(i => this.nodes[i]?.id).join(' → ')}`);
    }
    
    console.log(`[GA] ✅ FINAL ROUTE (${route.length} nodes): ${route.map(i => this.nodes[i]?.id).join(' → ')}`);
    
    return route;
  }

  /**
   * Build a battery-aware path that includes charging stations when needed
   */
  buildBatteryAwarePath(fromIndex, toIndex, payload = 0) {
    // First try direct path
    const directPath = this.findValidPath(fromIndex, toIndex, 4);
    if (directPath && this.isPathBatteryFeasible(directPath, payload)) {
      console.log(`[GA] 🔋 Direct path is battery-feasible: ${directPath.map(i => this.nodes[i]?.id).join(' → ')}`);
      return directPath;
    }
    
    console.log(`[GA] 🔋 Direct path needs charging support, finding charging stations...`);
    
    // Find charging stations that can help
    const viableCharging = this.findChargingStationsForPath(fromIndex, toIndex, payload);
    
    if (viableCharging.length > 0) {
      // Try paths through charging stations
      for (const chargingIndex of viableCharging) {
        const pathToCharging = this.findValidPath(fromIndex, chargingIndex, 4);
        const pathFromCharging = this.findValidPath(chargingIndex, toIndex, 4);
        
        if (pathToCharging && pathFromCharging && 
            this.isPathBatteryFeasible(pathToCharging, payload) && 
            this.isPathBatteryFeasible(pathFromCharging, payload)) {
          
          const fullPath = [...pathToCharging, ...pathFromCharging.slice(1)];
          console.log(`[GA] 🔋 Found battery-feasible path via ${this.nodes[chargingIndex]?.id}: ${fullPath.map(i => this.nodes[i]?.id).join(' → ')}`);
          return fullPath;
        }
      }
      
      // Try multiple charging stations if single ones don't work
      const multiChargingPath = this.buildMultiChargingPath(fromIndex, toIndex, payload, viableCharging);
      if (multiChargingPath) {
        console.log(`[GA] 🔋 Found multi-charging path: ${multiChargingPath.map(i => this.nodes[i]?.id).join(' → ')}`);
        return multiChargingPath;
      }
    }
    
    console.log(`[GA] ⚠️  No battery-feasible path found, returning direct path anyway`);
    return directPath; // Return direct path as fallback
  }

  /**
   * Check if a path is feasible with current battery capacity
   */
  isPathBatteryFeasible(path, payload = 0) {
    if (!path || path.length < 2) return true;
    
    let currentBattery = this.batteryCapacity;
    
    for (let i = 0; i < path.length - 1; i++) {
      const fromIndex = path[i];
      const toIndex = path[i + 1];
      
      const batteryCost = this.calculateBatteryCost(fromIndex, toIndex, payload);
      if (batteryCost === Infinity) return false;
      
      currentBattery -= batteryCost;
      
      // Check if we can recharge at charging stations
      const currentNode = this.nodes[toIndex];
      if (currentNode && currentNode.type === 'charging') {
        currentBattery = Math.min(this.batteryCapacity, currentBattery + 30); // Recharge 30%
      }
      
      // If battery goes below 0 and we're not at a charging station, path is not feasible
      if (currentBattery < 0 && (!currentNode || currentNode.type !== 'charging')) {
        return false;
      }
    }
    
    return currentBattery >= 0;
  }

  /**
   * Find charging stations that can help for a specific path
   */
  findChargingStationsForPath(fromIndex, toIndex, payload = 0) {
    const maxDistance = this.batteryCapacity * GA_CONFIG.K_NORM / (1 + GA_CONFIG.ALPHA * payload);
    const directDistance = this.getNodeDistance(fromIndex, toIndex);
    
    // If direct distance is within battery range, no charging needed
    if (directDistance * 1.5 < maxDistance) { // 1.5 safety factor
      return [];
    }
    
    // Find charging stations that are:
    // 1. Reachable from start
    // 2. Can reach destination
    // 3. Are positioned to help with the journey
    const candidateStations = this.chargingIndices.filter(chargingIndex => {
      const distanceToCharging = this.getNodeDistance(fromIndex, chargingIndex);
      const distanceFromCharging = this.getNodeDistance(chargingIndex, toIndex);
      
      // Station should be roughly on the path and within battery range
      return distanceToCharging < maxDistance * 0.8 && 
             distanceFromCharging < maxDistance * 0.8 &&
             (distanceToCharging + distanceFromCharging) < directDistance * 1.5;
    });
    
    // Sort by total distance (closest to direct path)
    candidateStations.sort((a, b) => {
      const distA = this.getNodeDistance(fromIndex, a) + this.getNodeDistance(a, toIndex);
      const distB = this.getNodeDistance(fromIndex, b) + this.getNodeDistance(b, toIndex);
      return distA - distB;
    });
    
    return candidateStations.slice(0, 5); // Return top 5 candidates
  }

  /**
   * Build a path through multiple charging stations if needed
   */
  buildMultiChargingPath(fromIndex, toIndex, payload, chargingStations) {
    // For very long distances, might need multiple charging stops
    const maxDistance = this.batteryCapacity * GA_CONFIG.K_NORM / (1 + GA_CONFIG.ALPHA * payload) * 0.8;
    const totalDistance = this.getNodeDistance(fromIndex, toIndex);
    
    if (totalDistance < maxDistance * 2) {
      return null; // Single charging should be enough
    }
    
    // Try to find a chain of charging stations
    let currentPos = fromIndex;
    const pathIndices = [fromIndex];
    
    while (this.getNodeDistance(currentPos, toIndex) > maxDistance) {
      // Find the charging station that gets us closest to destination while staying in range
      let bestCharging = null;
      let bestProgress = 0;
      
      for (const chargingIndex of chargingStations) {
        if (pathIndices.includes(chargingIndex)) continue; // Don't revisit
        
        const distanceToCharging = this.getNodeDistance(currentPos, chargingIndex);
        const distanceFromCharging = this.getNodeDistance(chargingIndex, toIndex);
        
        if (distanceToCharging < maxDistance) {
          const progress = this.getNodeDistance(fromIndex, toIndex) - distanceFromCharging;
          if (progress > bestProgress) {
            bestProgress = progress;
            bestCharging = chargingIndex;
          }
        }
      }
      
      if (bestCharging === null) {
        break; // Can't find suitable charging station
      }
      
      pathIndices.push(bestCharging);
      currentPos = bestCharging;
    }
    
    // Add final destination
    pathIndices.push(toIndex);
    
    // Verify the complete path is valid
    if (this.isPathBatteryFeasible(pathIndices, payload)) {
      return pathIndices;
    }
    
    return null;
  }

  /**
   * Find a charging station that can serve as intermediate point between two nodes
   */
  findChargingStationBetween(fromIndex, toIndex) {
    // Find charging stations that are reachable from 'from' and can reach 'to'
    const viableStations = this.chargingIndices.filter(chargingIndex => {
      const pathToCharging = this.findValidPath(fromIndex, chargingIndex, 3);
      const pathFromCharging = this.findValidPath(chargingIndex, toIndex, 3);
      return pathToCharging && pathFromCharging;
    });
    
    if (viableStations.length > 0) {
      // Return closest charging station
      let bestStation = viableStations[0];
      let bestDistance = this.getNodeDistance(fromIndex, bestStation) + this.getNodeDistance(bestStation, toIndex);
      
      viableStations.forEach(station => {
        const totalDistance = this.getNodeDistance(fromIndex, station) + this.getNodeDistance(station, toIndex);
        if (totalDistance < bestDistance) {
          bestDistance = totalDistance;
          bestStation = station;
        }
      });
      
      return bestStation;
    }
    
    return null;
  }

  /**
   * Enhanced pathfinding with better connectivity checking
   */
  findValidPath(startIndex, endIndex, maxLength = 5) {
    if (startIndex === endIndex) return [startIndex];
    
    const queue = [[startIndex]];
    const visited = new Set();
    
    while (queue.length > 0) {
      const path = queue.shift();
      const currentNode = path[path.length - 1];
      
      if (path.length > maxLength) continue;
      
      const pathKey = path.join('-');
      if (visited.has(pathKey)) continue;
      visited.add(pathKey);
      
      const neighbors = this.adjacencyList.get(currentNode) || [];
      
      for (const neighbor of neighbors) {
        if (path.includes(neighbor)) continue; // Avoid cycles
        
        const newPath = [...path, neighbor];
        
        if (neighbor === endIndex) {
          return newPath;
        }
        
        if (newPath.length < maxLength) {
          queue.push(newPath);
        }
      }
    }
    
    return null; // No valid path found
  }

  /**
   * Select the closest hub to a target node
   */
  selectClosestHub(targetIndex) {
    let closestHub = this.hubIndices[0];
    let minDistance = this.getNodeDistance(closestHub, targetIndex);
    
    for (const hubIndex of this.hubIndices) {
      const distance = this.getNodeDistance(hubIndex, targetIndex);
      if (distance < minDistance) {
        minDistance = distance;
        closestHub = hubIndex;
      }
    }
    
    return closestHub;
  }

  /**
   * Select the best neighbor node to move towards target
   */
  selectBestNeighbor(currentNode, targetNode, neighbors) {
    if (neighbors.includes(targetNode)) {
      return targetNode; // Direct path to target
    }
    
    // Choose neighbor that's closest to target
    let bestNeighbor = null;
    let bestDistance = Infinity;
    
    neighbors.forEach(neighbor => {
      const distance = this.getNodeDistance(neighbor, targetNode);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestNeighbor = neighbor;
      }
    });
    
    return bestNeighbor;
  }

  /**
   * Get euclidean distance between two nodes
   */
  getNodeDistance(nodeIndex1, nodeIndex2) {
    const node1 = this.nodes[nodeIndex1];
    const node2 = this.nodes[nodeIndex2];
    if (!node1 || !node2) return Infinity;
    
    const latDiff = node1.lat - node2.lat;
    const lngDiff = node1.lng - node2.lng;
    return Math.sqrt(latDiff * latDiff + lngDiff * lngDiff);
  }

  /**
   * Calculate battery cost between two nodes
   */
  calculateBatteryCost(fromIndex, toIndex, payload = 0) {
    const key = `${fromIndex}-${toIndex}`;
    const edge = this.edgeMap.get(key);
    
    if (!edge) {
      return Infinity; // No direct connection
    }
    
    const distance = edge.distance;
    const batteryCost = (distance * (1 + GA_CONFIG.ALPHA * payload)) / GA_CONFIG.K_NORM;
    
    return batteryCost;
  }

  /**
   * Initialize population with random routes
   */
  initializePopulation() {
    const population = [];
    
    for (let i = 0; i < GA_CONFIG.POPULATION_SIZE; i++) {
      const individual = this.generateRandomRoute();
      population.push(individual);
    }
    
    console.log(`[GA] Generated initial population of ${population.length} individuals`);
    return population;
  }

  /**
   * Evaluate fitness of an individual
   * Lower fitness = better route
   */
  evaluateFitness(individual) {
    const route = individual.route;
    
    // Check route validity FIRST
    if (!this.isRouteValid(route)) {
      individual.fitness = GA_CONFIG.INVALID_ROUTE_PENALTY;
      individual.isValid = false;
      return individual.fitness;
    }
    
    let totalCost = 0;
    let currentBattery = this.batteryCapacity;
    const batteryHistory = [currentBattery];
    let payload = 0;
    let pickupDone = false;
    let deliveryDone = false;
    
    // Check if pickup and delivery are in the route at all
    const hasPickup = route.includes(this.pickupIndex);
    const hasDelivery = route.includes(this.deliveryIndex);
    
    // MASSIVE penalty if pickup or delivery are missing from route
    if (!hasPickup) {
      individual.fitness = GA_CONFIG.MISSION_FAILURE_PENALTY;
      individual.isValid = false;
      return individual.fitness;
    }
    
    if (!hasDelivery) {
      individual.fitness = GA_CONFIG.MISSION_FAILURE_PENALTY;
      individual.isValid = false;
      return individual.fitness;
    }
    
    // Simulate route execution
    for (let i = 0; i < route.length - 1; i++) {
      const fromIndex = route[i];
      const toIndex = route[i + 1];
      
      // Update payload based on current position
      if (fromIndex === this.pickupIndex && !pickupDone) {
        payload = this.maxPayload;
        pickupDone = true;
      }
      if (fromIndex === this.deliveryIndex && pickupDone && !deliveryDone) {
        payload = 0;
        deliveryDone = true;
      }
      
      // Calculate battery cost for this segment
      const batteryCost = this.calculateBatteryCost(fromIndex, toIndex, payload);
      
      if (batteryCost === Infinity) {
        individual.fitness = GA_CONFIG.INVALID_ROUTE_PENALTY;
        individual.isValid = false;
        return individual.fitness;
      }
      
      currentBattery -= batteryCost;
      batteryHistory.push(currentBattery);
      
      // Check if we can recharge at charging stations
      const currentNode = this.nodes[toIndex];
      if (currentNode && currentNode.type === 'charging') {
        currentBattery = Math.min(this.batteryCapacity, currentBattery + 30); // Recharge 30%
        batteryHistory[batteryHistory.length - 1] = currentBattery;
      }
      
      // Add battery cost to total
      totalCost += batteryCost;
      
      // Battery penalty if low
      if (currentBattery < 0) {
        const penalty = GA_CONFIG.BATTERY_PENALTY * Math.abs(currentBattery);
        totalCost += penalty;
      }
    }
    
    // CRITICAL: Mission completion penalties
    if (!pickupDone) {
      totalCost += GA_CONFIG.MISSION_FAILURE_PENALTY;
    }
    if (!deliveryDone) {
      totalCost += GA_CONFIG.MISSION_FAILURE_PENALTY;
    }
    
    // Additional penalty if pickup happens after delivery (wrong order)
    const pickupPos = route.indexOf(this.pickupIndex);
    const deliveryPos = route.indexOf(this.deliveryIndex);
    if (pickupPos >= deliveryPos) {
      const penalty = GA_CONFIG.MISSION_FAILURE_PENALTY / 2;
      totalCost += penalty;
    }
    
    // Length penalty (prefer shorter routes)
    const lengthPenalty = route.length * 2;
    totalCost += lengthPenalty;
    
    individual.fitness = totalCost;
    individual.batteryHistory = batteryHistory;
    individual.isValid = pickupDone && deliveryDone && currentBattery >= 0;
    
    return individual.fitness;
  }

  /**
   * Check if route is valid (contains pickup and delivery in correct order AND follows graph edges)
   */
  isRouteValid(route) {
    if (route.length < 4) {
      console.log(`[GA] ❌ Route too short: ${route.length} nodes (minimum 4)`);
      return false;
    }
    
    // ENHANCED DEBUG: Show actual route with both indices and names
    const routeDebugInfo = route.map(idx => {
      const node = this.nodes[idx];
      return `${idx}:"${node?.id || 'UNKNOWN'}"`;
    }).join(' → ');
    console.log(`[GA] 🔍 Validating route (idx:name): ${routeDebugInfo}`);
    
    const pickupPos = route.indexOf(this.pickupIndex);
    const deliveryPos = route.indexOf(this.deliveryIndex);
    
    // Must contain both pickup and delivery, pickup before delivery
    if (pickupPos < 0) {
      console.log(`[GA] ❌ Route missing pickup point: ${this.pickupNodeId} (index ${this.pickupIndex})`);
      console.log(`[GA]     Searching for index ${this.pickupIndex} in route: [${route.join(', ')}]`);
      console.log(`[GA]     Route names: ${route.map(i => this.nodes[i]?.id).join(' → ')}`);
      
      // Check if pickup exists by name instead of index
      const pickupByName = route.find(idx => this.nodes[idx]?.id === this.pickupNodeId);
      if (pickupByName !== undefined) {
        console.error(`[GA] 🚨 CRITICAL: Pickup "${this.pickupNodeId}" found at index ${pickupByName}, but expected at ${this.pickupIndex}!`);
        console.error(`[GA] 🚨 This indicates index mapping corruption!`);
      }
      
      return false;
    }
    
    if (deliveryPos < 0) {
      console.log(`[GA] ❌ Route missing delivery point: ${this.deliveryNodeId} (index ${this.deliveryIndex})`);
      console.log(`[GA]     Searching for index ${this.deliveryIndex} in route: [${route.join(', ')}]`);
      console.log(`[GA]     Route names: ${route.map(i => this.nodes[i]?.id).join(' → ')}`);
      
      // Check if delivery exists by name instead of index
      const deliveryByName = route.find(idx => this.nodes[idx]?.id === this.deliveryNodeId);
      if (deliveryByName !== undefined) {
        console.error(`[GA] 🚨 CRITICAL: Delivery "${this.deliveryNodeId}" found at index ${deliveryByName}, but expected at ${this.deliveryIndex}!`);
        console.error(`[GA] 🚨 This indicates index mapping corruption!`);
      }
      
      return false;
    }
    
    if (pickupPos >= deliveryPos) {
      console.log(`[GA] ❌ Wrong order: pickup at ${pickupPos}, delivery at ${deliveryPos}`);
      return false;
    }
    
    // Check that all consecutive nodes are connected in the graph
    for (let i = 0; i < route.length - 1; i++) {
      const fromIndex = route[i];
      const toIndex = route[i + 1];
      
      const neighbors = this.adjacencyList.get(fromIndex) || [];
      if (!neighbors.includes(toIndex)) {
        console.log(`[GA] ❌ Invalid edge: ${fromIndex} (${this.nodes[fromIndex]?.id}) -> ${toIndex} (${this.nodes[toIndex]?.id})`);
        return false;
      }
    }
    
    console.log(`[GA] ✅ Route is valid: ${route.map(i => this.nodes[i]?.id).join(' → ')}`);
    return true;
  }

  /**
   * Tournament selection
   */
  tournamentSelection(population) {
    const tournament = [];
    
    for (let i = 0; i < GA_CONFIG.TOURNAMENT_SIZE; i++) {
      const randomIndex = Math.floor(Math.random() * population.length);
      tournament.push(population[randomIndex]);
    }
    
    // Return best individual from tournament (lowest fitness)
    return tournament.reduce((best, individual) => 
      individual.fitness < best.fitness ? individual : best
    );
  }

  /**
   * Improved crossover that maintains graph connectivity
   */
  crossover(parent1, parent2) {
    if (Math.random() > GA_CONFIG.CROSSOVER_RATE) {
      return [parent1.copy(), parent2.copy()];
    }
    
    // Simple approach: take segments from each parent and repair if needed
    const route1 = parent1.route;
    const route2 = parent2.route;
    
    // Find pickup and delivery positions in both routes
    const p1PickupPos = route1.indexOf(this.pickupIndex);
    const p1DeliveryPos = route1.indexOf(this.deliveryIndex);
    const p2PickupPos = route2.indexOf(this.pickupIndex);
    const p2DeliveryPos = route2.indexOf(this.deliveryIndex);
    
    // Create children by taking different segments
    const child1 = this.buildValidRoute(route1, route2, p1PickupPos, p1DeliveryPos);
    const child2 = this.buildValidRoute(route2, route1, p2PickupPos, p2DeliveryPos);
    
    return [new Individual(child1), new Individual(child2)];
  }

  /**
   * Build a valid route by combining segments from two parents
   */
  buildValidRoute(primaryRoute, secondaryRoute, pickupPos, deliveryPos) {
    // Choose hubs closer to pickup/delivery
    const startHub = this.selectClosestHub(this.pickupIndex);
    const endHub = this.selectClosestHub(this.deliveryIndex);
    
    // ALWAYS use the complete valid route builder to ensure pickup/delivery inclusion
    return this.buildCompleteValidRoute(startHub, endHub);
  }

  /**
   * Enhanced mutation that ensures valid connectivity AND preserves pickup/delivery
   */
  mutate(individual) {
    if (Math.random() > GA_CONFIG.MUTATION_RATE) {
      return individual;
    }
    
    const route = [...individual.route];
    const mutationType = Math.random();
    
    // CRITICAL: Find pickup and delivery positions to protect them
    const pickupPos = route.indexOf(this.pickupIndex);
    const deliveryPos = route.indexOf(this.deliveryIndex);
    
    if (pickupPos < 0 || deliveryPos < 0) {
      console.warn(`[GA] 🚨 Mutation: Critical points missing in route before mutation!`);
      // Rebuild the route completely to ensure pickup/delivery are present
      return new Individual(this.buildCompleteValidRoute(route[0], route[route.length - 1]));
    }
    
    if (mutationType < 0.4) {
      // Insert charging station detour (but not between pickup and delivery)
      this.insertValidChargingDetour(route, pickupPos, deliveryPos);
    } else if (mutationType < 0.7) {
      // Remove unnecessary intermediate nodes (but NEVER pickup/delivery)
      this.removeValidIntermediate(route, pickupPos, deliveryPos);
    } else {
      // Rebuild a segment with better path (but preserve pickup/delivery)
      this.rebuildSegment(route, pickupPos, deliveryPos);
    }
    
    // CRITICAL: Verify pickup and delivery are still present after mutation
    const newPickupPos = route.indexOf(this.pickupIndex);
    const newDeliveryPos = route.indexOf(this.deliveryIndex);
    
    if (newPickupPos < 0 || newDeliveryPos < 0 || newPickupPos >= newDeliveryPos) {
      console.warn(`[GA] 🚨 Mutation damaged route! Rebuilding...`);
      // Emergency rebuild with forced pickup/delivery inclusion
      return new Individual(this.buildCompleteValidRoute(route[0], route[route.length - 1]));
    }
    
    return new Individual(route);
  }

  /**
   * Insert a valid charging station detour (protected version)
   */
  insertValidChargingDetour(route, pickupPos, deliveryPos) {
    if (route.length >= GA_CONFIG.MAX_ROUTE_LENGTH) return;
    
    // Only allow insertions that don't affect pickup→delivery path
    for (let i = 0; i < route.length - 1; i++) {
      // SKIP any positions that could affect pickup→delivery sequence
      if (i >= pickupPos && i <= deliveryPos) continue;
      
      const fromNode = route[i];
      const toNode = route[i + 1];
      
      // Find charging stations that can create valid detour
      const validDetours = this.chargingIndices.filter(chargingNode => {
        if (route.includes(chargingNode)) return false;
        
        const pathToCharging = this.findValidPath(fromNode, chargingNode, 3);
        const pathFromCharging = this.findValidPath(chargingNode, toNode, 3);
        
        return pathToCharging && pathFromCharging;
      });
      
      if (validDetours.length > 0) {
        const selectedCharging = validDetours[Math.floor(Math.random() * validDetours.length)];
        
        // Build the detour path
        const pathToCharging = this.findValidPath(fromNode, selectedCharging, 3);
        const pathFromCharging = this.findValidPath(selectedCharging, toNode, 3);
        
        if (pathToCharging && pathFromCharging && pathToCharging.length > 1 && pathFromCharging.length > 1) {
          // Remove direct connection and insert detour
          const detourPath = [...pathToCharging.slice(1), ...pathFromCharging.slice(1, -1)];
          route.splice(i + 1, 0, ...detourPath);
          return; // Only do one detour per mutation
        }
      }
    }
  }

  /**
   * Remove intermediate nodes while preserving pickup/delivery (protected version)
   */
  removeValidIntermediate(route, pickupPos, deliveryPos) {
    if (route.length <= 4) return;
    
    // Find removable segments (NEVER remove pickup or delivery)
    for (let i = 1; i < route.length - 1; i++) {
      if (i === pickupPos || i === deliveryPos) continue; // PROTECT critical points
      
      const prevNode = route[i - 1];
      const nextNode = route[i + 1];
      
      // Check if we can create direct valid path
      const directPath = this.findValidPath(prevNode, nextNode, 3);
      if (directPath && directPath.length === 2) {
        route.splice(i, 1);
        return; // Only remove one node per mutation
      }
    }
  }

  /**
   * Rebuild a segment of the route with better path (protected version)
   */
  rebuildSegment(route, pickupPos, deliveryPos) {
    if (route.length < 4) return;
    
    // Choose random segment to rebuild, but NEVER include pickup/delivery positions
    const availableSegments = [];
    for (let start = 1; start < route.length - 2; start++) {
      for (let end = start + 1; end < route.length - 1; end++) {
        // Skip segments that contain pickup or delivery
        if ((start <= pickupPos && pickupPos <= end) || 
            (start <= deliveryPos && deliveryPos <= end)) {
          continue;
        }
        availableSegments.push({ start, end });
      }
    }
    
    if (availableSegments.length === 0) return; // No safe segments to rebuild
    
    const { start: segmentStart, end: segmentEnd } = availableSegments[Math.floor(Math.random() * availableSegments.length)];
    
    const fromNode = route[segmentStart - 1];
    const toNode = route[segmentEnd + 1];
    
    // Try to find better path
    const newPath = this.findValidPath(fromNode, toNode, 5);
    if (newPath && newPath.length > 2) {
      // Replace segment with new path
      const replacement = newPath.slice(1, -1); // Remove start and end nodes
      route.splice(segmentStart, segmentEnd - segmentStart + 1, ...replacement);
    }
  }

  /**
   * Run the genetic algorithm
   */
  run() {
    console.log(`[GA] 🚁 Starting Genetic Algorithm`);
    console.log(`[GA] 📊 Configuration: Population=${GA_CONFIG.POPULATION_SIZE}, Generations=${GA_CONFIG.GENERATIONS}`);
    console.log(`[GA] 🎯 Mission: ${this.pickupNodeId} → ${this.deliveryNodeId}`);
    console.log(`[GA] 🔋 Battery: ${this.batteryCapacity}%, Payload: ${this.maxPayload}kg`);
    console.log(`[GA] ===============================================`);
    
    // Initialize population
    let population = this.initializePopulation();
    
    // Evaluate initial population
    population.forEach(individual => this.evaluateFitness(individual));
    
    let bestFitness = Infinity;
    let bestIndividual = null;
    let generationsSinceImprovement = 0;
    
    // Track fitness evolution
    const fitnessHistory = [];
    const validRouteHistory = [];
    
    // Main evolution loop
    for (let generation = 0; generation < GA_CONFIG.GENERATIONS; generation++) {
      // Sort population by fitness
      population.sort((a, b) => a.fitness - b.fitness);
      
      // Calculate generation statistics
      const genBest = population[0].fitness;
      const genWorst = population[population.length - 1].fitness;
      const genAvg = population.reduce((sum, ind) => sum + ind.fitness, 0) / population.length;
      const validCount = population.filter(ind => ind.isValid).length;
      
      fitnessHistory.push({ generation, best: genBest, avg: genAvg, worst: genWorst });
      validRouteHistory.push({ generation, validCount, totalCount: population.length });
      
      // Track best individual
      if (population[0].fitness < bestFitness) {
        bestFitness = population[0].fitness;
        bestIndividual = population[0].copy();
        generationsSinceImprovement = 0;
        
        console.log(`[GA] 🎉 Gen ${generation}: NEW BEST FITNESS = ${bestFitness.toFixed(2)}`);
        console.log(`[GA]     Route: ${bestIndividual.route.map(i => this.nodes[i]?.id).join(' → ')}`);
        console.log(`[GA]     Valid: ${bestIndividual.isValid ? '✅' : '❌'}, Steps: ${bestIndividual.route.length}`);
        
        if (bestIndividual.isValid) {
          const finalBattery = bestIndividual.batteryHistory[bestIndividual.batteryHistory.length - 1] || 0;
          console.log(`[GA]     Battery remaining: ${finalBattery.toFixed(1)}%`);
        }
      } else {
        generationsSinceImprovement++;
      }
      
      // Generation progress log every 5 generations
      if (generation % 5 === 0 || generation === GA_CONFIG.GENERATIONS - 1) {
        console.log(`[GA] 📈 Gen ${generation}/${GA_CONFIG.GENERATIONS}: Best=${genBest.toFixed(2)}, Avg=${genAvg.toFixed(2)}, Valid=${validCount}/${population.length}`);
      }
      
      // Early stopping if no improvement
      if (generationsSinceImprovement > 20) {
        console.log(`[GA] ⏹️  Early stopping at generation ${generation} (no improvement for 20 gens)`);
        break;
      }
      
      // Create new population
      const newPopulation = [];
      
      // Elitism: keep best individuals
      for (let i = 0; i < GA_CONFIG.ELITE_SIZE; i++) {
        newPopulation.push(population[i].copy());
      }
      
      // Generate offspring
      while (newPopulation.length < GA_CONFIG.POPULATION_SIZE) {
        const parent1 = this.tournamentSelection(population);
        const parent2 = this.tournamentSelection(population);
        
        const [child1, child2] = this.crossover(parent1, parent2);
        
        const mutatedChild1 = this.mutate(child1);
        const mutatedChild2 = this.mutate(child2);
        
        this.evaluateFitness(mutatedChild1);
        this.evaluateFitness(mutatedChild2);
        
        newPopulation.push(mutatedChild1);
        if (newPopulation.length < GA_CONFIG.POPULATION_SIZE) {
          newPopulation.push(mutatedChild2);
        }
      }
      
      population = newPopulation;
    }
    
    // Final result
    population.sort((a, b) => a.fitness - b.fitness);
    const finalBest = population[0];
    
    console.log(`[GA] ===============================================`);
    console.log(`[GA] 🏁 ALGORITHM COMPLETED`);
    console.log(`[GA] 🏆 Best fitness: ${finalBest.fitness.toFixed(2)}`);
    console.log(`[GA] 📏 Route length: ${finalBest.route.length} steps`);
    console.log(`[GA] ✅ Valid: ${finalBest.isValid}`);
    console.log(`[GA] 🛤️  Route: ${finalBest.route.map(index => this.nodes[index]?.id || `Unknown-${index}`).join(' → ')}`);
    
    if (finalBest.isValid) {
      const finalBattery = finalBest.batteryHistory[finalBest.batteryHistory.length - 1] || 0;
      const batteryUsed = this.batteryCapacity - finalBattery;
      console.log(`[GA] 🔋 Battery used: ${batteryUsed.toFixed(1)}%, Remaining: ${finalBattery.toFixed(1)}%`);
    }
    
    // Display fitness evolution summary
    console.log(`[GA] 📊 FITNESS EVOLUTION:`);
    const step = Math.max(1, Math.floor(fitnessHistory.length / 10));
    for (let i = 0; i < fitnessHistory.length; i += step) {
      const data = fitnessHistory[i];
      console.log(`[GA]     Gen ${data.generation.toString().padStart(3)}: ${data.best.toFixed(2).padStart(10)} (avg: ${data.avg.toFixed(2)})`);
    }
    
    // Display valid routes evolution
    console.log(`[GA] 🎯 VALID ROUTES EVOLUTION:`);
    for (let i = 0; i < validRouteHistory.length; i += step) {
      const data = validRouteHistory[i];
      const percentage = ((data.validCount / data.totalCount) * 100).toFixed(1);
      console.log(`[GA]     Gen ${data.generation.toString().padStart(3)}: ${data.validCount}/${data.totalCount} (${percentage}%)`);
    }
    
    console.log(`[GA] ===============================================`);
    
    return {
      success: finalBest.isValid && finalBest.fitness < GA_CONFIG.INVALID_ROUTE_PENALTY,
      route_indices: finalBest.route,
      route_names: finalBest.route.map(index => this.nodes[index]?.id || `Unknown-${index}`),
      battery_history: finalBest.batteryHistory,
      fitness: finalBest.fitness,
      generations: Math.min(GA_CONFIG.GENERATIONS, generationsSinceImprovement + 1),
      stats: {
        success: finalBest.isValid,
        steps: finalBest.route.length,
        battery_used: this.batteryCapacity - (finalBest.batteryHistory[finalBest.batteryHistory.length - 1] || 0),
        battery_final: finalBest.batteryHistory[finalBest.batteryHistory.length - 1] || 0,
        total_cost: finalBest.fitness
      },
      // Add evolution data for analysis
      fitness_evolution: fitnessHistory,
      valid_routes_evolution: validRouteHistory
    };
  }
}
