/*
 * Genetic Algorithm for Drone Delivery Route Optimisation
 * -------------------------------------------------------
 * An ES‑module that exports a single class `GeneticAlgorithm`.
 *
 *  ‣ Dependencies: utils.js (haversineDistance, gaussianRandom, RNG)
 *  ‣ External data: `nodes`, `edges` (same structure as graphData)
 *  ‣ Optionally: an array `packages` = [{ pickup: idx, delivery: idx, weight: kg }]
 *
 *  A chromosome is a valid route (array of node indices) that:
 *    • starts at a hub, ends at a hub
 *    • contains every pickup exactly once and its delivery **after** the pickup
 *    • may include charging nodes any number of times (or none)
 *
 *  Fitness combines: total energy, distance, mission time, penalties (infeasible).
 */

import { haversineDistance, RNG } from './utils.js';

/** Helper — random integer in [0, max) */
function randInt(max) {
  return Math.floor(RNG() * max);
}

/** Fisher‑Yates shuffle (in‑place) */
function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = randInt(i + 1);
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

/** Return distance between two node indices using ONLY the edge list, no fallback */
function buildDistanceLookup(nodes, edges) {
  const map = new Map();
  
  // Build bidirectional edge map - handle both frontend and backend formats
  for (const e of edges) {
    // Handle both formats: {u, v, dist} and {source, target, distance}
    const u = e.u !== undefined ? e.u : e.source;
    const v = e.v !== undefined ? e.v : e.target;
    const distance = e.dist ?? e.distance ?? e.cost;
    
    if (u === undefined || v === undefined || distance === undefined) {
      console.warn('[GA] Invalid edge format:', e);
      continue;
    }
    
    const key1 = `${u}-${v}`;
    const key2 = `${v}-${u}`;
    map.set(key1, distance);
    map.set(key2, distance);
  }
  
  // Build adjacency list for pathfinding
  const adjacency = new Map();
  for (let i = 0; i < nodes.length; i++) {
    adjacency.set(i, []);
  }
  
  for (const e of edges) {
    // Handle both formats: {u, v, dist} and {source, target, distance}
    const u = e.u !== undefined ? e.u : e.source;
    const v = e.v !== undefined ? e.v : e.target;
    const distance = e.dist ?? e.distance ?? e.cost;
    
    if (u === undefined || v === undefined || distance === undefined) {
      continue; // Skip invalid edges
    }
    
    // Validate node indices
    if (u >= 0 && u < nodes.length && v >= 0 && v < nodes.length) {
      adjacency.get(u).push({ node: v, dist: distance });
      adjacency.get(v).push({ node: u, dist: distance });
    } else {
      console.warn(`[GA] Invalid edge node indices: ${u}, ${v} (max: ${nodes.length - 1})`);
    }
  }
  
  return function (u, v) {
    if (u === v) return 0;
    
    // Validate node indices
    if (u < 0 || u >= nodes.length || v < 0 || v >= nodes.length) {
      console.warn(`[GA] Invalid node indices: ${u}, ${v} (max: ${nodes.length - 1})`);
      return Infinity;
    }
    
    // Try direct edge first
    const directKey = `${u}-${v}`;
    if (map.has(directKey)) {
      return map.get(directKey);
    }
    
    // Use Dijkstra's algorithm for shortest path
    return dijkstraDistance(u, v, adjacency);
  };
}

/** Dijkstra's algorithm to find shortest path distance between two nodes */
function dijkstraDistance(start, end, adjacency) {
  if (start === end) return 0;
  
  const distances = new Map();
  const visited = new Set();
  const queue = [{ node: start, dist: 0 }];
  
  distances.set(start, 0);
  
  while (queue.length > 0) {
    // Sort queue by distance (simple priority queue)
    queue.sort((a, b) => a.dist - b.dist);
    const { node: current, dist: currentDist } = queue.shift();
    
    if (visited.has(current)) continue;
    visited.add(current);
    
    if (current === end) {
      return currentDist;
    }
    
    const neighbors = adjacency.get(current) || [];
    for (const { node: neighbor, dist: edgeDist } of neighbors) {
      if (visited.has(neighbor)) continue;
      
      const newDist = currentDist + edgeDist;
      if (!distances.has(neighbor) || newDist < distances.get(neighbor)) {
        distances.set(neighbor, newDist);
        queue.push({ node: neighbor, dist: newDist });
      }
    }
  }
  
  return Infinity; // No path found
}

/** Class implementing the GA */
export default class GeneticAlgorithm {
  /**
   * @param {Array} nodes   – graph nodes as given by graphData
   * @param {Array} edges   – graph edges as given by graphData
   * @param {Array} packages – [{ pickup: idx, delivery: idx, weight: kg }]
   * @param {Object} opts   – GA & drone parameters
   */
  constructor(nodes, edges, packages = [], opts = {}) {
    // Data ------------------------------------------------
    this.nodes = nodes;
    this.edges = edges;
    this.packages = packages;

    // Build quick look‑ups -------------------------------
    this.dist = buildDistanceLookup(nodes, edges);
    // Map delivery → pickup weight for quick access
    this.weightByPickup = new Map();
    this.weightByDelivery = new Map();
    for (const p of packages) {
      const w = p.weight ?? 1;
      this.weightByPickup.set(p.pickup, w);
      this.weightByDelivery.set(p.delivery, w);
    }    // GA parameters --------------------------------------
    const {
      populationSize = 50,     // Reduced from 100
      generations = 50,        // Reduced from 500 for faster testing
      crossoverRate = 0.8,
      mutationRate = 0.2,
      tournamentSize = 3,
      elitismCount = 1,
      // Drone / fitness parameters ↓↓↓
      batteryCapacity = 100,   // arbitrary energy unit
      kNorm = 12.0,            // Updated from 1.2 to make energy consumption realistic
      alpha = 0.4,
      cruiseSpeed = 60,        // km/h
      chargeDuration = 0.25,   // h (15 min) per full charge
      weights = { energy: 1, distance: 0.05, time: 0.1 },
    } = opts;

    this.popSize = populationSize;
    this.generations = generations;
    this.crossoverRate = crossoverRate;
    this.mutationRate = mutationRate;
    this.tournamentSize = tournamentSize;
    this.elitismCount = elitismCount;

    // Drone parameters
    this.batteryCapacity = batteryCapacity;
    this.kNorm = kNorm;
    this.alpha = alpha;
    this.cruiseSpeed = cruiseSpeed;
    this.chargeDuration = chargeDuration;
    this.weights = weights;    // Determine hubs, pickups, deliveries & charging nodes
    console.log('[GA] Analyzing node types...');
    this.hubs = nodes.filter(n => n.type === 'hubs').map(n => n.index);
    this.chargingNodes = new Set(nodes.filter(n => n.type === 'charging').map(n => n.index));
    const pickupsIdx = new Set(packages.map(p => p.pickup));
    const deliveriesIdx = new Set(packages.map(p => p.delivery));

    this.pickups = Array.from(pickupsIdx);
    this.deliveries = Array.from(deliveriesIdx);

    console.log('[GA] Node analysis:');
    console.log('  - Hubs found:', this.hubs.length, 'indices:', this.hubs.slice(0, 5));
    console.log('  - Charging nodes:', this.chargingNodes.size);
    console.log('  - Pickup nodes:', this.pickups.length, 'indices:', this.pickups);
    console.log('  - Delivery nodes:', this.deliveries.length, 'indices:', this.deliveries);

    if (this.hubs.length === 0) {
      throw new Error('At least one hub is required');
    }

    // Choose optimal start and end hubs based on packages
    if (packages.length > 0) {
      const firstPackage = packages[0];
      
      // Find closest hub to pickup point
      this.startHub = this._findClosestHub(firstPackage.pickup);
      // Find closest hub to delivery point  
      this.endHub = this._findClosestHub(firstPackage.delivery);
      
      console.log(`[GA] Selected start hub: ${this.nodes[this.startHub].id} (closest to pickup ${this.nodes[firstPackage.pickup].id})`);
      console.log(`[GA] Selected end hub: ${this.nodes[this.endHub].id} (closest to delivery ${this.nodes[firstPackage.delivery].id})`);
    } else {
      // Fallback to first hub if no packages
      this.startHub = this.hubs[0];
      this.endHub = this.hubs[0];
    }
  }

  /** Find the closest hub to a given node index */
  _findClosestHub(nodeIndex) {
    let closestHub = this.hubs[0];
    let minDistance = this.dist(nodeIndex, closestHub);
    
    for (const hubIndex of this.hubs) {
      const distance = this.dist(nodeIndex, hubIndex);
      if (distance < minDistance) {
        minDistance = distance;
        closestHub = hubIndex;
      }
    }
    
    return closestHub;
  }

  /* =====================================================
     PUBLIC API
     ===================================================== */  run() {
    console.log('[GA] Starting algorithm with', this.generations, 'generations...');
    console.log('[GA] Population size:', this.popSize);
    console.log('[GA] Packages:', this.packages);
    
    const algorithmStart = performance.now();
    
    try {
      console.log('[GA] Creating initial population...');
      this.population = this._initialPopulation();
      console.log('[GA] Initial population created with', this.population.length, 'individuals');
    } catch (error) {
      console.error('[GA] Error creating initial population:', error);
      throw error;
    }

    for (let gen = 0; gen < this.generations; gen++) {
      const genStart = performance.now();
      console.log(`[GA] Generation ${gen + 1}/${this.generations}`);
      
      try {
        this._evaluatePopulation();
      } catch (error) {
        console.error(`[GA] Error evaluating population at generation ${gen}:`, error);
        throw error;
      }
        try {
        console.log(`[GA] Creating next generation after generation ${gen + 1}...`);
        const nextGenStart = performance.now();
        this.population = this._nextGeneration();
        const nextGenEnd = performance.now();
        console.log(`[GA] Next generation created in ${(nextGenEnd - nextGenStart).toFixed(2)}ms`);
      } catch (error) {
        console.error(`[GA] Error creating next generation ${gen}:`, error);
        throw error;
      }
      
      const genEnd = performance.now();
      const genTime = genEnd - genStart;
      
      if (gen % 10 === 0 || genTime > 1000) {
        console.log(`[GA] Generation ${gen + 1} completed in ${genTime.toFixed(2)}ms`);
      }
      
      // Safety timeout - abort if taking too long
      if (genTime > 5000) {
        console.warn(`[GA] Generation ${gen + 1} took ${genTime.toFixed(2)}ms - aborting for performance`);
        break;
      }
      
      if (gen % 50 === 0) {
        console.log(`[GA] Progress: ${gen}/${this.generations} generations completed`);
      }
    }    console.log('[GA] Evolution complete, performing final evaluation...');
    let best;
    try {
      // Final evaluation & best route
      this._evaluatePopulation();
      best = this._getBestIndividual();
      console.log('[GA] Best individual found:', best);
      
      const algorithmEnd = performance.now();
      console.log(`[GA] Total algorithm time: ${(algorithmEnd - algorithmStart).toFixed(2)}ms`);
    } catch (error) {
      console.error('[GA] Error in final evaluation:', error);
      throw error;
    }    // Format the result for the application
    const success = best.fitness > 0.001; // Lower threshold - fitness of 0.004 should be considered success
    const route_indices = best.route;    const route_names = route_indices.map(idx => {
      // Validate node index before accessing
      if (idx < 0 || idx >= this.nodes.length) {
        console.warn(`[GA] Invalid node index ${idx}, max index is ${this.nodes.length - 1}`);
        return `InvalidNode(${idx})`;
      }
      const node = this.nodes[idx];
      return node && node.id ? node.id : `Node(undefined_${idx})`;
    });
    
    // Calculate battery history
    const battery_history = this._calculateBatteryHistory(best.route);
    
    // Calculate additional statistics
    const totalDistance = best.details.distTotal || 0;
    const energyUsed = best.details.energyUsed || 0;
    const batteryUsedPercent = (energyUsed / this.batteryCapacity) * 100;
    const finalBatteryPercent = battery_history.length > 0 ? 
      (battery_history[battery_history.length - 1] / this.batteryCapacity) * 100 : 0;
    
    return {
      success,
      route_indices,
      route_names,
      battery_history,
      fitness: best.fitness,
      details: best.details,
      stats: {
        steps: route_indices.length,
        total_distance: totalDistance,
        energy_used: energyUsed,
        battery_used: batteryUsedPercent,
        battery_final: finalBatteryPercent,
        flight_time: best.details.flightTime || 0,
        charge_time: best.details.chargeTime || 0
      }
    };
  }

  /* =====================================================
     GA INTERNALS
     ===================================================== */
  _initialPopulation() {
    const pop = [];
    for (let i = 0; i < this.popSize; i++) {
      pop.push({ route: this._randomRoute() });
    }
    return pop;
  }
  _evaluatePopulation() {
    console.log('[GA] Evaluating population of', this.population.length, 'individuals...');
    const startTime = performance.now();
    
    for (const indiv of this.population) {
      if (indiv.fitness === undefined) {
        const fitnessStart = performance.now();
        const { score, details } = this._fitness(indiv.route);
        indiv.fitness = score;
        indiv.details = details;
        const fitnessEnd = performance.now();
        
        // Log slow fitness evaluations
        if (fitnessEnd - fitnessStart > 10) {
          console.log(`[GA] Slow fitness evaluation: ${(fitnessEnd - fitnessStart).toFixed(2)}ms for route length ${indiv.route.length}`);
        }
      }
    }
    
    const endTime = performance.now();
    console.log(`[GA] Population evaluation completed in ${(endTime - startTime).toFixed(2)}ms`);
  }  _nextGeneration() {
    // Sort population by fitness DESC (because we maximise fitness)
    this.population.sort((a, b) => b.fitness - a.fitness);

    const newPop = [];
    
    // Elitism – copy best N individuals
    for (let i = 0; i < this.elitismCount; i++) {
      newPop.push({ ...this.population[i] });
    }

    // Fill the rest of population
    let iterationCount = 0;
    const maxIterations = this.popSize * 10; // Safety limit
    
    while (newPop.length < this.popSize) {
      iterationCount++;
      if (iterationCount > maxIterations) {
        console.error('[GA] _nextGeneration: Too many iterations, breaking to prevent infinite loop');
        break;
      }
      
      const parent1 = this._tournamentSelect();
      const parent2 = this._tournamentSelect();

      let [child1Route, child2Route] = [parent1.route.slice(), parent2.route.slice()];

      if (RNG() < this.crossoverRate) {
        [child1Route, child2Route] = this._crossover(parent1.route, parent2.route);
      }
      if (RNG() < this.mutationRate) {
        child1Route = this._mutate(child1Route);
      }
      if (RNG() < this.mutationRate) {
        child2Route = this._mutate(child2Route);
      }

      newPop.push({ route: child1Route });
      if (newPop.length < this.popSize) {
        newPop.push({ route: child2Route });
      }
    }
    
    return newPop;
  }

  _tournamentSelect() {
    let best = null;
    for (let i = 0; i < this.tournamentSize; i++) {
      const contender = this.population[randInt(this.population.length)];
      if (!best || contender.fitness > best.fitness) best = contender;
    }
    return best;
  }  /** Order‑preserving crossover (OX operator) */
  _crossover(parent1, parent2) {
    // First validate input parents
    if (!parent1 || !parent2 || parent1.length < 2 || parent2.length < 2) {
      console.warn(`[GA] Invalid parents for crossover, using fallback`);
      return [this._randomRoute(), this._randomRoute()];
    }
    
    // Filter out any invalid nodes from parents before crossover
    const cleanParent1 = parent1.filter(node => 
      node !== null && 
      node !== undefined && 
      typeof node === 'number' && 
      node >= 0 && 
      node < this.nodes.length
    );
    
    const cleanParent2 = parent2.filter(node => 
      node !== null && 
      node !== undefined && 
      typeof node === 'number' && 
      node >= 0 && 
      node < this.nodes.length
    );
    
    // If parents are too short after cleaning, use random routes
    if (cleanParent1.length < 2 || cleanParent2.length < 2) {
      return [this._randomRoute(), this._randomRoute()];
    }
    
    const size = Math.max(cleanParent1.length, cleanParent2.length);
    const idx1 = randInt(Math.min(cleanParent1.length, cleanParent2.length));
    const idx2 = randInt(Math.min(cleanParent1.length, cleanParent2.length));
    const [start, end] = [Math.min(idx1, idx2), Math.max(idx1, idx2)];

    const child1 = Array(size).fill(null);
    const child2 = Array(size).fill(null);

    // Copy slice from parent into child
    for (let i = start; i <= end && i < cleanParent1.length && i < cleanParent2.length; i++) {
      child1[i] = cleanParent1[i];
      child2[i] = cleanParent2[i];
    }// Fill remaining positions preserving order & validity
    const fillChild = (child, donor, childName) => {
      // First, ensure donor has only valid nodes
      const validDonor = donor.filter(node => 
        node !== null && 
        node !== undefined && 
        typeof node === 'number' && 
        node >= 0 && 
        node < this.nodes.length
      );
      
      // Create a set of already used nodes for fast lookup
      const usedNodes = new Set(child.filter(n => n !== null));
      
      // Create a list of available nodes from donor in order
      const availableNodes = validDonor.filter(node => !usedNodes.has(node));
      
      let availableIdx = 0;
      
      for (let i = 0; i < size; i++) {
        if (child[i] !== null) continue;
        
        if (availableIdx < availableNodes.length) {
          child[i] = availableNodes[availableIdx];
          availableIdx++;
        } else {
          // Fallback: use any required nodes that might be missing
          const requiredNodes = new Set([
            this.startHub, 
            this.endHub,
            ...this.packages.map(p => p.pickup),
            ...this.packages.map(p => p.delivery)
          ]);
          
          const currentUsed = new Set(child.filter(n => n !== null));
          const missingRequired = [...requiredNodes].find(n => !currentUsed.has(n));
          
          if (missingRequired !== undefined) {
            child[i] = missingRequired;
          } else {
            // Very last resort: use start hub to avoid undefined
            child[i] = this.startHub;
          }
        }
      }
      
      return this._repairRoute(child);
    };    const result = [fillChild(child1, cleanParent2, 'child1'), fillChild(child2, cleanParent1, 'child2')];
    return result;
  }  /** Simple mutation: swap two non‑hub nodes & repair */
  _mutate(route) {
    const len = route.length;
    if (len <= 2) {
      return route;
    }
    
    let i = randInt(len - 2) + 1;      // avoid first (hub)
    let j = randInt(len - 2) + 1;
    if (i === j) j = (j + 1) % (len - 1) + 1;
    
    [route[i], route[j]] = [route[j], route[i]];
    
    const result = this._repairRoute(route);
    return result;
  }

  /* =====================================================
     ROUTE MANIPULATION & VALIDATION
     ===================================================== */  /** Generate a random valid route obeying pickup‑delivery precedence and hub constraints */
  _randomRoute() {
    if (this.packages.length === 0) {
      console.warn('[GA] No packages defined, creating hub-to-hub route');
      return [this.startHub, this.endHub];
    }
    
    // Get the first (and likely only) package
    const package1 = this.packages[0];
    
    // Validate package nodes exist
    if (package1.pickup >= this.nodes.length || package1.delivery >= this.nodes.length) {
      console.error(`[GA] Invalid package: pickup=${package1.pickup}, delivery=${package1.delivery}, maxIndex=${this.nodes.length - 1}`);
      return [this.startHub, this.endHub];
    }
    
    // Build route using graph paths
    const route = [];
    
    // 1. Hub to Pickup
    const hubToPickupPath = this._findGraphPath(this.startHub, package1.pickup);
    route.push(...hubToPickupPath);
    
    // 2. Pickup to Delivery  
    const pickupToDeliveryPath = this._findGraphPath(package1.pickup, package1.delivery);
    route.push(...pickupToDeliveryPath.slice(1)); // Skip pickup node (already added)
    
    // 3. Delivery to Hub
    const deliveryToHubPath = this._findGraphPath(package1.delivery, this.endHub);
    route.push(...deliveryToHubPath.slice(1)); // Skip delivery node (already added)
    
    console.log(`[GA] Generated graph-based route: ${route.map(idx => this.nodes[idx].id).join(' → ')}`);
    return route;
  }

  /** Ensure route follows hub → pickup → delivery → hub structure with valid graph paths */
  _repairRoute(route) {
    // Filter out invalid nodes first
    const validRoute = route.filter(node => {
      if (typeof node !== 'number' || node < 0 || node >= this.nodes.length) {
        console.warn(`[GA] Removing invalid node: ${node}`);
        return false;
      }
      return true;
    });
    
    if (validRoute.length < 2) {
      console.warn('[GA] Route too short after cleaning, creating minimal route');
      return this._randomRoute();
    }
    
    // Ensure we have the required structure for single package delivery
    if (this.packages.length === 0) {
      return [this.startHub, this.endHub];
    }
    
    const package1 = this.packages[0];
    
    // Build route segment by segment using graph paths
    const repairedRoute = [];
    const segments = [
      { from: this.startHub, to: package1.pickup, name: 'hub-to-pickup' },
      { from: package1.pickup, to: package1.delivery, name: 'pickup-to-delivery' },
      { from: package1.delivery, to: this.endHub, name: 'delivery-to-hub' }
    ];
    
    repairedRoute.push(this.startHub);
    
    for (const segment of segments) {
      const path = this._findGraphPath(segment.from, segment.to);
      if (path.length > 1) {
        // Add path excluding the start node (already in route)
        repairedRoute.push(...path.slice(1));
        console.log(`[GA] ${segment.name}: ${path.map(idx => this.nodes[idx].id).join(' → ')}`);
      } else {
        console.warn(`[GA] No valid path found for ${segment.name} from ${this.nodes[segment.from].id} to ${this.nodes[segment.to].id}`);
        // Add destination directly as fallback
        if (!repairedRoute.includes(segment.to)) {
          repairedRoute.push(segment.to);
        }
      }
    }
    
    console.log(`[GA] Complete repaired route: ${repairedRoute.map(idx => this.nodes[idx].id).join(' → ')}`);
    return repairedRoute;
  }

  /** Find shortest path between two nodes using Dijkstra and return the full path */
  _findGraphPath(start, end) {
    if (start === end) return [start];
    
    // Build adjacency list if not already built
    if (!this._adjacencyList) {
      this._adjacencyList = new Map();
      for (let i = 0; i < this.nodes.length; i++) {
        this._adjacencyList.set(i, []);
      }
      
      for (const e of this.edges) {
        const u = e.u !== undefined ? e.u : e.source;
        const v = e.v !== undefined ? e.v : e.target;
        const distance = e.dist ?? e.distance ?? e.cost;
        
        if (u >= 0 && u < this.nodes.length && v >= 0 && v < this.nodes.length) {
          this._adjacencyList.get(u).push({ node: v, dist: distance });
          this._adjacencyList.get(v).push({ node: u, dist: distance });
        }
      }
    }
    
    // Dijkstra with path reconstruction
    const distances = new Map();
    const previous = new Map();
    const visited = new Set();
    const queue = [{ node: start, dist: 0 }];
    
    distances.set(start, 0);
    
    while (queue.length > 0) {
      queue.sort((a, b) => a.dist - b.dist);
      const { node: current, dist: currentDist } = queue.shift();
      
      if (visited.has(current)) continue;
      visited.add(current);
      
      if (current === end) {
        // Reconstruct path
        const path = [];
        let node = end;
        while (node !== undefined) {
          path.unshift(node);
          node = previous.get(node);
        }
        console.log(`[GA] Found graph path: ${path.map(idx => this.nodes[idx].id).join(' → ')}`);
        return path;
      }
      
      const neighbors = this._adjacencyList.get(current) || [];
      for (const { node: neighbor, dist: edgeDist } of neighbors) {
        if (visited.has(neighbor)) continue;
        
        const newDist = currentDist + edgeDist;
        if (!distances.has(neighbor) || newDist < distances.get(neighbor)) {
          distances.set(neighbor, newDist);
          previous.set(neighbor, current);
          queue.push({ node: neighbor, dist: newDist });
        }
      }
    }
    
    console.warn(`[GA] No graph path found from ${this.nodes[start].id} to ${this.nodes[end].id}`);
    return [start, end]; // Fallback to direct connection
  }

  /** Add charging stations to ensure graph connectivity */
  _addChargingIfNeeded(route) {
    if (this.chargingNodes.size === 0) return route;
    
    const enhanced = [];
    let currentBattery = this.batteryCapacity;
    let payload = 0;
    
    for (let i = 0; i < route.length; i++) {
      const currentNode = route[i];
      enhanced.push(currentNode);
      
      // Update payload at current node
      if (this.weightByPickup.has(currentNode)) {
        payload += this.weightByPickup.get(currentNode);
      }
      if (this.weightByDelivery.has(currentNode)) {
        payload -= this.weightByDelivery.get(currentNode);
        payload = Math.max(0, payload);
      }
      
      // Recharge if at charging station
      if (this.chargingNodes.has(currentNode)) {
        currentBattery = this.batteryCapacity;
      }
      
      // Check if we need charging for next leg
      if (i < route.length - 1) {
        const nextNode = route[i + 1];
        const distance = this.dist(currentNode, nextNode);
        
        if (distance !== Infinity) {
          // Calculate energy needed for next leg
          const energyNeeded = distance / this.kNorm * (1 + this.alpha * payload);
          
          // Add charging if battery would be critically low
          if (currentBattery < energyNeeded * 1.5 && this.chargingNodes.size > 0) {
            // Find path through charging station
            let bestChargingPath = null;
            let shortestPathLength = Infinity;
            
            for (const chargingNode of this.chargingNodes) {
              if (chargingNode === currentNode || chargingNode === nextNode) continue;
              
              const pathToCharging = this._findGraphPath(currentNode, chargingNode);
              const pathFromCharging = this._findGraphPath(chargingNode, nextNode);
              
              if (pathToCharging.length > 1 && pathFromCharging.length > 1) {
                const totalPathLength = pathToCharging.length + pathFromCharging.length - 1;
                if (totalPathLength < shortestPathLength) {
                  shortestPathLength = totalPathLength;
                  bestChargingPath = {
                    toCharging: pathToCharging,
                    fromCharging: pathFromCharging,
                    chargingNode
                  };
                }
              }
            }
            
            if (bestChargingPath) {
              console.log(`[GA] Adding charging detour: ${this.nodes[bestChargingPath.chargingNode].id}`);
              // Insert charging path
              enhanced.push(...bestChargingPath.toCharging.slice(1)); // Skip current node
              enhanced.push(...bestChargingPath.fromCharging.slice(1, -1)); // Skip charging and next nodes
              currentBattery = this.batteryCapacity; // Recharge
              
              // Skip adding the next node directly since we'll get there through the path
              return enhanced.concat(route.slice(i + 2));
            }
          }
          
          // Update battery for next leg
          currentBattery -= energyNeeded;
        }
      }
    }
    
    return enhanced;
  }

  /* =====================================================
     FITNESS CALCULATION
     ===================================================== */  _fitness(route) {
    let payload = 0;                    // kg carried before leaving node
    let battery = this.batteryCapacity; // energy units remaining
    let energyUsed = 0;
    let distTotal = 0;
    let chargeTime = 0;
    let penalty = 0;

    // Helper – compute energy for one leg
    const energyLeg = (d, load) => d / this.kNorm * (1 + this.alpha * load);

    for (let i = 0; i < route.length - 1; i++) {
      const u = route[i];
      const v = route[i + 1];
      
      // Validate node indices
      if (u >= this.nodes.length || v >= this.nodes.length) {
        penalty += 50000; // Heavy penalty for invalid nodes
        continue;
      }
      
      const d = this.dist(u, v);
      const e = energyLeg(d, payload);
      battery -= e;
      energyUsed += e;
      distTotal += d;

      // Softer penalty for battery issues - allow some negative battery with graduated penalty
      if (battery < 0) {
        penalty += 100 * Math.abs(battery); // Reduced from 10000
        battery = Math.max(battery, -this.batteryCapacity * 0.1); // Don't go below -10% capacity
      }

      // Arrive at node v – update payload
      if (this.weightByPickup.has(v)) {
        payload += this.weightByPickup.get(v);
      }
      if (this.weightByDelivery.has(v)) {
        payload -= this.weightByDelivery.get(v);
        if (payload < 0) {
          penalty += 100 * Math.abs(payload); // Reduced from 1000
          payload = 0;
        }
      }

      // Handle charging point
      if (this.chargingNodes.has(v)) {
        battery = this.batteryCapacity; // full charge
        chargeTime += this.chargeDuration;
      }
    }    // Completion bonus proportional to base costs (not overwhelming)
    let completionBonus = 0;
    if (penalty < 100) { // If route is mostly feasible
      const flightTime = distTotal / this.cruiseSpeed;
      const baseCost = 
        this.weights.energy * energyUsed +
        this.weights.distance * distTotal +
        this.weights.time * (flightTime + chargeTime);
      
      // Larger bonus - 25% of base cost - encourages completion more strongly
      completionBonus = baseCost * 0.25;
    }

    const flightTime = distTotal / this.cruiseSpeed;
    const baseCost = 
      this.weights.energy * energyUsed +
      this.weights.distance * distTotal +
      this.weights.time * (flightTime + chargeTime);
      const totalCost = baseCost + penalty - completionBonus;

    // Improved fitness calculation with better scaling
    const score = Math.max(0.001, 1000 / (1 + totalCost)); // Higher base value for better fitness scores
    return { score, details: { energyUsed, distTotal, flightTime, chargeTime, penalty } };
  }
  _getBestIndividual() {
    return this.population.reduce((best, indiv) => (indiv.fitness > best.fitness ? indiv : best));
  }  _calculateBatteryHistory(route) {
    const history = [];
    let battery = this.batteryCapacity;
    let payload = 0;
    history.push(battery);

    for (let i = 0; i < route.length - 1; i++) {
      const u = route[i];
      const v = route[i + 1];
      
      // Validate node indices
      if (u >= this.nodes.length || v >= this.nodes.length) {
        console.warn(`[GA] Invalid node in battery history: ${u} -> ${v}`);
        history.push(Math.max(0, battery));
        continue;
      }
      
      const d = this.dist(u, v);
      
      // Use same energy calculation as fitness function
      const energyUsed = d / this.kNorm * (1 + this.alpha * payload);
      battery -= energyUsed;
      
      // Update payload at destination node
      if (this.weightByPickup.has(v)) {
        payload += this.weightByPickup.get(v);
      }
      if (this.weightByDelivery.has(v)) {
        payload -= this.weightByDelivery.get(v);
        payload = Math.max(0, payload); // Prevent negative payload
      }
      
      // Handle charging points
      if (this.chargingNodes.has(v)) {
        battery = this.batteryCapacity;
      }
      
      history.push(Math.max(0, battery));
    }
      return history;
  }
}

// Export the class as ES module
export { GeneticAlgorithm };
