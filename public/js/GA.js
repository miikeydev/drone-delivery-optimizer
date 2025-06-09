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

/** Return distance between two node indices using the edge list if available, otherwise haversine */
function buildDistanceLookup(nodes, edges) {
  const map = new Map();
  for (const e of edges) {
    const key = `${Math.min(e.u, e.v)}-${Math.max(e.u, e.v)}`;
    map.set(key, e.dist ?? e.distance);
  }
  return function (u, v) {
    if (u === v) return 0;
    const key = `${Math.min(u, v)}-${Math.max(u, v)}`;
    if (map.has(key)) return map.get(key);
    // fallback to haversine
    const a = nodes[u];
    const b = nodes[v];
    return haversineDistance(a.lat, a.lng, b.lat, b.lng);
  };
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
      kNorm = 1.2,
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

    // Precompute a default start & end hub
    this.startHub = this.hubs[0];
    this.endHub = this.hubs[0]; // circular mission default
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
    }
      // Format the result for the application
    const success = best.fitness > 0.1; // Consider successful if fitness is reasonable
    const route_indices = best.route;const route_names = route_indices.map(idx => {
      const node = this.nodes[idx];
      return node ? node.id : `Node(${idx})`;
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
  }
  _nextGeneration() {
    console.log('[GA] _nextGeneration: Starting...');
    
    // Sort population by fitness DESC (because we maximise fitness)
    console.log('[GA] _nextGeneration: Sorting population...');
    this.population.sort((a, b) => b.fitness - a.fitness);

    const newPop = [];
    
    // Elitism – copy best N individuals
    console.log('[GA] _nextGeneration: Adding elite individuals...');
    for (let i = 0; i < this.elitismCount; i++) {
      newPop.push({ ...this.population[i] });
    }
    console.log(`[GA] _nextGeneration: Added ${this.elitismCount} elite individuals`);

    // Fill the rest of population
    console.log('[GA] _nextGeneration: Filling rest of population...');
    let iterationCount = 0;
    const maxIterations = this.popSize * 10; // Safety limit
    
    while (newPop.length < this.popSize) {
      iterationCount++;
      if (iterationCount > maxIterations) {
        console.error('[GA] _nextGeneration: Too many iterations, breaking to prevent infinite loop');
        break;
      }
      
      if (iterationCount % 10 === 0) {
        console.log(`[GA] _nextGeneration: Iteration ${iterationCount}, population size: ${newPop.length}/${this.popSize}`);
      }
      
      console.log(`[GA] _nextGeneration: Selecting parents (iteration ${iterationCount})...`);
      const parent1 = this._tournamentSelect();
      const parent2 = this._tournamentSelect();

      let [child1Route, child2Route] = [parent1.route.slice(), parent2.route.slice()];

      console.log(`[GA] _nextGeneration: Applying crossover/mutation (iteration ${iterationCount})...`);
      if (RNG() < this.crossoverRate) {
        console.log(`[GA] _nextGeneration: Performing crossover...`);
        [child1Route, child2Route] = this._crossover(parent1.route, parent2.route);
        console.log(`[GA] _nextGeneration: Crossover completed`);
      }
      if (RNG() < this.mutationRate) {
        console.log(`[GA] _nextGeneration: Mutating child1...`);
        child1Route = this._mutate(child1Route);
        console.log(`[GA] _nextGeneration: Child1 mutation completed`);
      }
      if (RNG() < this.mutationRate) {
        console.log(`[GA] _nextGeneration: Mutating child2...`);
        child2Route = this._mutate(child2Route);
        console.log(`[GA] _nextGeneration: Child2 mutation completed`);
      }

      console.log(`[GA] _nextGeneration: Adding children to population...`);
      newPop.push({ route: child1Route });
      if (newPop.length < this.popSize) {
        newPop.push({ route: child2Route });
      }
      console.log(`[GA] _nextGeneration: Children added, population size now: ${newPop.length}`);
    }
    
    console.log(`[GA] _nextGeneration: Completed after ${iterationCount} iterations`);
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
    console.log('[GA] _crossover: Starting crossover operation...');
    
    const size = parent1.length;
    const idx1 = randInt(size);
    const idx2 = randInt(size);
    const [start, end] = [Math.min(idx1, idx2), Math.max(idx1, idx2)];
    
    console.log(`[GA] _crossover: Crossover points: [${start}, ${end}] for size ${size}`);

    const child1 = Array(size).fill(null);
    const child2 = Array(size).fill(null);

    // Copy slice from parent into child
    for (let i = start; i <= end; i++) {
      child1[i] = parent1[i];
      child2[i] = parent2[i];
    }
    console.log('[GA] _crossover: Copied slices to children');

    // Fill remaining positions preserving order & validity
    const fillChild = (child, donor, childName) => {
      console.log(`[GA] _crossover: Filling ${childName}...`);
      
      // Create a set of already used nodes for fast lookup
      const usedNodes = new Set(child.filter(n => n !== null));
      console.log(`[GA] _crossover: ${childName} already has ${usedNodes.size} nodes`);
      
      // Create a list of available nodes from donor in order
      const availableNodes = donor.filter(node => !usedNodes.has(node));
      console.log(`[GA] _crossover: ${childName} has ${availableNodes.length} available nodes from donor`);
      
      let availableIdx = 0;
      
      for (let i = 0; i < size; i++) {
        if (child[i] !== null) continue;
        
        if (availableIdx < availableNodes.length) {
          child[i] = availableNodes[availableIdx];
          availableIdx++;
        } else {
          // Fallback: use any remaining node from the original parent
          console.warn(`[GA] _crossover: ${childName} ran out of available nodes at position ${i}`);
          const allNodes = [...new Set([...parent1, ...parent2])];
          const currentUsed = new Set(child.filter(n => n !== null));
          const unusedNode = allNodes.find(n => !currentUsed.has(n));
          child[i] = unusedNode || donor[i % donor.length]; // Ultimate fallback
        }
      }
      
      console.log(`[GA] _crossover: ${childName} filled, calling repair...`);
      return this._repairRoute(child);
    };

    const result = [fillChild(child1, parent2, 'child1'), fillChild(child2, parent1, 'child2')];
    console.log('[GA] _crossover: Crossover completed');
    return result;
  }
  /** Simple mutation: swap two non‑hub nodes & repair */
  _mutate(route) {
    console.log('[GA] _mutate: Starting mutation...');
    
    const len = route.length;
    if (len <= 2) {
      console.log('[GA] _mutate: Route too short for mutation, returning as-is');
      return route;
    }
    
    let i = randInt(len - 2) + 1;      // avoid first (hub)
    let j = randInt(len - 2) + 1;
    if (i === j) j = (j + 1) % (len - 1) + 1;
    
    console.log(`[GA] _mutate: Swapping positions ${i} and ${j} (values: ${route[i]}, ${route[j]})`);
    [route[i], route[j]] = [route[j], route[i]];
    
    console.log('[GA] _mutate: Calling repair after mutation...');
    const result = this._repairRoute(route);
    console.log('[GA] _mutate: Mutation completed');
    return result;
  }

  /* =====================================================
     ROUTE MANIPULATION & VALIDATION
     ===================================================== */

  /** Generate a random valid route obeying pickup‑delivery precedence */
  _randomRoute() {
    // 1. Start with the list of pickups & deliveries interleaved randomly
    const core = [];
    for (const p of this.packages) {
      core.push(p.pickup, p.delivery);
    }
    shuffle(core);

    // 2. Repair precedence constraint (move delivery after its pickup)
    for (const { pickup, delivery } of this.packages) {
      const pIdx = core.indexOf(pickup);
      const dIdx = core.indexOf(delivery);
      if (dIdx < pIdx) {
        core.splice(dIdx, 1);      // remove delivery
        core.splice(pIdx + 1, 0, delivery); // re‑insert just after pickup
      }
    }

    // 3. Optionally insert charging nodes (random chance)
    const route = [this.startHub, ...core, this.endHub];
    return route;
  }

  /** Ensure uniqueness of nodes, pickup -> delivery precedence, keep hubs fixed */
  _repairRoute(route) {
    // Remove duplicates except hubs
    const seen = new Set([this.startHub, this.endHub]);
    const cleaned = [this.startHub];

    for (let i = 1; i < route.length - 1; i++) {
      const node = route[i];
      if (!seen.has(node)) {
        cleaned.push(node);
        seen.add(node);
      }
    }
    cleaned.push(this.endHub);

    // Ensure every pickup comes before its delivery
    for (const { pickup, delivery } of this.packages) {
      const pIdx = cleaned.indexOf(pickup);
      const dIdx = cleaned.indexOf(delivery);
      if (pIdx === -1 && dIdx === -1) {
        // Neither present (should not happen) – insert both
        cleaned.splice(cleaned.length - 1, 0, pickup, delivery);
      } else if (pIdx === -1) {
        // Only delivery present – insert pickup before it
        cleaned.splice(dIdx, 0, pickup);
      } else if (dIdx === -1) {
        // Only pickup present – insert delivery after
        cleaned.splice(pIdx + 1, 0, delivery);
      } else if (dIdx < pIdx) {
        // Wrong order – move delivery after pickup
        cleaned.splice(dIdx, 1);
        cleaned.splice(pIdx + 1, 0, delivery);
      }
    }
    return cleaned;
  }

  /* =====================================================
     FITNESS CALCULATION
     ===================================================== */
  _fitness(route) {
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
      const d = this.dist(u, v);

      const e = energyLeg(d, payload);
      battery -= e;
      energyUsed += e;
      distTotal += d;

      if (battery < 0) {
        penalty += 10000 * Math.abs(battery);
        battery = 0;
      }

      // Arrive at node v – update payload
      if (this.weightByPickup.has(v)) {
        payload += this.weightByPickup.get(v);
      }
      if (this.weightByDelivery.has(v)) {
        payload -= this.weightByDelivery.get(v);
        if (payload < 0) {
          penalty += 1000 * Math.abs(payload);
          payload = 0;
        }
      }

      // Handle charging point
      if (this.chargingNodes.has(v)) {
        battery = this.batteryCapacity; // full charge
        chargeTime += this.chargeDuration;
      }
    }

    const flightTime = distTotal / this.cruiseSpeed;
    const cost =
      this.weights.energy * energyUsed +
      this.weights.distance * distTotal +
      this.weights.time * (flightTime + chargeTime) +
      penalty;

    const score = 1 / (1 + cost); // maximise
    return { score, details: { energyUsed, distTotal, flightTime, chargeTime, penalty } };
  }
  _getBestIndividual() {
    return this.population.reduce((best, indiv) => (indiv.fitness > best.fitness ? indiv : best));
  }
  _calculateBatteryHistory(route) {
    const history = [];
    let battery = this.batteryCapacity;
    let payload = 0;
    history.push(battery);

    for (let i = 0; i < route.length - 1; i++) {
      const u = route[i];
      const v = route[i + 1];
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
