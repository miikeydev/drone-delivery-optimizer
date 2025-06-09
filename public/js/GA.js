import { RNG } from './utils.js';
import { buildDistanceLookup, findGraphPath } from './pathfinding.js';

function randInt(max) {
  return Math.floor(RNG() * max);
}

function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = randInt(i + 1);
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

export default class GeneticAlgorithm {
  constructor(nodes, edges, packages = [], opts = {}) {
    this.nodes = nodes;
    this.edges = edges;
    this.packages = packages;

    this.dist = buildDistanceLookup(nodes, edges);
    this.weightByPickup = new Map();
    this.weightByDelivery = new Map();
    for (const p of packages) {
      const w = p.weight ?? 1;
      this.weightByPickup.set(p.pickup, w);
      this.weightByDelivery.set(p.delivery, w);
    }

    const {
      populationSize = 50,
      generations = 50,
      crossoverRate = 0.8,
      mutationRate = 0.2,
      tournamentSize = 3,
      elitismCount = 1,
      batteryCapacity = 100,
      kNorm = 12.0,
      alpha = 0.4,
      cruiseSpeed = 60,
      chargeDuration = 0.25,
      weights = { energy: 1, distance: 0.05, time: 0.1 },
    } = opts;

    Object.assign(this, {
      popSize: populationSize, generations, crossoverRate, mutationRate,
      tournamentSize, elitismCount, batteryCapacity, kNorm, alpha,
      cruiseSpeed, chargeDuration, weights
    });

    this.hubs = nodes.filter(n => n.type === 'hubs').map(n => n.index);
    this.chargingNodes = new Set(nodes.filter(n => n.type === 'charging').map(n => n.index));
    this.pickups = Array.from(new Set(packages.map(p => p.pickup)));
    this.deliveries = Array.from(new Set(packages.map(p => p.delivery)));

    if (this.hubs.length === 0) {
      throw new Error('At least one hub is required');
    }

    if (packages.length > 0) {
      const firstPackage = packages[0];
      this.startHub = this._findClosestHub(firstPackage.pickup);
      this.endHub = this._findClosestHub(firstPackage.delivery);
    } else {
      this.startHub = this.hubs[0];
      this.endHub = this.hubs[0];
    }

    this._buildAdjacencyList();
  }

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

  _buildAdjacencyList() {
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

  run() {
    console.log('[GA] Starting algorithm...');
    const algorithmStart = performance.now();
    
    this.population = this._initialPopulation();

    for (let gen = 0; gen < this.generations; gen++) {
      this._evaluatePopulation();
      this.population = this._nextGeneration();
    }

    this._evaluatePopulation();
    const best = this._getBestIndividual();
    
    const algorithmEnd = performance.now();
    console.log(`[GA] Completed in ${(algorithmEnd - algorithmStart).toFixed(2)}ms`);

    const success = best.fitness > 0.001;
    const route_indices = best.route;
    const route_names = route_indices.map(idx => {
      if (idx < 0 || idx >= this.nodes.length) {
        return `InvalidNode(${idx})`;
      }
      const node = this.nodes[idx];
      return node && node.id ? node.id : `Node(undefined_${idx})`;
    });
    
    const battery_history = this._calculateBatteryHistory(best.route);
    const totalDistance = best.details.distTotal || 0;
    const energyUsed = best.details.energyUsed || 0;
    const batteryUsedPercent = (energyUsed / this.batteryCapacity) * 100;
    const finalBatteryPercent = battery_history.length > 0 ? 
      (battery_history[battery_history.length - 1] / this.batteryCapacity) * 100 : 0;
    
    return {
      success, route_indices, route_names, battery_history,
      fitness: best.fitness, details: best.details,
      stats: {
        steps: route_indices.length, total_distance: totalDistance,
        energy_used: energyUsed, battery_used: batteryUsedPercent,
        battery_final: finalBatteryPercent,
        flight_time: best.details.flightTime || 0,
        charge_time: best.details.chargeTime || 0
      }
    };
  }

  _initialPopulation() {
    const pop = [];
    for (let i = 0; i < this.popSize; i++) {
      pop.push({ route: this._randomRoute() });
    }
    return pop;
  }

  _evaluatePopulation() {
    for (const indiv of this.population) {
      if (indiv.fitness === undefined) {
        const { score, details } = this._fitness(indiv.route);
        indiv.fitness = score;
        indiv.details = details;
      }
    }
  }

  _nextGeneration() {
    this.population.sort((a, b) => b.fitness - a.fitness);
    const newPop = [];
    
    for (let i = 0; i < this.elitismCount; i++) {
      newPop.push({ ...this.population[i] });
    }

    while (newPop.length < this.popSize) {
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
  }

  _crossover(parent1, parent2) {
    if (!parent1 || !parent2 || parent1.length < 2 || parent2.length < 2) {
      return [this._randomRoute(), this._randomRoute()];
    }
    
    const cleanParent1 = parent1.filter(node => 
      typeof node === 'number' && node >= 0 && node < this.nodes.length
    );
    const cleanParent2 = parent2.filter(node => 
      typeof node === 'number' && node >= 0 && node < this.nodes.length
    );
    
    if (cleanParent1.length < 2 || cleanParent2.length < 2) {
      return [this._randomRoute(), this._randomRoute()];
    }
    
    const size = Math.max(cleanParent1.length, cleanParent2.length);
    const idx1 = randInt(Math.min(cleanParent1.length, cleanParent2.length));
    const idx2 = randInt(Math.min(cleanParent1.length, cleanParent2.length));
    const [start, end] = [Math.min(idx1, idx2), Math.max(idx1, idx2)];

    const child1 = Array(size).fill(null);
    const child2 = Array(size).fill(null);

    for (let i = start; i <= end && i < cleanParent1.length && i < cleanParent2.length; i++) {
      child1[i] = cleanParent1[i];
      child2[i] = cleanParent2[i];
    }

    const fillChild = (child, donor) => {
      const validDonor = donor.filter(node => 
        typeof node === 'number' && node >= 0 && node < this.nodes.length
      );
      
      const usedNodes = new Set(child.filter(n => n !== null));
      const availableNodes = validDonor.filter(node => !usedNodes.has(node));
      
      let availableIdx = 0;
      
      for (let i = 0; i < size; i++) {
        if (child[i] !== null) continue;
        
        if (availableIdx < availableNodes.length) {
          child[i] = availableNodes[availableIdx];
          availableIdx++;
        } else {
          child[i] = this.startHub;
        }
      }
      
      return this._repairRoute(child);
    };

    return [fillChild(child1, cleanParent2), fillChild(child2, cleanParent1)];
  }

  _mutate(route) {
    const len = route.length;
    if (len <= 2) return route;
    
    let i = randInt(len - 2) + 1;
    let j = randInt(len - 2) + 1;
    if (i === j) j = (j + 1) % (len - 1) + 1;
    
    [route[i], route[j]] = [route[j], route[i]];
    
    return this._repairRoute(route);
  }

  _randomRoute() {
    if (this.packages.length === 0) {
      return [this.startHub, this.endHub];
    }
    
    const package1 = this.packages[0];
    
    if (package1.pickup >= this.nodes.length || package1.delivery >= this.nodes.length) {
      return [this.startHub, this.endHub];
    }
    
    const route = [];
    
    const hubToPickupPath = findGraphPath(this.startHub, package1.pickup, this._adjacencyList);
    route.push(...hubToPickupPath);
    
    const pickupToDeliveryPath = findGraphPath(package1.pickup, package1.delivery, this._adjacencyList);
    route.push(...pickupToDeliveryPath.slice(1));
    
    const deliveryToHubPath = findGraphPath(package1.delivery, this.endHub, this._adjacencyList);
    route.push(...deliveryToHubPath.slice(1));
    
    return route;
  }

  _repairRoute(route) {
    const validRoute = route.filter(node => {
      return typeof node === 'number' && node >= 0 && node < this.nodes.length;
    });
    
    if (validRoute.length < 2) {
      return this._randomRoute();
    }
    
    if (this.packages.length === 0) {
      return [this.startHub, this.endHub];
    }
    
    const package1 = this.packages[0];
    const repairedRoute = [];
    const segments = [
      { from: this.startHub, to: package1.pickup },
      { from: package1.pickup, to: package1.delivery },
      { from: package1.delivery, to: this.endHub }
    ];
    
    repairedRoute.push(this.startHub);
    
    for (const segment of segments) {
      const path = findGraphPath(segment.from, segment.to, this._adjacencyList);
      if (path.length > 1) {
        repairedRoute.push(...path.slice(1));
      } else if (!repairedRoute.includes(segment.to)) {
        repairedRoute.push(segment.to);
      }
    }
    
    return repairedRoute;
  }

  _fitness(route) {
    let payload = 0;
    let battery = this.batteryCapacity;
    let energyUsed = 0;
    let distTotal = 0;
    let chargeTime = 0;
    let penalty = 0;

    const energyLeg = (d, load) => d / this.kNorm * (1 + this.alpha * load);

    for (let i = 0; i < route.length - 1; i++) {
      const u = route[i];
      const v = route[i + 1];
      
      if (u >= this.nodes.length || v >= this.nodes.length) {
        penalty += 50000;
        continue;
      }
      
      const d = this.dist(u, v);
      const e = energyLeg(d, payload);
      battery -= e;
      energyUsed += e;
      distTotal += d;

      if (battery < 0) {
        penalty += 100 * Math.abs(battery);
        battery = Math.max(battery, -this.batteryCapacity * 0.1);
      }

      if (this.weightByPickup.has(v)) {
        payload += this.weightByPickup.get(v);
      }
      if (this.weightByDelivery.has(v)) {
        payload -= this.weightByDelivery.get(v);
        if (payload < 0) {
          penalty += 100 * Math.abs(payload);
          payload = 0;
        }
      }

      if (this.chargingNodes.has(v)) {
        battery = this.batteryCapacity;
        chargeTime += this.chargeDuration;
      }
    }

    let completionBonus = 0;
    if (penalty < 100) {
      const flightTime = distTotal / this.cruiseSpeed;
      const baseCost = 
        this.weights.energy * energyUsed +
        this.weights.distance * distTotal +
        this.weights.time * (flightTime + chargeTime);
      
      completionBonus = baseCost * 0.25;
    }

    const flightTime = distTotal / this.cruiseSpeed;
    const baseCost = 
      this.weights.energy * energyUsed +
      this.weights.distance * distTotal +
      this.weights.time * (flightTime + chargeTime);
      
    const totalCost = baseCost + penalty - completionBonus;
    const score = Math.max(0.001, 1000 / (1 + totalCost));
    
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
      
      if (u >= this.nodes.length || v >= this.nodes.length) {
        history.push(Math.max(0, battery));
        continue;
      }
      
      const d = this.dist(u, v);
      const energyUsed = d / this.kNorm * (1 + this.alpha * payload);
      battery -= energyUsed;
      
      if (this.weightByPickup.has(v)) {
        payload += this.weightByPickup.get(v);
      }
      if (this.weightByDelivery.has(v)) {
        payload -= this.weightByDelivery.get(v);
        payload = Math.max(0, payload);
      }
      
      if (this.chargingNodes.has(v)) {
        battery = this.batteryCapacity;
      }
      
      history.push(Math.max(0, battery));
    }
    
    return history;
  }
}

export { GeneticAlgorithm };
