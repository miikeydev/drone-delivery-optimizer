// genetic-algorithm.js
// Implémentation de l'algorithme génétique pour l'optimisation de tournées de drones

// Constantes pour l'algorithme génétique
const POPULATION_SIZE = 50;        // Taille de la population
const MAX_GENERATIONS = 100;       // Nombre maximum de générations
const MUTATION_RATE = 0.2;         // Probabilité de mutation
const CROSSOVER_RATE = 0.8;        // Probabilité de croisement
const ELITE_SIZE = 5;              // Nombre d'élites à préserver
const TOURNAMENT_SIZE = 3;         // Taille du tournoi pour la sélection

// Constantes pour la fonction de fitness
const LAMBDA = 1000;               // Pénalité pour dépassement de batterie (λ)
const MU = 10;                     // Pénalité légère pour les recharges (μ)
const BATTERY_MAX = 100;           // Capacité maximale de la batterie (Bₘₐₓ)
const K_NORMALIZATION = 10.8;      // Facteur de normalisation k pour Δbᵢ
const ALPHA_SURCHARGE = 0.2;       // Facteur de surcharge α pour colis multiples

// Structure de données auxiliaire pour la recherche du plus court chemin
class PriorityQueue {
  constructor() {
    this.values = [];
  }
  
  enqueue(val, priority) {
    this.values.push({val, priority});
    this.sort();
  }
  
  dequeue() {
    return this.values.shift();
  }
  
  sort() {
    this.values.sort((a, b) => a.priority - b.priority);
  }
  
  isEmpty() {
    return this.values.length === 0;
  }
}

/**
 * Trouve le chemin le plus court entre deux nœuds en utilisant Dijkstra
 * @param {Array} nodes - Liste des nœuds
 * @param {Array} edges - Liste des arêtes
 * @param {number} startIdx - Index du nœud de départ
 * @param {number} endIdx - Index du nœud d'arrivée
 * @returns {Array} - Liste des indices des nœuds formant le chemin le plus court
 */
function findShortestPath(nodes, edges, startIdx, endIdx) {
  const distances = {};
  const previous = {};
  const queue = new PriorityQueue();
  
  // Initialisation
  for (let i = 0; i < nodes.length; i++) {
    if (i === startIdx) {
      distances[i] = 0;
      queue.enqueue(i, 0);
    } else {
      distances[i] = Infinity;
    }
    previous[i] = null;
  }
  
  // Créer un mapping des arêtes pour un accès plus rapide
  const edgeMap = {};
  edges.forEach(edge => {
    if (!edgeMap[edge.source]) {
      edgeMap[edge.source] = [];
    }
    edgeMap[edge.source].push({
      target: edge.target,
      cost: edge.cost
    });
  });
  
  // Algorithme de Dijkstra
  while (!queue.isEmpty()) {
    const current = queue.dequeue().val;
    
    if (current === endIdx) break;
    
    if (edgeMap[current]) {
      for (const neighbor of edgeMap[current]) {
        const alt = distances[current] + neighbor.cost;
        if (alt < distances[neighbor.target]) {
          distances[neighbor.target] = alt;
          previous[neighbor.target] = current;
          queue.enqueue(neighbor.target, alt);
        }
      }
    }
  }
  
  // Reconstruction du chemin
  const path = [];
  let current = endIdx;
  
  while (current !== null) {
    path.unshift(current);
    current = previous[current];
  }
  
  return path.length > 1 ? path : null; // Retourne null si aucun chemin n'est trouvé
}

/**
 * Calcule la consommation de batterie pour une arête
 * @param {Object} edge - L'arête à évaluer
 * @param {number} payloadCount - Nombre de colis transportés
 * @returns {number} - Consommation de batterie pour cette arête
 */
function calculateBatteryConsumption(edge, payloadCount) {
  return (edge.cost / K_NORMALIZATION) * (1 + ALPHA_SURCHARGE * (payloadCount - 1));
}

/**
 * Génère un individu aléatoire (chemin) valide
 * @param {Array} nodes - Liste des nœuds
 * @param {Array} edges - Liste des arêtes
 * @param {number} specifiedStartHub - Index du hub de départ spécifié (optionnel)
 * @param {number} specifiedEndHub - Index du hub d'arrivée spécifié (optionnel)
 * @returns {Array} - Séquence d'indices de nœuds représentant un chemin
 */
function generateRandomIndividual(nodes, edges, specifiedStartHub = null, specifiedEndHub = null) {
  // Trouver les hubs, pickup et delivery points
  const hubIndices = [];
  const pickups = [];
  const deliveries = [];
  
  // Parcourir tous les nœuds pour identifier les types
  nodes.forEach((node, idx) => {
    if (node.type === 'hubs') hubIndices.push(idx);
    else if (node.type === 'pickup') pickups.push(idx);
    else if (node.type === 'delivery') deliveries.push(idx);
  });
  
  // S'il n'y a pas suffisamment de points pour créer un chemin valide
  if (hubIndices.length === 0 || pickups.length === 0 || deliveries.length === 0) {
    return null;
  }
  
  // Utiliser les hubs spécifiés s'ils sont fournis, sinon choisir aléatoirement
  const startHubIdx = specifiedStartHub !== null ? specifiedStartHub : hubIndices[Math.floor(Math.random() * hubIndices.length)];
  const pickupIdx = pickups[Math.floor(Math.random() * pickups.length)];
  const deliveryIdx = deliveries[Math.floor(Math.random() * deliveries.length)];
  const endHubIdx = specifiedEndHub !== null ? specifiedEndHub : hubIndices[Math.floor(Math.random() * hubIndices.length)];
  
  // Créer un chemin : hub_start -> pickup -> delivery -> hub_end
  let path = [];
  
  // Chemin de hub_start à pickup
  const path1 = findShortestPath(nodes, edges, startHubIdx, pickupIdx);
  if (path1) path = path.concat(path1);
  
  // Chemin de pickup à delivery
  const path2 = findShortestPath(nodes, edges, pickupIdx, deliveryIdx);
  if (path2 && path2.length > 1) path = path.concat(path2.slice(1)); // Évite de répéter pickup
  
  // Chemin de delivery à hub_end
  const path3 = findShortestPath(nodes, edges, deliveryIdx, endHubIdx);
  if (path3 && path3.length > 1) path = path.concat(path3.slice(1)); // Évite de répéter delivery
  
  return path;
}

/**
 * Évalue la fitness d'un individu (chemin)
 * @param {Array} path - Le chemin à évaluer (séquence d'indices de nœuds)
 * @param {Array} nodes - Liste des nœuds
 * @param {Array} edges - Liste des arêtes
 * @param {number} maxBattery - Capacité maximale de batterie
 * @param {number} maxPayload - Capacité maximale en nombre de colis
 * @returns {number} - Score de fitness (plus petit est meilleur)
 */
function evaluateFitness(path, nodes, edges, maxBattery, maxPayload) {
  if (!path || path.length < 2) return Infinity; // Chemin invalide
  
  let totalCost = 0;
  let batteryLevel = maxBattery;
  let payloadCount = 0;
  let rechargeCount = 0;
  let batteryPenalty = 0;
  
  // Map pour retrouver facilement une arête entre deux nœuds
  const edgeMap = {};
  edges.forEach(edge => {
    if (!edgeMap[edge.source]) edgeMap[edge.source] = {};
    edgeMap[edge.source][edge.target] = edge;
  });
  
  // Parcourir le chemin et calculer les coûts et pénalités
  let segmentStart = 0; // Début du segment actuel (depuis la dernière recharge)
  
  for (let i = 0; i < path.length - 1; i++) {
    const currentNode = nodes[path[i]];
    const nextNode = nodes[path[i + 1]];
    const edge = edgeMap[path[i]] && edgeMap[path[i]][path[i + 1]];
    
    // Si l'arête n'existe pas, la fitness est infiniment mauvaise
    if (!edge) return Infinity;
    
    // Mettre à jour le nombre de colis
    if (currentNode.type === 'pickup') {
      payloadCount = Math.min(payloadCount + 1, maxPayload);
    } else if (currentNode.type === 'delivery' && payloadCount > 0) {
      payloadCount = Math.max(0, payloadCount - 1);
    }
    
    // Calculer la consommation de batterie pour cette arête
    const consumption = calculateBatteryConsumption(edge, payloadCount);
    
    // Si on est sur une station de recharge, réinitialiser la batterie
    if (currentNode.type === 'charging') {
      batteryLevel = maxBattery;
      rechargeCount++;
      segmentStart = i; // Début d'un nouveau segment
    }
    
    // Consommer la batterie
    batteryLevel -= consumption;
    
    // Ajouter le coût de base de l'arête
    totalCost += edge.cost;
    
    // Si la batterie est épuisée, appliquer une pénalité et considérer une recharge fictive
    if (batteryLevel < 0) {
      // Calculer la pénalité pour ce segment
      const segmentPenalty = Math.pow(Math.abs(batteryLevel), 2);
      batteryPenalty += segmentPenalty;
      
      // Réinitialiser comme si on avait rechargé
      batteryLevel = maxBattery - consumption;
      rechargeCount++;
      segmentStart = i;
    }
  }
  
  // Calculer la fonction de fitness selon J(E) donnée
  const fitness = totalCost + LAMBDA * batteryPenalty + MU * rechargeCount;
  
  return fitness;
}

/**
 * Sélectionne un parent par tournoi
 * @param {Array} population - Liste des individus
 * @param {Array} fitnessScores - Liste des scores de fitness
 * @returns {Array} - Le parent sélectionné
 */
function tournamentSelection(population, fitnessScores) {
  let bestIdx = Math.floor(Math.random() * population.length);
  let bestFitness = fitnessScores[bestIdx];
  
  for (let i = 1; i < TOURNAMENT_SIZE; i++) {
    const idx = Math.floor(Math.random() * population.length);
    if (fitnessScores[idx] < bestFitness) {
      bestIdx = idx;
      bestFitness = fitnessScores[idx];
    }
  }
  
  return population[bestIdx];
}

/**
 * Croise deux parents pour créer un enfant
 * @param {Array} parent1 - Premier parent
 * @param {Array} parent2 - Deuxième parent
 * @param {Array} nodes - Liste des nœuds
 * @param {Array} edges - Liste des arêtes
 * @returns {Array} - L'enfant créé
 */
function crossover(parent1, parent2, nodes, edges) {
  if (Math.random() > CROSSOVER_RATE) {
    return parent1.slice(); // Pas de croisement
  }
  
  // Point de croisement
  const cutPoint = Math.floor(Math.random() * (parent1.length - 1)) + 1;
  
  // Créer l'enfant en combinant les segments des parents
  const child = parent1.slice(0, cutPoint);
  
  // Ajouter un chemin du dernier nœud du segment de parent1 au premier nœud du segment de parent2
  const lastNode = parent1[cutPoint - 1];
  const nextNode = parent2[Math.floor(Math.random() * parent2.length)]; // Prendre un nœud aléatoire de parent2
  
  // Trouver un chemin entre ces deux nœuds
  const connectingPath = findShortestPath(nodes, edges, lastNode, nextNode);
  
  // Si un chemin existe, l'ajouter (en évitant de répéter le dernier nœud)
  if (connectingPath && connectingPath.length > 1) {
    child.push(...connectingPath.slice(1));
  }
  
  return child;
}

/**
 * Effectue une mutation sur un individu
 * @param {Array} individual - L'individu à muter
 * @param {Array} nodes - Liste des nœuds
 * @param {Array} edges - Liste des arêtes
 * @returns {Array} - L'individu muté
 */
function mutate(individual, nodes, edges) {
  if (Math.random() > MUTATION_RATE || individual.length < 3) {
    return individual.slice(); // Pas de mutation
  }
  
  const mutatedIndividual = individual.slice();
  
  // Type de mutation : remplacer un sous-chemin par un nouveau chemin
  const startIdx = Math.floor(Math.random() * (mutatedIndividual.length - 2));
  const endIdx = Math.min(startIdx + 2 + Math.floor(Math.random() * 3), mutatedIndividual.length - 1);
  
  const sourceNode = mutatedIndividual[startIdx];
  const targetNode = mutatedIndividual[endIdx];
  
  // Trouver un nouveau chemin
  const newPath = findShortestPath(nodes, edges, sourceNode, targetNode);
  
  // Si un nouveau chemin est trouvé, remplacer le segment
  if (newPath && newPath.length > 1) {
    mutatedIndividual.splice(startIdx, endIdx - startIdx + 1, ...newPath);
  }
  
  return mutatedIndividual;
}

/**
 * Répare un chromosome si nécessaire pour garantir qu'il commence et termine par les hubs spécifiés
 * @param {Array} individual - L'individu à réparer
 * @param {Array} nodes - Liste des nœuds
 * @param {Array} edges - Liste des arêtes
 * @param {number} startHubIdx - Index du hub de départ (optionnel)
 * @param {number} endHubIdx - Index du hub d'arrivée (optionnel)
 * @returns {Array} - L'individu réparé
 */
function repairChromosome(individual, nodes, edges, startHubIdx = null, endHubIdx = null) {
  if (individual.length < 2) return null; // Individu trop court
  
  let repairedIndividual = individual.slice();
  
  // Vérifier si le premier nœud est le hub de départ spécifié ou un hub quelconque
  if ((startHubIdx !== null && repairedIndividual[0] !== startHubIdx) || 
      (startHubIdx === null && nodes[repairedIndividual[0]].type !== 'hubs')) {
    
    // Hub à utiliser au début
    const startHub = startHubIdx !== null ? startHubIdx : nodes
      .map((node, idx) => node.type === 'hubs' ? idx : -1)
      .filter(idx => idx !== -1)
      [Math.floor(Math.random() * nodes.filter(n => n.type === 'hubs').length)];
    
    if (startHub !== undefined) {
      const path = findShortestPath(nodes, edges, startHub, repairedIndividual[0]);
      
      if (path && path.length > 1) {
        repairedIndividual = path.concat(repairedIndividual.slice(1));
      }
    }
  }
  
  // Vérifier si le dernier nœud est le hub d'arrivée spécifié ou un hub quelconque
  if ((endHubIdx !== null && repairedIndividual[repairedIndividual.length - 1] !== endHubIdx) || 
      (endHubIdx === null && nodes[repairedIndividual[repairedIndividual.length - 1]].type !== 'hubs')) {
    
    // Hub à utiliser à la fin
    const endHub = endHubIdx !== null ? endHubIdx : nodes
      .map((node, idx) => node.type === 'hubs' ? idx : -1)
      .filter(idx => idx !== -1)
      [Math.floor(Math.random() * nodes.filter(n => n.type === 'hubs').length)];
    
    if (endHub !== undefined) {
      const path = findShortestPath(nodes, edges, repairedIndividual[repairedIndividual.length - 1], endHub);
      
      if (path && path.length > 1) {
        repairedIndividual = repairedIndividual.concat(path.slice(1));
      }
    }
  }
  
  return repairedIndividual;
}

/**
 * Exécute l'algorithme génétique pour trouver une solution optimale
 * @param {Array} nodes - Liste des nœuds
 * @param {Array} edges - Liste des arêtes
 * @param {number} batteryCapacity - Capacité maximale de la batterie
 * @param {number} maxPayload - Capacité maximale de colis
 * @param {number} startHubIdx - Index du hub de départ (optionnel)
 * @param {number} endHubIdx - Index du hub d'arrivée (optionnel)
 * @returns {Object} - Résultat de l'algorithme génétique
 */
export function runGeneticAlgorithm(nodes, edges, batteryCapacity, maxPayload, startHubIdx = null, endHubIdx = null) {  console.log('Genetic Algorithm started', { 
    nodes: nodes.length, 
    edges: edges.length, 
    batteryCapacity, 
    maxPayload,
    startHubIdx,
    endHubIdx
  });
    // 1. Initialisation de la population
  let population = [];
  for (let i = 0; i < POPULATION_SIZE; i++) {
    const individual = generateRandomIndividual(nodes, edges, startHubIdx, endHubIdx);
    if (individual) {
      population.push(individual);
    }
  }
  
  if (population.length === 0) {
    alert('Impossible de générer une population initiale valide !');
    return null;
  }
  
  console.log(`Population initiale générée avec ${population.length} individus`);
  
  // Variables pour suivre les meilleures solutions
  let bestIndividual = null;
  let bestFitness = Infinity;
  let generationsWithoutImprovement = 0;
  
  // Historique pour visualiser l'évolution
  const fitnessHistory = [];
  
  // Boucle principale
  for (let generation = 0; generation < MAX_GENERATIONS; generation++) {
    // Évaluer la fitness de chaque individu
    const fitnessScores = population.map(individual => 
      evaluateFitness(individual, nodes, edges, batteryCapacity, maxPayload)
    );
    
    // Trouver le meilleur individu de cette génération
    const bestGenerationIdx = fitnessScores.indexOf(Math.min(...fitnessScores));
    const bestGenerationIndividual = population[bestGenerationIdx];
    const bestGenerationFitness = fitnessScores[bestGenerationIdx];
    
    // Sauvegarder si c'est la meilleure solution globale
    if (bestGenerationFitness < bestFitness) {
      bestIndividual = bestGenerationIndividual.slice();
      bestFitness = bestGenerationFitness;
      generationsWithoutImprovement = 0;
    } else {
      generationsWithoutImprovement++;
    }
    
    // Enregistrer les statistiques
    fitnessHistory.push({
      generation,
      best: bestGenerationFitness,
      average: fitnessScores.reduce((sum, f) => sum + f, 0) / fitnessScores.length
    });
    
    // Afficher les progrès
    if (generation % 10 === 0 || generation === MAX_GENERATIONS - 1) {
      console.log(`Génération ${generation}: Meilleur fitness = ${bestGenerationFitness.toFixed(2)}`);
    }
    
    // Créer une nouvelle génération
    const newPopulation = [];
    
    // Élitisme : conserver les meilleurs individus
    const elites = population
      .map((individual, idx) => ({ individual, fitness: fitnessScores[idx] }))
      .sort((a, b) => a.fitness - b.fitness)
      .slice(0, ELITE_SIZE)
      .map(item => item.individual);
    
    newPopulation.push(...elites);
    
    // Compléter avec de nouveaux individus
    while (newPopulation.length < POPULATION_SIZE) {
      // Sélection
      const parent1 = tournamentSelection(population, fitnessScores);
      const parent2 = tournamentSelection(population, fitnessScores);
      
      // Croisement
      let child = crossover(parent1, parent2, nodes, edges);
      
      // Mutation
      child = mutate(child, nodes, edges);
        // Réparation
      child = repairChromosome(child, nodes, edges, startHubIdx, endHubIdx);
      
      if (child) {
        newPopulation.push(child);
      }
    }
    
    // Remplacer l'ancienne population par la nouvelle
    population = newPopulation;
    
    // Critère d'arrêt
    if (generationsWithoutImprovement >= 20) {
      console.log(`Arrêt anticipé après ${generation} générations sans amélioration`);
      break;
    }
  }
    // Afficher le résultat
  if (bestIndividual) {
    console.log('Meilleure solution trouvée:', { 
      pathLength: bestIndividual.length, 
      fitness: bestFitness
    });
    
    // Calculer les détails des segments
    const segmentDetails = [];
    
    if (bestIndividual && bestIndividual.length > 1) {
      for (let i = 0; i < bestIndividual.length - 1; i++) {
        const startNode = bestIndividual[i];
        const endNode = bestIndividual[i + 1];
        
        // Trouver l'arête correspondante
        const edge = edges.find(e => e.source === startNode && e.target === endNode);
        
        if (edge) {
          // Calculer la consommation de batterie pour ce segment
          const payloadCount = nodes[startNode].type === 'pickup' ? 1 : 0;
          const batteryUsed = calculateBatteryConsumption(edge, payloadCount);
          
          segmentDetails.push({
            distance: edge.distance,
            cost: edge.cost,
            batteryUsed: batteryUsed
          });
        }
      }
    }
    
    // Calcul des statistiques globales
    const totalDistance = segmentDetails.reduce((sum, segment) => sum + segment.distance, 0);
    const totalCost = segmentDetails.reduce((sum, segment) => sum + segment.cost, 0);
    const batteryUsage = segmentDetails.reduce((sum, segment) => sum + segment.batteryUsed, 0);
    
    // Déterminer le nombre de recharges nécessaires
    let remainingBattery = 100;
    let recharges = 0;
    
    for (const segment of segmentDetails) {
      if (segment.batteryUsed > remainingBattery) {
        recharges++;
        remainingBattery = 100;
      }
      remainingBattery -= segment.batteryUsed;
    }
    
    return {
      bestRoute: bestIndividual,
      segmentDetails: segmentDetails,
      totalDistance: totalDistance,
      totalCost: totalCost,
      batteryUsage: batteryUsage,
      recharges: recharges,
      generation: MAX_GENERATIONS - generationsWithoutImprovement,
      bestFitness: bestFitness,
      fitnessHistory: fitnessHistory
    };
  }
  
  // Si aucune solution n'est trouvée
  return null;
}

// La fonction displayRoute a été déplacée vers app.js pour une meilleure séparation des responsabilités
