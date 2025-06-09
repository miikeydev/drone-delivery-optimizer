/**
 * Script de test pour déboguer l'algorithme génétique
 */

// Simuler un environnement de navigateur minimal
global.performance = {
  now: () => Date.now()
};

// Importer les modules nécessaires
import GeneticAlgorithm from './public/js/GA.js';

// Créer des données de test simplifiées
const testNodes = [
  { index: 0, id: 'hub_1', type: 'hubs', lat: 48.8566, lng: 2.3522 },
  { index: 1, id: 'pickup_1', type: 'normal', lat: 48.8566, lng: 2.3522 },
  { index: 2, id: 'delivery_1', type: 'normal', lat: 48.8566, lng: 2.3522 },
  { index: 3, id: 'charging_1', type: 'charging', lat: 48.8566, lng: 2.3522 }
];

const testEdges = [
  { source: 0, target: 1, distance: 1000 },
  { source: 1, target: 2, distance: 1500 },
  { source: 2, target: 0, distance: 1200 },
  { source: 0, target: 3, distance: 800 },
  { source: 3, target: 1, distance: 900 },
  { source: 3, target: 2, distance: 1100 }
];

const testPackages = [
  { pickup: 1, delivery: 2, weight: 1.5 }
];

console.log('Testing GA with simplified data...');
console.log('Nodes:', testNodes.length);
console.log('Edges:', testEdges.length);
console.log('Packages:', testPackages.length);

try {
  const ga = new GeneticAlgorithm(testNodes, testEdges, testPackages, {
    populationSize: 10,
    generations: 5,
    crossoverRate: 0.8,
    mutationRate: 0.1
  });
  
  console.log('GA instance created successfully');
  console.log('Starting GA execution...');
  
  const result = ga.run();
  
  console.log('GA completed successfully!');
  console.log('Result:', result);
  
} catch (error) {
  console.error('GA Test failed:', error);
  console.error('Stack trace:', error.stack);
}
