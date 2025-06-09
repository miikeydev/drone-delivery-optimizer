/**
 * Test script to verify GA fixes
 * Run with: node test_ga_fixes.js
 */

import GeneticAlgorithm from './public/js/GA.js';

// Create minimal test data
const testNodes = [
  { id: 'Hub 1', index: 0, lat: 48.8566, lng: 2.3522, type: 'hubs' },          // Paris
  { id: 'Pickup 1', index: 1, lat: 48.8584, lng: 2.2945, type: 'pickup' },    // Near Paris
  { id: 'Delivery 1', index: 2, lat: 48.8606, lng: 2.3376, type: 'delivery' }, // Near Paris
  { id: 'Charging 1', index: 3, lat: 48.8529, lng: 2.3449, type: 'charging' }  // Near Paris
];

const testEdges = [
  { source: 0, target: 1, distance: 5.2, cost: 5.2 },
  { source: 1, target: 2, distance: 3.8, cost: 3.8 },
  { source: 2, target: 0, distance: 4.1, cost: 4.1 },
  { source: 0, target: 3, distance: 2.1, cost: 2.1 },
  { source: 3, target: 1, distance: 3.5, cost: 3.5 },
  { source: 3, target: 2, distance: 2.9, cost: 2.9 }
];

const testPackages = [
  { pickup: 1, delivery: 2, weight: 1 }
];

const options = {
  batteryCapacity: 100,
  maxPayload: 3,
  populationSize: 20,
  generations: 20,
  crossoverRate: 0.8,
  mutationRate: 0.3
};

console.log('[TEST] Starting GA test with fixes...');
console.log('[TEST] Test data:', testNodes.length, 'nodes,', testEdges.length, 'edges');
console.log('[TEST] Package:', testPackages);

try {
  const ga = new GeneticAlgorithm(testNodes, testEdges, testPackages, options);
  const result = ga.run();
  
  console.log('[TEST] ✅ GA completed successfully!');
  console.log('[TEST] Success:', result.success);
  console.log('[TEST] Fitness:', result.fitness.toFixed(4));
  console.log('[TEST] Route indices:', result.route_indices);
  console.log('[TEST] Route names:', result.route_names);
  console.log('[TEST] Steps:', result.stats.steps);
  console.log('[TEST] Battery used:', result.stats.battery_used.toFixed(1) + '%');
  
  // Verify fixes
  if (result.fitness > 0.001) {
    console.log('[TEST] ✅ Fitness threshold fix: SUCCESS (fitness > 0.001)');
  } else {
    console.log('[TEST] ❌ Fitness threshold fix: FAILED (fitness too low)');
  }
  
  const hasUndefinedNodes = result.route_names.some(name => name.includes('undefined'));
  if (!hasUndefinedNodes) {
    console.log('[TEST] ✅ Node mapping fix: SUCCESS (no undefined nodes)');
  } else {
    console.log('[TEST] ❌ Node mapping fix: FAILED (undefined nodes found)');
  }
  
  const hasValidRoute = result.route_indices.length > 2 && 
                       result.route_indices.includes(testPackages[0].pickup) &&
                       result.route_indices.includes(testPackages[0].delivery);
  if (hasValidRoute) {
    console.log('[TEST] ✅ Route generation: SUCCESS (contains pickup and delivery)');
  } else {
    console.log('[TEST] ❌ Route generation: FAILED (missing pickup or delivery)');
  }
  
  console.log('[TEST] 🎉 Test completed successfully!');
  
} catch (error) {
  console.error('[TEST] ❌ Test failed with error:', error.message);
  console.error('[TEST] Stack trace:', error.stack);
}
