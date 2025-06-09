// Integration test for UI-Algorithm consistency
import { GeneticAlgorithm } from './public/js/GA.js';

console.log('[INTEGRATION TEST] Testing UI-Algorithm consistency...');

// Simulate network data similar to what the UI would use
const networkData = {
    nodes: [
        { id: 'hub1', type: 'hubs', lat: 48.8566, lng: 2.3522, name: 'Hub 1', index: 0 },
        { id: 'pickup1', type: 'pickup', lat: 48.8606, lng: 2.3376, name: 'Pickup Point 1', index: 1 },
        { id: 'delivery1', type: 'delivery', lat: 48.8530, lng: 2.3499, name: 'Delivery Point 1', index: 2 },
        { id: 'pickup2', type: 'pickup', lat: 48.8584, lng: 2.2945, name: 'Pickup Point 2', index: 3 },
        { id: 'delivery2', type: 'delivery', lat: 48.8499, lng: 2.3501, name: 'Delivery Point 2', index: 4 }
    ],
    edges: [
        { from: 'hub1', to: 'pickup1', distance: 2.1 },
        { from: 'hub1', to: 'delivery1', distance: 1.8 },
        { from: 'hub1', to: 'pickup2', distance: 3.2 },
        { from: 'hub1', to: 'delivery2', distance: 1.9 },
        { from: 'pickup1', to: 'delivery1', distance: 1.5 },
        { from: 'pickup1', to: 'pickup2', distance: 2.8 },
        { from: 'pickup1', to: 'delivery2', distance: 2.1 },
        { from: 'delivery1', to: 'pickup2', distance: 2.9 },
        { from: 'delivery1', to: 'delivery2', distance: 0.8 },
        { from: 'pickup2', to: 'delivery2', distance: 2.7 }
    ]
};

// Test different pickup-delivery combinations
const testCases = [
    {
        name: 'Single package',
        packages: [{ pickup: 1, delivery: 2, weight: 1.5 }],
        description: 'pickup1 -> delivery1'
    },
    {
        name: 'Two packages',
        packages: [
            { pickup: 1, delivery: 2, weight: 1.0 },
            { pickup: 3, delivery: 4, weight: 2.0 }
        ],
        description: 'pickup1 -> delivery1, pickup2 -> delivery2'
    },
    {
        name: 'Cross delivery',
        packages: [
            { pickup: 1, delivery: 4, weight: 1.5 },
            { pickup: 3, delivery: 2, weight: 1.8 }
        ],
        description: 'pickup1 -> delivery2, pickup2 -> delivery1'
    }
];

console.log(`[INTEGRATION TEST] Testing ${testCases.length} scenarios...`);

for (let i = 0; i < testCases.length; i++) {
    const testCase = testCases[i];
    console.log(`\n[TEST ${i + 1}] ${testCase.name}: ${testCase.description}`);
      try {
        const ga = new GeneticAlgorithm(networkData.nodes, networkData.edges, testCase.packages, {
            populationSize: 30,
            generations: 15,
            enableLogging: false // Reduced logging for cleaner output
        });
        
        const startTime = performance.now();
        const result = ga.run();
        const endTime = performance.now();
          if (result.success) {
            console.log(`✅ SUCCESS - Fitness: ${result.fitness.toFixed(4)}, Time: ${(endTime - startTime).toFixed(2)}ms`);
            console.log(`   Route: ${result.route_names.join(' -> ')}`);
            console.log(`   Steps: ${result.route_indices.length}, Battery: ${(result.details.energyUsed).toFixed(1)}%`);
            
            // Validate the route contains all required pickup/delivery nodes
            const requiredNodes = new Set();
            testCase.packages.forEach(pkg => {
                requiredNodes.add(pkg.pickup);
                requiredNodes.add(pkg.delivery);
            });
            
            const routeNodes = new Set(result.route_indices);
            let allNodesVisited = true;
            for (const node of requiredNodes) {
                if (!routeNodes.has(node)) {
                    allNodesVisited = false;
                    console.log(`❌ Missing required node: ${node}`);
                }
            }
            
            if (allNodesVisited) {
                console.log(`✅ All required nodes visited`);
            }
            
        } else {
            console.log(`❌ FAILED - ${result.error || 'Unknown error'}`);
        }
        
    } catch (error) {
        console.log(`❌ ERROR - ${error.message}`);
    }
}

console.log('\n[INTEGRATION TEST] Integration test completed!');
