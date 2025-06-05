// server.js
const express = require('express');
const path = require('path');
const app = express();
const fs = require('fs');

// Accept JSON request bodies with increased limit for graph data
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Servir les fichiers statiques depuis le dossier public
app.use(express.static(path.join(__dirname, 'public')));

// Servir les fichiers statiques depuis le dossier data
app.use('/data', express.static(path.join(__dirname, 'data')));

// Si tu veux forcer l'envoi de index.html sur /
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/index.html'));
});

// Route API pour simuler les endpoints utilisés dans map-generator.js
app.get('/api/strategic-points', (req, res) => {
  // Simuler la récupération des points depuis la logique JavaScript
  // En production, ceci récupérerait les vrais points du city-picker
  const mockNodes = [];
  
  // Générer des points de test (remplacez par la vraie logique)
  const nodeTypes = [
    { type: 'hubs', count: 20 },
    { type: 'charging', count: 50 },
    { type: 'delivery', count: 20 },
    { type: 'pickup', count: 20 }
  ];
  
  let nodeId = 0;
  nodeTypes.forEach(({ type, count }) => {
    for (let i = 0; i < count; i++) {
      mockNodes.push({
        id: `${type.charAt(0).toUpperCase() + type.slice(1)} ${i + 1}`,
        lat: 41.3 + Math.random() * (51.1 - 41.3),
        lng: -5.2 + Math.random() * (9.5 - (-5.2)),
        type: type,
        index: nodeId++
      });
    }
  });
  
  res.json({ 
    status: 'success', 
    nodes: mockNodes 
  });
});

app.get('/api/osrm-status', (req, res) => {
  res.json({ status: 'online' });
});

// New endpoint for algorithm execution with PPO integration
app.post('/api/run-algorithm', (req, res) => {
  const { algorithm, batteryCapacity, maxPayload, startNode, endNode } = req.body;
  
  console.log(`🚁 Executing ${algorithm} algorithm`);
  console.log(`📊 Parameters: battery=${batteryCapacity}, payload=${maxPayload}`);
  console.log(`🗺️ Route: ${startNode} -> ${endNode}`);
  
  if (algorithm === 'ppo' || algorithm === 'gnn') {
    // PPO algorithm execution with live inference
    const { spawn } = require('child_process');
    const path = require('path');
    
    const modelPath = path.join(__dirname, 'models', 'drone_ppo_full_random_20250605_163145_final.zip');
    const graphPath = path.join(__dirname, 'data', 'graph.json');
    
    console.log(`📂 Model path: ${modelPath}`);
    console.log(`📂 Graph path: ${graphPath}`);
    
    // Check if model exists
    const fs = require('fs');
    if (!fs.existsSync(modelPath)) {
      console.log('❌ PPO model not found');
      res.status(404).json({
        status: 'error',
        message: 'PPO model not found. Please train a model first.',
        route: [],
        route_names: [],
        stats: { success: false }
      });
      return;
    } else {
      console.log('✅ Model file exists');
    }

    // Check if graph exists
    if (!fs.existsSync(graphPath)) {
      console.log('❌ Graph data not found');
      res.status(404).json({
        status: 'error', 
        message: 'Graph data not found. Please generate network first.',
        route: [],
        route_names: [],
        stats: { success: false }
      });
      return;
    } else {
      console.log('✅ Graph file exists');
    }
    
    // Run live inference
    console.log('🚀 Starting inference process...');
    const inferenceProcess = spawn('python', [
      path.join(__dirname, 'python', 'live_inference.py'),
      '--model', modelPath,
      '--graph', graphPath,
      '--pickup', startNode || 'Pickup 1',
      '--delivery', endNode || 'Delivery 1', 
      '--battery', batteryCapacity.toString(),
      '--payload', maxPayload.toString()
      // Removed --quiet to see all logs
    ], {
      cwd: __dirname
    });
    
    let outputData = '';
    let errorData = '';
    
    inferenceProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log(`[Python STDOUT]: ${output}`);
      outputData += output;
    });
    
    inferenceProcess.stderr.on('data', (data) => {
      const error = data.toString();
      console.error(`[Python STDERR]: ${error}`);
      errorData += error;
    });
    
    inferenceProcess.on('close', (code) => {
      console.log(`🏁 Python process finished with code: ${code}`);
      console.log(`📊 Total output length: ${outputData.length} chars`);
      
      if (code === 0) {
        try {
          // Extract JSON from output (after "JSON OUTPUT:" line)
          const jsonStart = outputData.indexOf('JSON OUTPUT:');
          if (jsonStart !== -1) {
            const jsonStr = outputData.substring(jsonStart + 'JSON OUTPUT:'.length).trim();
            console.log(`📄 Extracted JSON (${jsonStr.length} chars): ${jsonStr.substring(0, 200)}...`);
            
            const result = JSON.parse(jsonStr);
            
            // Format for frontend
            const response = {
              status: result.status,
              message: result.message,
              route: result.route_names || [],
              route_indices: result.route || [],
              actions: result.actions || [],
              rewards: result.rewards || [],
              costs: result.costs || [],
              battery_history: result.battery_history || [],
              stats: {
                success: result.stats.success || false,
                distance: result.stats.total_cost || 0,
                batteryUsed: result.stats.battery_used || 0,
                steps: result.stats.total_steps || 0,
                recharges: result.stats.recharges || 0,
                termination_reason: result.stats.termination_reason || 'unknown'
              },
              graph_info: result.graph_info || {}
            };
            
            console.log(`✅ PPO inference completed: ${response.stats.success ? 'SUCCESS' : 'FAILED'}`);
            console.log(`📊 Route: ${response.route.join(' → ')}`);
            console.log(`📊 Steps: ${response.stats.steps}, Battery: ${response.stats.batteryUsed}%`);
            
            res.json(response);
          } else {
            throw new Error('No JSON output found in Python output');
          }
        } catch (e) {
          console.error('❌ Failed to parse PPO output:', e);
          console.error('Raw output:', outputData);
          res.status(500).json({
            status: 'error',
            message: `Failed to parse PPO output: ${e.message}`,
            route: [],
            stats: { success: false, distance: 0, batteryUsed: 0 },
            debug: {
              rawOutput: outputData.substring(0, 1000),
              error: e.message
            }
          });
        }
      } else {
        console.error(`❌ PPO inference failed with code ${code}`);
        console.error('Error output:', errorData);
        res.status(500).json({
          status: 'error',
          message: `PPO inference failed with code ${code}: ${errorData}`,
          route: [],
          stats: { success: false, distance: 0, batteryUsed: 0 },
          debug: {
            exitCode: code,
            stderr: errorData,
            stdout: outputData
          }
        });
      }
    });
    
    // Set timeout
    setTimeout(() => {
      console.log('⏰ PPO inference timeout - killing process');
      inferenceProcess.kill();
      res.status(408).json({
        status: 'timeout',
        message: 'PPO inference timed out after 30 seconds',
        route: [],
        stats: { success: false, distance: 0, batteryUsed: 0 }
      });
    }, 30000); // 30 second timeout
    
  } else {
    // Genetic Algorithm placeholder
    setTimeout(() => {
      res.json({ 
        status: 'success', 
        message: `${algorithm} algorithm executed successfully`,
        route: ['Hub 1', 'Pickup 3', 'Delivery 5', 'Hub 1'],
        stats: {
          success: true,
          distance: 150.5,
          batteryUsed: batteryCapacity * 0.8,
          steps: 12
        }
      });
    }, 1500);
  }
});

// New endpoint to save graph data for PPO training
app.post('/api/save-graph', (req, res) => {
  const path = require('path');
  
  try {
    const graphData = req.body;
    const dataDir = path.join(__dirname, 'data');
    
    // Ensure data directory exists
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
    
    // Save graph to JSON file
    const filePath = path.join(dataDir, 'graph.json');
    fs.writeFileSync(filePath, JSON.stringify(graphData, null, 2));
    
    console.log(`Graph saved: ${graphData.nodes.length} nodes, ${graphData.edges.length} edges`);
    res.json({ status: 'success', message: 'Graph saved successfully' });
  } catch (error) {
    console.error('Error saving graph:', error);
    res.status(500).json({ status: 'error', message: 'Failed to save graph' });
  }
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`🚀 Serveur démarré sur http://localhost:${PORT}`);
});


