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
  
  console.log(`Executing ${algorithm} algorithm with battery=${batteryCapacity}, payload=${maxPayload}`);
  console.log(`Route: ${startNode} -> ${endNode}`);
  
  if (algorithm === 'ppo') {
    // PPO algorithm execution
    const { spawn } = require('child_process');
    const path = require('path');
    
    const modelPath = path.join(__dirname, 'python', 'models', 'drone_ppo.zip');
    const graphPath = path.join(__dirname, 'data', 'graph.json');
    
    // Check if model exists, if not, start training
    const fs = require('fs');
    if (!fs.existsSync(modelPath)) {
      console.log('PPO model not found, starting training...');
      
      // Start training process
      const trainProcess = spawn('python', [
        path.join(__dirname, 'python', 'train_simpleppo.py'),
        '--graph', graphPath,
        '--timesteps', '1000000',  // Reduced for faster training
        '--save-path', path.join(__dirname, 'python', 'models', 'drone_ppo')
      ], {
        cwd: __dirname
      });
      
      trainProcess.stdout.on('data', (data) => {
        console.log(`Training: ${data}`);
      });
      
      trainProcess.stderr.on('data', (data) => {
        console.error(`Training error: ${data}`);
      });
      
      trainProcess.on('close', (code) => {
        console.log(`Training finished with code ${code}`);
      });
      
      // Return training status
      res.json({
        status: 'training',
        message: 'PPO model training started. This may take several minutes.',
        route: [],
        stats: {
          distance: 0,
          time: 0,
          batteryUsed: 0
        }
      });
      
      return;
    }
    
    // Model exists, run evaluation
    const evalProcess = spawn('python', [
      path.join(__dirname, 'python', 'eval_simpleppo.py'),
      '--graph', graphPath,
      '--model', modelPath,
      '--start', startNode || 'Hub 1',
      '--end', endNode || 'Delivery 1'
    ], {
      cwd: __dirname
    });
    
    let outputData = '';
    let errorData = '';
    
    evalProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });
    
    evalProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`PPO eval error: ${data}`);
    });
    
    evalProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(outputData);
          res.json(result);
        } catch (e) {
          console.error('Failed to parse PPO output:', e);
          res.status(500).json({
            status: 'error',
            message: 'Failed to parse PPO output',
            route: [],
            stats: { distance: 0, time: 0, batteryUsed: 0 }
          });
        }
      } else {
        console.error(`PPO evaluation failed with code ${code}`);
        res.status(500).json({
          status: 'error',
          message: `PPO evaluation failed: ${errorData}`,
          route: [],
          stats: { distance: 0, time: 0, batteryUsed: 0 }
        });
      }
    });
    
    // Set timeout
    setTimeout(() => {
      evalProcess.kill();
      res.status(408).json({
        status: 'timeout',
        message: 'PPO evaluation timed out',
        route: [],
        stats: { distance: 0, time: 0, batteryUsed: 0 }
      });
    }, 30000); // 30 second timeout
    
  } else {
    // Genetic Algorithm placeholder
    setTimeout(() => {
      res.json({ 
        status: 'success', 
        message: `${algorithm} algorithm executed successfully`,
        route: [],
        stats: {
          distance: 150.5,
          time: 45.2,
          batteryUsed: batteryCapacity * 0.8
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


