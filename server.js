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
  
  console.log(`🚁 Running ${algorithm}: ${startNode} -> ${endNode}`);
  
  if (algorithm === 'ppo') {
    const { spawn } = require('child_process');
    // FIXED: Try multiple possible model paths
    const possibleModelPaths = [
      path.join(__dirname, 'python', 'models', 'drone_ppo_enhanced_final.zip'),
      path.join(__dirname, 'models', 'drone_ppo_enhanced_final.zip'),
      path.join(__dirname, 'python', 'models', 'best_model.zip'),
      path.join(__dirname, 'models', 'best_model.zip')
    ];
    
    let modelPath = null;
    for (const testPath of possibleModelPaths) {
      if (fs.existsSync(testPath)) {
        modelPath = testPath;
        break;
      }
    }
    
    const graphPath = path.join(__dirname, 'data', 'graph.json');
    
    // Better validation with helpful message
    if (!modelPath) {
      return res.json({
        status: 'error',
        message: 'PPO model not found. Please train a model first using: python train.py --graph data/graph.json',
        searched_paths: possibleModelPaths,
        help: 'Run training first: cd python && python train.py --graph ../data/graph.json --timesteps 100000'
      });
    }
    
    if (!fs.existsSync(graphPath)) {
      return res.json({
        status: 'error', 
        message: 'Graph data not found. Generate network first by clicking on the map.',
        graph_path: graphPath
      });
    }
    
    console.log(`Using model: ${modelPath}`);
    
    // FIXED: Use spawn with array arguments to handle spaces properly
    const inferenceProcess = spawn('python', [
      path.join(__dirname, 'python', 'live_inference.py'),
      '--model', modelPath,
      '--graph', graphPath,
      '--pickup', startNode,  // No quotes needed with spawn array
      '--delivery', endNode,  // No quotes needed with spawn array
      '--battery', batteryCapacity.toString(),
      '--payload', maxPayload.toString()
    ], {
      cwd: path.join(__dirname, 'python')  // Set working directory
    });
    
    let output = '';
    let errorOutput = '';
    let responseSent = false;
    
    // Timeout protection
    const timeout = setTimeout(() => {
      if (!responseSent) {
        responseSent = true;
        inferenceProcess.kill();
        res.json({
          status: 'error',
          message: 'PPO inference timeout (30s)',
          help: 'The model might be too large or the environment setup is incomplete'
        });
      }
    }, 30000);
    
    inferenceProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    inferenceProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });
    
    inferenceProcess.on('close', (code) => {
      clearTimeout(timeout);
      if (responseSent) return;
      responseSent = true;
      
      if (code === 0) {
        try {
          // Find JSON in output
          const jsonMatch = output.match(/\{[\s\S]*\}/);
          if (!jsonMatch) {
            throw new Error('No JSON found in output');
          }
          
          const result = JSON.parse(jsonMatch[0]);
          
          // Format for frontend compatibility
          res.json({
            status: 'success',
            message: 'PPO inference completed',
            stats: {
              success: result.success,
              steps: result.steps,
              batteryUsed: result.battery_used,
              termination_reason: result.termination_reason
            },
            route_indices: result.route_indices,
            route_names: result.route_names,
            battery_history: result.battery_history
          });
        } catch (e) {
          res.json({
            status: 'error',
            message: 'Failed to parse PPO inference result',
            error: e.message,
            stdout: output,
            stderr: errorOutput
          });
        }
      } else {
        res.json({
          status: 'error',
          message: 'PPO inference failed',
          exit_code: code,
          stdout: output,
          stderr: errorOutput,
          help: 'Check if Python environment has stable-baselines3 installed'
        });
      }
    });
    
  } else {
    // Genetic Algorithm placeholder
    setTimeout(() => {
      res.json({ 
        status: 'success', 
        message: `${algorithm} completed`,
        route: ['Hub 1', 'Pickup 3', 'Delivery 5'],
        stats: { success: true, distance: 150.5, batteryUsed: 60, steps: 12 }
      });
    }, 1000);
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

// Add this new endpoint after the existing /api/run-algorithm endpoint
app.post('/api/run-ppo-inference', async (req, res) => {
  try {
    const { pickupNode, deliveryNode, batteryCapacity, maxPayload } = req.body;
    
    console.log('🚁 Running PPO inference:', { pickupNode, deliveryNode, batteryCapacity, maxPayload });
    
    // FIXED: Try multiple possible model paths
    const possibleModelPaths = [
      path.join(__dirname, 'python', 'models', 'drone_ppo_enhanced_final.zip'),
      path.join(__dirname, 'models', 'drone_ppo_enhanced_final.zip'),
      path.join(__dirname, 'python', 'models', 'best_model.zip'),
      path.join(__dirname, 'models', 'best_model.zip')
    ];
    
    let modelPath = null;
    for (const testPath of possibleModelPaths) {
      if (fs.existsSync(testPath)) {
        modelPath = testPath;
        break;
      }
    }
    
    const graphPath = path.join(__dirname, 'data', 'graph.json');
    
    // Check if model exists
    if (!modelPath) {
      return res.json({
        status: 'error',
        message: 'Trained model not found. Please train a model first.',
        help: 'Run: cd python && python train.py --graph ../data/graph.json --timesteps 100000',
        searched_paths: possibleModelPaths
      });
    }
    
    // Check if graph exists
    if (!fs.existsSync(graphPath)) {
      return res.json({
        status: 'error',
        message: 'Graph data not found. Please generate network first.'
      });
    }
    
    console.log(`Using model: ${modelPath}`);
    
    // FIXED: Use spawn instead of exec to properly handle spaces in arguments
    const { spawn } = require('child_process');
    const pythonScript = path.join(__dirname, 'python', 'live_inference.py');
    
    const args = [
      pythonScript,
      '--model', modelPath,
      '--graph', graphPath,
      '--pickup', pickupNode,   // Properly handled by spawn
      '--delivery', deliveryNode, // Properly handled by spawn
      '--battery', batteryCapacity.toString(),
      '--payload', maxPayload.toString()
    ];
    
    console.log('Executing:', 'python', args.join(' '));
    
    const inferenceProcess = spawn('python', args, {
      cwd: path.join(__dirname, 'python')
    });
    
    let output = '';
    let errorOutput = '';
    let responseSent = false;
    
    // Timeout protection
    const timeout = setTimeout(() => {
      if (!responseSent) {
        responseSent = true;
        inferenceProcess.kill();
        res.json({
          status: 'error',
          message: 'PPO inference timeout (30s)',
          help: 'The model might be too large or the environment setup is incomplete'
        });
      }
    }, 30000);
    
    inferenceProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    inferenceProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });
    
    inferenceProcess.on('close', (code) => {
      clearTimeout(timeout);
      if (responseSent) return;
      responseSent = true;
      
      console.log(`PPO inference process exited with code: ${code}`);
      console.log('STDOUT:', output);
      if (errorOutput) {
        console.log('STDERR:', errorOutput);
      }
      
      if (code === 0) {
        // Try to read the result file
        const resultPath = path.join(__dirname, 'python', 'inference_result.json');
        
        if (fs.existsSync(resultPath)) {
          try {
            const resultData = JSON.parse(fs.readFileSync(resultPath, 'utf8'));
            
            res.json({
              status: 'success',
              message: 'PPO inference completed',
              result: resultData,
              // Format for frontend compatibility
              stats: {
                success: resultData.success,
                steps: resultData.steps,
                batteryUsed: resultData.battery_used,
                termination_reason: resultData.termination_reason
              },
              route_indices: resultData.route_indices,
              route_names: resultData.route_names,
              battery_history: resultData.battery_history,
              actions: resultData.actions,
              action_types: resultData.action_types
            });
            
            // Clean up result file
            fs.unlinkSync(resultPath);
            
          } catch (parseError) {
            console.error('Error parsing inference result:', parseError);
            res.json({
              status: 'error',
              message: 'Failed to parse inference result',
              error: parseError.message
            });
          }
        } else {
          res.json({
            status: 'error',
            message: 'No inference result file generated',
            stdout: output,
            stderr: errorOutput
          });
        }
      } else {
        res.json({
          status: 'error',
          message: 'PPO inference failed',
          exit_code: code,
          stdout: output,
          stderr: errorOutput,
          help: 'Check if Python environment has stable-baselines3 installed'
        });
      }
    });
    
    inferenceProcess.on('error', (error) => {
      clearTimeout(timeout);
      if (!responseSent) {
        responseSent = true;
        console.error('PPO inference spawn error:', error);
        res.json({
          status: 'error',
          message: 'Failed to start PPO inference process',
          error: error.message,
          help: 'Make sure Python is installed and in PATH'
        });
      }
    });
    
  } catch (error) {
    console.error('PPO inference endpoint error:', error);
    res.json({
      status: 'error',
      message: 'Internal server error',
      error: error.message
    });
  }
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`🚀 Serveur démarré sur http://localhost:${PORT}`);
});


