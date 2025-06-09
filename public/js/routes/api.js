import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import ppoService from '../services/ppoService.js';
import config from '../config/index.js';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = express.Router();

router.get('/strategic-points', (req, res) => {
  const mockNodes = [];
  
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

router.get('/osrm-status', (req, res) => {
  res.json({ status: 'online' });
});

router.post('/run-algorithm', async (req, res) => {
  const { algorithm, batteryCapacity, maxPayload, startNode, endNode } = req.body;
  
  console.log(`Running ${algorithm}: ${startNode} -> ${endNode}`);
  
  try {
    const result = await ppoService.runLegacyInference({
      algorithm,
      batteryCapacity,
      maxPayload,
      startNode,
      endNode
    });
    
    res.json(result);
  } catch (error) {
    console.error('Algorithm execution error:', error);
    res.json({
      status: 'error',
      message: 'Algorithm execution failed',
      error: error.message
    });
  }
});

router.post('/run-ppo-inference', async (req, res) => {
  const { pickupNode, deliveryNode, batteryCapacity, maxPayload } = req.body;
  
  console.log('Running PPO inference:', { pickupNode, deliveryNode, batteryCapacity, maxPayload });
  
  try {
    const result = await ppoService.runInference({
      pickupNode,
      deliveryNode,
      batteryCapacity,
      maxPayload
    });
    
    res.json(result);
  } catch (error) {
    console.error('PPO inference endpoint error:', error);
    res.json({
      status: 'error',
      message: 'Internal server error',
      error: error.message
    });
  }
});

router.post('/save-graph', (req, res) => {
  try {
    const graphData = req.body;
    const dataDir = config.paths.data;
    
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
    
    const filePath = path.join(dataDir, 'graph.json');
    fs.writeFileSync(filePath, JSON.stringify(graphData, null, 2));
    
    console.log(`Graph saved: ${graphData.nodes.length} nodes, ${graphData.edges.length} edges`);
    res.json({ status: 'success', message: 'Graph saved successfully' });
  } catch (error) {
    console.error('Error saving graph:', error);
    res.status(500).json({ status: 'error', message: 'Failed to save graph' });
  }
});

export default router;
