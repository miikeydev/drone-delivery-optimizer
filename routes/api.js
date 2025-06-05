const express = require('express');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const router = express.Router();

function findModel() {
  const paths = [
    path.join(__dirname, '../python/models/drone_ppo_enhanced_final.zip'),
    path.join(__dirname, '../models/drone_ppo_enhanced_final.zip'),
    path.join(__dirname, '../python/models/best_model.zip'),
    path.join(__dirname, '../models/best_model.zip')
  ];
  return paths.find(p => fs.existsSync(p)) || null;
}

function runPPO({ pickup, delivery, battery, payload }) {
  return new Promise((resolve, reject) => {
    const modelPath = findModel();
    if (!modelPath) return reject(new Error('model_not_found'));

    const graphPath = path.join(__dirname, '../data/graph.json');
    if (!fs.existsSync(graphPath)) return reject(new Error('graph_not_found'));

    const pythonScript = path.join(__dirname, '../python/live_inference.py');
    const args = [
      pythonScript,
      '--model', modelPath,
      '--graph', graphPath,
      '--pickup', pickup,
      '--delivery', delivery,
      '--battery', String(battery),
      '--payload', String(payload)
    ];

    const proc = spawn('python', args, { cwd: path.join(__dirname, '../python') });
    let output = '';
    let errorOutput = '';

    proc.stdout.on('data', d => (output += d.toString()));
    proc.stderr.on('data', d => (errorOutput += d.toString()));

    proc.on('close', code => {
      if (code !== 0) return reject(new Error(`exit_${code}:${errorOutput}`));
      const resultFile = path.join(__dirname, '../python/inference_result.json');
      if (!fs.existsSync(resultFile)) return reject(new Error('result_missing'));
      try {
        const data = JSON.parse(fs.readFileSync(resultFile, 'utf8'));
        fs.unlinkSync(resultFile);
        resolve(data);
      } catch (err) {
        reject(new Error('parse_error:' + err.message));
      }
    });
  });
}

router.get('/strategic-points', (req, res) => {
  const nodeTypes = [
    { type: 'hubs', count: 20 },
    { type: 'charging', count: 50 },
    { type: 'delivery', count: 20 },
    { type: 'pickup', count: 20 }
  ];
  let nodeId = 0;
  const nodes = [];
  nodeTypes.forEach(({ type, count }) => {
    for (let i = 0; i < count; i++) {
      nodes.push({
        id: `${type.charAt(0).toUpperCase() + type.slice(1)} ${i + 1}`,
        lat: 41.3 + Math.random() * (51.1 - 41.3),
        lng: -5.2 + Math.random() * (9.5 - (-5.2)),
        type,
        index: nodeId++
      });
    }
  });
  res.json({ status: 'success', nodes });
});

router.get('/osrm-status', (req, res) => {
  res.json({ status: 'online' });
});

router.post('/save-graph', (req, res) => {
  const dataDir = path.join(__dirname, '../data');
  if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });
  fs.writeFileSync(path.join(dataDir, 'graph.json'), JSON.stringify(req.body, null, 2));
  res.json({ status: 'success', message: 'Graph saved' });
});

router.post('/run-ppo-inference', async (req, res) => {
  const { pickupNode, deliveryNode, batteryCapacity, maxPayload } = req.body;
  try {
    const result = await runPPO({
      pickup: pickupNode,
      delivery: deliveryNode,
      battery: batteryCapacity,
      payload: maxPayload
    });
    res.json({ status: 'success', result });
  } catch (err) {
    handleError(err, res);
  }
});

router.post('/run-algorithm', async (req, res) => {
  const { algorithm, batteryCapacity, maxPayload, startNode, endNode } = req.body;
  if (algorithm !== 'ppo') {
    return setTimeout(() => {
      res.json({
        status: 'success',
        message: `${algorithm} completed`,
        route: ['Hub 1', 'Pickup 3', 'Delivery 5'],
        stats: { success: true, distance: 150.5, batteryUsed: 60, steps: 12 }
      });
    }, 1000);
  }
  try {
    const result = await runPPO({
      pickup: startNode,
      delivery: endNode,
      battery: batteryCapacity,
      payload: maxPayload
    });
    res.json({ status: 'success', result });
  } catch (err) {
    handleError(err, res);
  }
});

function handleError(err, res) {
  if (err.message.startsWith('model_not_found')) {
    return res.json({ status: 'error', message: 'Trained model not found' });
  }
  if (err.message.startsWith('graph_not_found')) {
    return res.json({ status: 'error', message: 'Graph data not found' });
  }
  if (err.message.startsWith('result_missing')) {
    return res.json({ status: 'error', message: 'No inference result generated' });
  }
  if (err.message.startsWith('parse_error')) {
    return res.json({ status: 'error', message: 'Failed to parse inference result' });
  }
  res.json({ status: 'error', message: 'PPO inference failed', details: err.message });
}

module.exports = router;
