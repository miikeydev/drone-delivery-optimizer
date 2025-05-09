const express = require('express');
const path = require('path');
const { strategicPoints } = require('./src/data/strategic-points');
const osrmService = require('./src/services/osrm-service');

const app = express();
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/strategic-points', (_, res) => {
  res.json(strategicPoints);
});

// Routing endpoint using OSRM public API
app.get('/api/route', async (req, res) => {
  const { from, to } = req.query;
  
  if (!from || !to) {
    return res.status(400).json({ error: 'Missing required parameters: from and to' });
  }
  
  try {
    const route = await osrmService.getRoute(from, to);
    
    // Calculate weather cost factor
    const alpha = 0.1, beta = 0.05;
    const v = 0;  // wind speed in m/s (will be connected to weather API later)
    const r = 0;  // rain in mm/h (will be connected to weather API later)
    const cost = route.distance * (1 + alpha * v + beta * r);
    
    res.json({ ...route, cost });
  } catch (error) {
    res.status(500).json({ 
      error: error.message,
      details: 'OSRM routing error'
    });
  }
});

// Status endpoint to check if OSRM is available
app.get('/api/status', async (_, res) => {
  try {
    // Test with a simple route request
    await osrmService.getRoute('2.3522,48.8566', '5.3698,43.2965');
    res.json({ osrm: 'running' });
  } catch (error) {
    res.json({ osrm: 'not available', error: error.message });
  }
});

app.listen(3000, () => console.log('Server running on http://localhost:3000'));
