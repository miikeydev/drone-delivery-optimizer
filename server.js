// server.js
const express = require('express');
const path = require('path');
const app = express();

// Accept JSON request bodies
app.use(express.json());

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
  res.json({ status: 'success', points: [] });
});

app.get('/api/osrm-status', (req, res) => {
  res.json({ status: 'online' });
});

// New endpoint for algorithm execution - placeholder for future implementation
app.post('/api/run-algorithm', (req, res) => {
  const { algorithm, batteryCapacity, maxPayload, startNode, endNode } = req.body;
  
  console.log(`Executing ${algorithm} algorithm with battery=${batteryCapacity}, payload=${maxPayload}`);
  
  // This is a placeholder - the real implementation would come later
  // Simulate processing delay
  setTimeout(() => {
    res.json({ 
      status: 'success', 
      message: `${algorithm} algorithm executed successfully`,
      route: [],  // This would contain the actual route in a real implementation
      stats: {
        distance: 150.5,
        time: 45.2,
        batteryUsed: batteryCapacity * 0.8
      }
    });
  }, 1500);
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`🚀 Serveur démarré sur http://localhost:${PORT}`);
});


