const express = require('express');
const path = require('path');
const { strategicPoints } = require('./src/data/strategic-points');
const osrmService = require('./src/services/osrm-service');
const matrixService = require('./src/services/matrix-service');

const app = express();
app.use(express.static(path.join(__dirname, 'public')));

// Créer le dossier data s'il n'existe pas
const fs = require('fs');
try {
  if (!fs.existsSync('./data')) {
    fs.mkdirSync('./data');
    console.log('Data directory created');
  }
} catch (err) {
  console.error('Error creating data directory:', err);
}

// Initialiser la matrice de distances au démarrage
(async function initializeMatrix() {
  try {
    // Essayer d'abord de charger depuis le fichier (format auto-détecté)
    const loaded = await matrixService.loadMatrixFromFile('auto');
    
    // Si le chargement a échoué ou si la matrice est vide, la calculer
    if (!loaded || !matrixService.distanceMatrix.initialized) {
      console.log('Calculating distance matrix from scratch...');
      await matrixService.calculateMatrix(strategicPoints);
      
      // Sauvegarder au format compact par défaut
      await matrixService.saveMatrixToFile('compact');
    }
    
    console.log('Distance matrix initialized successfully');
  } catch (error) {
    console.error('Error initializing distance matrix:', error);
  }
})();

// Endpoint pour sauvegarder manuellement la matrice
app.get('/api/matrix/save', async (req, res) => {
  try {
    const format = req.query.format || 'compact';
    const success = await matrixService.saveMatrixToFile(format);
    
    if (success) {
      res.json({ 
        status: 'success', 
        message: `Matrix saved successfully in ${format} format`,
        stats: matrixService.distanceMatrix.getStats()
      });
    } else {
      res.status(500).json({ status: 'error', message: 'Failed to save matrix' });
    }
  } catch (error) {
    res.status(500).json({ status: 'error', message: error.message });
  }
});

// Endpoint pour recharger la matrice depuis le fichier
app.get('/api/matrix/load', async (req, res) => {
  try {
    const format = req.query.format || 'auto';
    const success = await matrixService.loadMatrixFromFile(format);
    
    if (success) {
      res.json({ 
        status: 'success', 
        message: `Matrix loaded successfully from ${matrixService.distanceMatrix.format} format`,
        stats: matrixService.distanceMatrix.getStats()
      });
    } else {
      res.status(500).json({ status: 'error', message: 'Failed to load matrix' });
    }
  } catch (error) {
    res.status(500).json({ status: 'error', message: error.message });
  }
});

// Endpoint pour convertir le format de la matrice
app.get('/api/matrix/convert', async (req, res) => {
  try {
    const targetFormat = req.query.format || 'compact';
    
    if (!['json', 'json-gz', 'compact'].includes(targetFormat)) {
      return res.status(400).json({ 
        status: 'error', 
        message: `Format non supporté: ${targetFormat}. Utilisez 'json', 'json-gz' ou 'compact'`
      });
    }
    
    const success = await matrixService.convertMatrixFormat(targetFormat);
    
    if (success) {
      res.json({ 
        status: 'success', 
        message: `Matrix converted to ${targetFormat} format`,
        stats: matrixService.distanceMatrix.getStats()
      });
    } else {
      res.status(500).json({ status: 'error', message: 'Failed to convert matrix' });
    }
  } catch (error) {
    res.status(500).json({ status: 'error', message: error.message });
  }
});

// Endpoint pour recalculer entièrement la matrice
app.get('/api/matrix/recalculate', async (req, res) => {
  try {
    await matrixService.calculateMatrix(strategicPoints);
    
    // Par défaut, sauvegarder au format compact
    const format = req.query.format || 'compact';
    await matrixService.saveMatrixToFile(format);
    
    res.json({ 
      status: 'success', 
      message: `Matrix recalculated successfully and saved in ${format} format`,
      stats: matrixService.distanceMatrix.getStats()
    });
  } catch (error) {
    res.status(500).json({ status: 'error', message: error.message });
  }
});

app.get('/api/strategic-points', (_, res) => {
  res.json(strategicPoints);
});

app.get('/api/matrix-stats', (_, res) => {
  res.json(matrixService.distanceMatrix.getStats());
});

// Routing endpoint using OSRM public API
app.get('/api/route', async (req, res) => {
  const { from, to } = req.query;
  
  if (!from || !to) {
    return res.status(400).json({ error: 'Missing required parameters: from and to' });
  }
  
  try {
    // Chercher les points stratégiques correspondants
    let fromPoint = null;
    let toPoint = null;
    
    // Format from/to peut être soit des coordonnées, soit des IDs
    if (from.includes(',') && to.includes(',')) {
      // Ce sont des coordonnées, on cherche les points les plus proches
      const fromCoords = from.split(',').map(parseFloat);
      const toCoords = to.split(',').map(parseFloat);
      
      // Recherche les points par coordonnées
      for (const feature of strategicPoints.features) {
        const coords = feature.geometry.coordinates;
        if (coords[0] === fromCoords[0] && coords[1] === fromCoords[1]) {
          fromPoint = feature;
        }
        if (coords[0] === toCoords[0] && coords[1] === toCoords[1]) {
          toPoint = feature;
        }
      }
    } else {
      // Ce sont des IDs
      fromPoint = strategicPoints.features.find(f => f.properties.id === from);
      toPoint = strategicPoints.features.find(f => f.properties.id === to);
    }
    
    if (!fromPoint || !toPoint) {
      // Si points non trouvés, faire un calcul direct
      const route = await osrmService.getRoute(from, to);
      const cost = calculateCost(route, from, to);
      
      // Normaliser le coût par rapport à la matrice
      const { minCost, maxCost } = matrixService.distanceMatrix.getStats();
      const normalizedScore = (cost - minCost) / (maxCost - minCost);
      
      return res.json({ 
        ...route, 
        cost,
        normalizedScore: isNaN(normalizedScore) ? 0.5 : normalizedScore,
        fromMatrix: false
      });
    }
    
    // Vérifier si la route existe dans la matrice
    const matrixEntry = matrixService.distanceMatrix.getEntry(
      fromPoint.properties.id, 
      toPoint.properties.id
    );
    
    if (matrixEntry) {
      // Normaliser le coût
      const normalizedScore = matrixService.distanceMatrix.getNormalizedScore(matrixEntry.cost);
      
      return res.json({
        ...matrixEntry,
        normalizedScore,
        fromMatrix: true
      });
    }
    
    // Si non trouvé dans la matrice, calculer à la demande
    const fromCoords = `${fromPoint.geometry.coordinates[0]},${fromPoint.geometry.coordinates[1]}`;
    const toCoords = `${toPoint.geometry.coordinates[0]},${toPoint.geometry.coordinates[1]}`;
    
    const route = await osrmService.getRoute(fromCoords, toCoords);
    const cost = calculateCost(route, fromCoords, toCoords);
    
    // Ajouter à la matrice
    matrixService.distanceMatrix.addEntry(
      fromPoint.properties.id,
      toPoint.properties.id,
      { 
        ...route, 
        cost,
        fromId: fromPoint.properties.id,
        toId: toPoint.properties.id,
        fromName: fromPoint.properties.name,
        toName: toPoint.properties.name
      }
    );
    
    // Normaliser le coût
    const normalizedScore = matrixService.distanceMatrix.getNormalizedScore(cost);
    
    res.json({ 
      ...route, 
      cost, 
      normalizedScore,
      fromMatrix: false,
      fromId: fromPoint.properties.id,
      toId: toPoint.properties.id,
      fromName: fromPoint.properties.name,
      toName: toPoint.properties.name
    });
    
  } catch (error) {
    res.status(500).json({ 
      error: error.message,
      details: 'OSRM routing error'
    });
  }
});

// Helper function to calculate cost with regional factors
function calculateCost(route, from, to) {
  // Extract coordinates
  let fromLon, fromLat, toLon, toLat;
  
  if (typeof from === 'string' && from.includes(',')) {
    [fromLon, fromLat] = from.split(',').map(parseFloat);
  } else if (Array.isArray(from)) {
    [fromLon, fromLat] = from;
  }
  
  if (typeof to === 'string' && to.includes(',')) {
    [toLon, toLat] = to.split(',').map(parseFloat);
  } else if (Array.isArray(to)) {
    [toLon, toLat] = to;
  }
  
  // Default if we can't parse
  if (!fromLon || !fromLat || !toLon || !toLat) {
    return route.distance;
  }
  
  // Simulate different cost factors based on the regions
  let regionFactor = 1.0;
  
  // North region (above 48°N) - higher cost factor
  if (fromLat > 48 || toLat > 48) {
    regionFactor = 1.5;
  }
  
  // South region (below 44°N) - medium-high cost factor
  else if (fromLat < 44 || toLat < 44) {
    regionFactor = 1.2;
  }
  
  // East region (above 5°E) - medium cost factor
  else if (fromLon > 5 || toLon > 5) {
    regionFactor = 1.1;
  }
  
  // West region (below 0°E) - lowest cost factor
  else if (fromLon < 0 || toLon < 0) {
    regionFactor = 0.9;
  }
  
  // Base cost depends on the distance
  const cost = route.distance * regionFactor;
  
  return cost;
}

// Status endpoint to check if OSRM is available
app.get('/api/status', async (_, res) => {
  try {
    // Test avec une requête de route simple
    await osrmService.getRoute('2.3522,48.8566', '5.3698,43.2965');
    res.json({ 
      osrm: 'running',
      matrix: matrixService.distanceMatrix.getStats()
    });
  } catch (error) {
    res.json({ 
      osrm: 'not available', 
      error: error.message,
      matrix: matrixService.distanceMatrix.getStats()
    });
  }
});

app.listen(3000, () => console.log('Server running on http://localhost:3000'));
