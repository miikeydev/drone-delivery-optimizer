// server.js
const express = require('express');
const path = require('path');
const app = express();

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

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`🚀 Serveur démarré sur http://localhost:${PORT}`);
});
