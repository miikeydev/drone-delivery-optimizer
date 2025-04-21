const express = require('express');
const app = express();
const { fetchOSM, toGeoJSON } = require('./ingestion/osm');

console.log('Happy developing ✨');

// Tester la récupération OSM
// Ex. : http://localhost:3000/api/osm?bbox=48.80,2.25,48.90,2.40
app.get('/api/osm', async (req, res, next) => {
  try {
    const bbox = req.query.bbox;
    const raw = await fetchOSM(bbox);
    const geojson = toGeoJSON(raw);
    res.json(geojson);
  } catch (err) {
    next(err);
  }
});

app.listen(3000, () => {
  console.log('Server running on http://localhost:3000');
});
