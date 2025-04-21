const axios = require('axios');
const osmtogeojson = require('osmtogeojson');

/**
 * Récupère les données OSM pour une bbox au format "minLat,minLon,maxLat,maxLon"
 */
async function fetchOSM(bbox) {
  const query = `
    [out:json][timeout:25];
    (
      node["highway"](${bbox});
      way["highway"](${bbox});
    );
    out body;
    >;
    out skel qt;`;
  const resp = await axios.post(
    'https://overpass-api.de/api/interpreter',
    query,
    { headers: { 'Content-Type': 'text/plain' } }
  );
  return resp.data;
}

/** Convertit le JSON OSM en GeoJSON */
function toGeoJSON(osmData) {
  return osmtogeojson(osmData);
}

module.exports = { fetchOSM, toGeoJSON };
