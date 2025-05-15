/**
 * Utilities for the drone delivery optimizer
 */
import seedrandom from 'https://cdn.skypack.dev/seedrandom@3.0.5';

// Change la graine ici pour avoir une carte reproductible
export const RNG = seedrandom('drone-project-42');

/**
 * Calculate haversine distance between two points in km
 */
export function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth radius in kilometers
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = 
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
}

/**
 * Generates a random number from a Gaussian distribution
 */
export function gaussianRandom(mean = 0, stdev = 1) {
  const u = 1 - RNG(); // Converting [0,1) to (0,1] using RNG instead of Math.random()
  const v = RNG();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stdev + mean;
}

/**
 * Convert RGB to Hex color code
 */
export function rgbToHex(r, g, b) {
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

/**
 * Helper to check if a point is inside metropolitan France.
 * Requires window.turf to be loaded (see index.html).
 */
let _francePoly = null;
export function setFrancePolygon(geojson) {
  _francePoly = geojson;
}
export function insideFrance(lat, lng) {
  if (!_francePoly) {
    throw new Error("France polygon not loaded. Call setFrancePolygon(geojson) first.");
  }
  if (!window.turf) {
    throw new Error("Turf.js is not loaded. Make sure to import it in your HTML.");
  }
  return window.turf.booleanPointInPolygon(
    window.turf.point([lng, lat]),
    _francePoly
  );
}
