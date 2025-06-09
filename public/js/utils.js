class SeededRandom {
  constructor(seed) {
    this.seed = seed % 2147483647;
    if (this.seed <= 0) this.seed += 2147483646;
  }
  
  next() {
    this.seed = (this.seed * 16807) % 2147483647;
    return (this.seed - 1) / 2147483646;
  }
}

const seededRNG = new SeededRandom(42);
export const RNG = () => seededRNG.next();

export function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = 
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
}

export function gaussianRandom(mean = 0, stdev = 1) {
  const u = 1 - RNG();
  const v = RNG();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stdev + mean;
}

export function rgbToHex(r, g, b) {
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

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
