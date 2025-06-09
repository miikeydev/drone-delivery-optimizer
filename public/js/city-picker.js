import { RNG, haversineDistance, insideFrance, setFrancePolygon } from './utils.js';

export const CONFIG = {
  DEFAULT_PARAMS: { hubCount: 20, deliveryCount: 20, pickupCount: 20, chargingCount: 50 },
  MIN_INTER_ROLE_KM: 3,
  MIN_DIST_HUB_CHARGING: 0.02,
  RELAY_CHARGING_EVERY_KM: 80,
  MIN_CITY_DISTANCE_KM: 120,
  RADIAL_PER_CITY: 6,
  RADIAL_R_MIN_KM: 5,
  RADIAL_R_MAX_KM: 10,
  BIG_CITY_THRESHOLD: 200000
};

function shuffle(array) {
  const arr = [...array];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(RNG() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function complete(list, target, pool, minDist = 0) {
  const shuffled = shuffle(pool);
  for (const c of shuffled) {
    if (list.length >= target) break;
    if (!insideFrance(c.lat, c.lng)) continue;
    if (alreadyTaken(c.lat, c.lng)) continue;
    const ok = list.every(p => 
      haversineDistance(p[0], p[1], c.lat, c.lng) > minDist);
    if (ok) pushSafe(list, c.lat, c.lng);
  }
}

const taken = [];
function alreadyTaken(lat, lng) {
  return taken.some(p => haversineDistance(lat, lng, p[0], p[1]) < CONFIG.MIN_INTER_ROLE_KM);
}
function pushSafe(arr, lat, lng) {
  if (!alreadyTaken(lat, lng) && insideFrance(lat, lng)) {
    arr.push([lat, lng]);
    taken.push([lat, lng]);
  }
}

function polarOffsetKm(lat, lng, minKm, maxKm) {
  const θ = RNG() * 2 * Math.PI;
  const r = minKm + RNG() * (maxKm - minKm);
  const km2deg = 1 / 111;
  return [lat + r * km2deg * Math.cos(θ), lng + r * km2deg * Math.sin(θ)];
}

const OFFSET = {
  hub:      { rKm: [8, 15] },
  delivery: { rKm: [0, 0] },
  pickup:   { rKm: [2, 5] }
};

export async function pickCityPoints() {
  try {
    const resp = await fetch('/data/cities.json');
    if (!resp.ok) {
      throw new Error('Impossible de charger les données des villes');
    }
    const { cities } = await resp.json();

    const big = cities.filter(c => c.pop > CONFIG.BIG_CITY_THRESHOLD)
                     .sort((a, b) => b.pop - a.pop);
    const medium = cities.filter(c => c.pop <= CONFIG.BIG_CITY_THRESHOLD);

    taken.length = 0;

    const hubs = [];
    const delivery = [];
    const pickup = [];
    const charging = [];

    big.forEach(city => {
      pushSafe(delivery, city.lat, city.lng);

      const [hLat, hLng] = polarOffsetKm(city.lat, city.lng, ...OFFSET.hub.rKm);
      pushSafe(hubs, hLat, hLng);

      const [pLat, pLng] = polarOffsetKm(city.lat, city.lng, ...OFFSET.pickup.rKm);
      pushSafe(pickup, pLat, pLng);

      for (let i = 0; i < CONFIG.RADIAL_PER_CITY; i++) {
        const θ = (i / CONFIG.RADIAL_PER_CITY) * 2 * Math.PI;
        const kmToDeg = 1 / 111;
        const rMin = CONFIG.RADIAL_R_MIN_KM;
        const rMax = CONFIG.RADIAL_R_MAX_KM;
        const r = rMin + RNG() * (rMax - rMin);
        const lat = city.lat + r * kmToDeg * Math.cos(θ);
        const lng = city.lng + r * kmToDeg * Math.sin(θ);
        if (insideFrance(lat, lng)) pushSafe(charging, lat, lng);
      }
    });

    for (let i = 0; i < big.length; i++) {
      for (let j = i + 1; j < big.length; j++) {
        const A = big[i], B = big[j];
        const d = haversineDistance(A.lat, A.lng, B.lat, B.lng);
        if (d < CONFIG.MIN_CITY_DISTANCE_KM) continue;
        const n = Math.floor(d / CONFIG.RELAY_CHARGING_EVERY_KM);
        for (let k = 1; k <= n; k++) {
          const t = k / (n + 1);
          const lat = A.lat + t * (B.lat - A.lat);
          const lng = A.lng + t * (B.lng - A.lng);
          if (!insideFrance(lat, lng)) continue;
          pushSafe(charging, lat, lng);
        }
      }
    }

    console.log("Current counts before completion:",
                hubs.length, delivery.length, pickup.length, charging.length);
    console.log("Target counts:",
                CONFIG.DEFAULT_PARAMS.hubCount, CONFIG.DEFAULT_PARAMS.deliveryCount,
                CONFIG.DEFAULT_PARAMS.pickupCount, CONFIG.DEFAULT_PARAMS.chargingCount);

    complete(hubs, CONFIG.DEFAULT_PARAMS.hubCount, medium, CONFIG.MIN_DIST_HUB_CHARGING);

    complete(delivery, CONFIG.DEFAULT_PARAMS.deliveryCount, medium);
    complete(pickup, CONFIG.DEFAULT_PARAMS.pickupCount, medium);

    complete(charging, CONFIG.DEFAULT_PARAMS.chargingCount, cities, CONFIG.MIN_DIST_HUB_CHARGING);

    console.log("Final counts after completion:",
                hubs.length, delivery.length, pickup.length, charging.length);

    console.log(
      "[city-picker] big cities:", big.length,
      "hubs:", hubs.length,
      "delivery:", delivery.length,
      "pickup:", pickup.length,
      "charging:", charging.length
    );

    return { hubs, charging, delivery, pickup };

  } catch (error) {
    console.error('Erreur lors du chargement des villes:', error);
    return { hubs: [], charging: [], delivery: [], pickup: [] };
  }
}
