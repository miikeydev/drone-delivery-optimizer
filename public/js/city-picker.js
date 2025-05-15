/**
 * Utilitaire pour tirer aléatoirement des villes françaises et les utiliser
 * comme points dans le réseau de livraison par drone
 */

/************** PARAMÈTRES **************************/
const BIG_CITIES_QUOTA = {
  hubs: 1,      // Hub par ville
  delivery: 1,  // Point de livraison
  pickup: 1,    // Point de collecte
  charging: 2   // Stations dans le périmètre urbain
};

// Paramètres par défaut pour les nombres souhaités
const DEFAULT_PARAMS = {
  hubCount: 20,
  chargingCount: 50,
  deliveryCount: 20,
  pickupCount: 20
};

// Seuil pour les grandes villes (en habitants)
const BIG_CITY_THRESHOLD = 200000; 

// Configuration des distances
const MIN_DIST_HUB_CHARGING = 0.02;  // ~2 km (en degrés lat/lon ≈ simplification)
const RELAY_CHARGING_EVERY_KM = 80;  // Station tous les 80 km sur les liaisons
const MIN_CITY_DISTANCE_KM = 120;    // Distance minimale pour placer des relais

// Paramètres pour les stations urbaines en anneau
const RADIAL_PER_CITY = 6;           // Nombre de bornes dans l'anneau
const RADIAL_R_MIN_KM = 5;           // Rayon minimum (km)
const RADIAL_R_MAX_KM = 10;          // Rayon maximum (km)
/***********************************************/

import { RNG } from './utils.js';
import { haversineDistance } from './utils.js';

/**
 * Mélange un tableau de manière aléatoire (Fisher-Yates shuffle)
 * @param {Array} array - Le tableau à mélanger
 * @returns {Array} - Le tableau mélangé
 */
function shuffle(array) {
  const arr = [...array]; // Copie pour ne pas modifier l'original
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(RNG() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]]; // Échange
  }
  return arr;
}

/**
 * Complète une liste de points jusqu'à atteindre le nombre cible
 * en respectant une distance minimale avec les points existants
 * 
 * @param {Array} list - Liste de points existants [lat, lng]
 * @param {number} target - Nombre cible à atteindre
 * @param {Array} pool - Liste de villes candidates {lat, lng}
 * @param {number} minDist - Distance minimum entre les points (en degrés)
 */
function complete(list, target, pool, minDist = 0) {
  const shuffled = shuffle(pool);
  for (const c of shuffled) {
    if (list.length >= target) break;
    
    // Vérifie que le point est suffisamment éloigné de tous les points existants
    const ok = list.every(p => 
      haversineDistance(p[0], p[1], c.lat, c.lng) > minDist);
    
    if (ok) list.push([c.lat, c.lng]);
  }
}

/**
 * Sélectionne les villes, crée les points obligatoires, puis complète
 * pour atteindre les quotas globaux.
 * 
 * @returns {Object} - Objet contenant les points pour chaque catégorie
 */
export async function pickCityPoints() {
  try {
    // Charger le fichier JSON des villes
    const resp = await fetch('/data/cities.json');
    if (!resp.ok) {
      throw new Error('Impossible de charger les données des villes');
    }
    
    const { cities } = await resp.json(); // [{name, lat, lng, pop}, …]
    
    // Séparer les grandes villes des villes moyennes
    const big = cities.filter(c => c.pop > BIG_CITY_THRESHOLD)
                     .sort((a, b) => b.pop - a.pop); // Gros → petit
    const medium = cities.filter(c => c.pop <= BIG_CITY_THRESHOLD);
    
    /* --------- 1. Points obligatoires par grande ville ---------- */
    const hubs = [];
    const delivery = [];
    const pickup = [];
    const charging = [];
    
    big.forEach(city => {
      // Ajouter un hub, un point de livraison et un point de collecte dans chaque grande ville
      hubs.push([city.lat, city.lng]);          // Hub
      delivery.push([city.lat, city.lng]);      // Delivery
      pickup.push([city.lat, city.lng]);        // Pickup
      
      // Ajouter des stations de charge en anneau autour de la ville
      for (let i = 0; i < RADIAL_PER_CITY; i++) {
        const θ = (i / RADIAL_PER_CITY) * 2 * Math.PI; // Répartition uniforme sur le cercle
        
        // Convertir km en degrés (approximation)
        const kmToDeg = 0.01; // ~1km ≈ 0.01 degré (approximation simplifiée)
        const rMin = RADIAL_R_MIN_KM * kmToDeg;
        const rMax = RADIAL_R_MAX_KM * kmToDeg;
        const r = rMin + RNG() * (rMax - rMin);
        
        charging.push([
          city.lat + r * Math.cos(θ),
          city.lng + r * Math.sin(θ)
        ]);
      }
    });

    /* --------- 2. Stations relais le long des axes ---------- */
    for (let i = 0; i < big.length; i++) {
      for (let j = i + 1; j < big.length; j++) {
        const A = big[i], B = big[j];
        const d = haversineDistance(A.lat, A.lng, B.lat, B.lng); // km
        
        if (d < MIN_CITY_DISTANCE_KM) continue; // Trop près, pas besoin de relais
        
        const n = Math.floor(d / RELAY_CHARGING_EVERY_KM);
        for (let k = 1; k <= n; k++) {
          const t = k / (n + 1); // Interpolation le long du segment
          charging.push([
            A.lat + t * (B.lat - A.lat),
            A.lng + t * (B.lng - A.lng)
          ]);
        }
      }
    }

    /* --------- 3. Compléter pour atteindre les quotas globaux ---------- */
    console.log("Current counts before completion:", 
                hubs.length, delivery.length, pickup.length, charging.length);
    console.log("Target counts:", 
                DEFAULT_PARAMS.hubCount, DEFAULT_PARAMS.deliveryCount, 
                DEFAULT_PARAMS.pickupCount, DEFAULT_PARAMS.chargingCount);

    // Hubs supplémentaires si besoin (dans les villes moyennes d'abord)
    complete(hubs, DEFAULT_PARAMS.hubCount, medium, MIN_DIST_HUB_CHARGING);

    // Delivery & pickup : on peut ré-utiliser la même logique
    complete(delivery, DEFAULT_PARAMS.deliveryCount, medium);
    complete(pickup, DEFAULT_PARAMS.pickupCount, medium);

    // Charging supplémentaires jusqu'au quota, en piochant partout
    complete(charging, DEFAULT_PARAMS.chargingCount, cities, MIN_DIST_HUB_CHARGING);

    console.log("Final counts after completion:", 
                hubs.length, delivery.length, pickup.length, charging.length);

    // LOG pour debug
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
    // Retourner un objet vide en cas d'erreur
    return { hubs: [], charging: [], delivery: [], pickup: [] };
  }
}
