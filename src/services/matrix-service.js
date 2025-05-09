const osrmService = require('./osrm-service');
const fs = require('fs').promises;
const path = require('path');
const zlib = require('zlib');
const util = require('util');

// Promisification des fonctions zlib
const gzip = util.promisify(zlib.gzip);
const gunzip = util.promisify(zlib.gunzip);

// Chemins vers les différents formats de fichiers
const MATRIX_DIR = path.join(__dirname, '../../data');
const MATRIX_FILE_PATH = path.join(MATRIX_DIR, 'distance-matrix.json');
const MATRIX_GZ_FILE_PATH = path.join(MATRIX_DIR, 'distance-matrix.json.gz');
const MATRIX_COMPACT_FILE_PATH = path.join(MATRIX_DIR, 'distance-matrix.compact.gz');

/**
 * Structure pour stocker la matrice des distances et les statistiques
 */
class DistanceMatrix {
  constructor() {
    this.matrix = {};
    this.initialized = false;
    this.minCost = Infinity;
    this.maxCost = 0;
    this.avgCost = 0;
    this.lastUpdated = null;
    this.fileSize = 0;
    this.format = 'json'; // Format par défaut: 'json', 'json-gz', 'compact'
  }

  /**
   * Calcule une clé unique pour une paire de points
   * @param {string} pointA - ID du premier point
   * @param {string} pointB - ID du deuxième point
   * @returns {string} - Clé unique
   */
  getKey(pointA, pointB) {
    // Toujours mettre les points dans le même ordre pour garantir l'unicité
    return [pointA, pointB].sort().join('-');
  }

  /**
   * Ajoute une entrée dans la matrice
   * @param {string} pointA - ID du premier point
   * @param {string} pointB - ID du deuxième point
   * @param {Object} data - Données de route entre les points
   */
  addEntry(pointA, pointB, data) {
    const key = this.getKey(pointA, pointB);
    
    // Optimiser les données pour réduire la taille
    const optimizedData = this.optimizeRouteData(data);
    
    this.matrix[key] = optimizedData;

    // Mettre à jour les statistiques
    if (data.cost < this.minCost) this.minCost = data.cost;
    if (data.cost > this.maxCost) this.maxCost = data.cost;
    
    this.lastUpdated = new Date();
  }

  /**
   * Optimise les données de route pour réduire la taille
   * @param {Object} data - Données complètes de la route
   * @returns {Object} - Données optimisées
   */
  optimizeRouteData(data) {
    // Ne conserver que les données essentielles
    const optimized = {
      // Informations de base (toujours nécessaires)
      fromId: data.fromId,
      toId: data.toId,
      fromName: data.fromName,
      toName: data.toName,
      distance: data.distance,
      duration: data.duration,
      cost: data.cost,
      
      // Simplifier la géométrie pour réduire la taille
      geometry: this.simplifyGeometry(data.geometry)
    };
    
    return optimized;
  }
  
  /**
   * Simplifie la géométrie pour réduire la taille
   * @param {Object} geometry - Géométrie GeoJSON
   * @returns {Object} - Géométrie simplifiée
   */
  simplifyGeometry(geometry) {
    if (!geometry || !geometry.coordinates || geometry.coordinates.length <= 2) {
      return geometry;
    }
    
    // Simplifier en réduisant le nombre de points (prendre 1 point sur n)
    const simplifyFactor = 5; // Prendre 1 point tous les 5 points
    const simplified = {
      type: geometry.type,
      coordinates: []
    };
    
    // Toujours garder le premier et le dernier point
    simplified.coordinates.push(geometry.coordinates[0]);
    
    // Sélectionner certains points intermédiaires
    for (let i = simplifyFactor; i < geometry.coordinates.length - simplifyFactor; i += simplifyFactor) {
      simplified.coordinates.push(geometry.coordinates[i]);
    }
    
    // Ajouter le dernier point
    simplified.coordinates.push(geometry.coordinates[geometry.coordinates.length - 1]);
    
    return simplified;
  }

  /**
   * Restaure la géométrie complète entre deux points si nécessaire
   * @param {string} pointA - ID du premier point
   * @param {string} pointB - ID du deuxième point 
   */
  async restoreFullGeometry(pointA, pointB) {
    try {
      const key = this.getKey(pointA, pointB);
      const entry = this.matrix[key];
      
      if (!entry) return false;
      
      // Si la géométrie a peu de points, c'est probablement une version simplifiée
      if (entry.geometry.coordinates.length < 10) {
        // Récupérer les coordonnées des points
        const fromPoint = strategicPoints.features.find(f => f.properties.id === entry.fromId);
        const toPoint = strategicPoints.features.find(f => f.properties.id === entry.toId);
        
        if (!fromPoint || !toPoint) return false;
        
        const fromCoords = `${fromPoint.geometry.coordinates[0]},${fromPoint.geometry.coordinates[1]}`;
        const toCoords = `${toPoint.geometry.coordinates[0]},${toPoint.geometry.coordinates[1]}`;
        
        // Récupérer la géométrie complète depuis OSRM
        const route = await osrmService.getRoute(fromCoords, toCoords);
        
        // Mettre à jour la géométrie
        entry.geometry = route.geometry;
        
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Error restoring geometry:', error);
      return false;
    }
  }

  /**
   * Récupère les données pour une paire de points
   * @param {string} pointA - ID du premier point
   * @param {string} pointB - ID du deuxième point
   * @returns {Object|null} - Données de route ou null si non disponibles
   */
  getEntry(pointA, pointB) {
    const key = this.getKey(pointA, pointB);
    return this.matrix[key] || null;
  }

  /**
   * Calcule un score normalisé pour un coût donné
   * @param {number} cost - Le coût à normaliser
   * @returns {number} - Score entre 0 et 1 (0 = min, 1 = max)
   */
  getNormalizedScore(cost) {
    if (this.maxCost === this.minCost) return 0.5; // Éviter division par zéro
    return (cost - this.minCost) / (this.maxCost - this.minCost);
  }

  /**
   * Fournit les statistiques de la matrice
   * @returns {Object} - Statistiques de la matrice
   */
  getStats() {
    return {
      minCost: this.minCost,
      maxCost: this.maxCost,
      avgCost: this.avgCost,
      entries: Object.keys(this.matrix).length,
      initialized: this.initialized,
      lastUpdated: this.lastUpdated,
      fileSize: this.fileSize,
      format: this.format
    };
  }

  /**
   * Charge les données à partir d'un objet sérialisé
   * @param {Object} data - Données sérialisées
   */
  loadFromData(data) {
    if (!data) return;
    
    this.matrix = data.matrix || {};
    this.minCost = data.minCost || Infinity;
    this.maxCost = data.maxCost || 0;
    this.avgCost = data.avgCost || 0;
    this.initialized = data.initialized || false;
    this.lastUpdated = data.lastUpdated ? new Date(data.lastUpdated) : null;
    this.format = data.format || 'json';
    
    console.log(`Loaded matrix with ${Object.keys(this.matrix).length} entries.`);
    if (this.initialized) {
      console.log(`Matrix stats: Min cost: ${this.minCost.toFixed(2)}, Max cost: ${this.maxCost.toFixed(2)}`);
    }
  }

  /**
   * Prépare les données pour la sérialisation
   * @returns {Object} - Données sérialisables
   */
  toJSON() {
    return {
      matrix: this.matrix,
      minCost: this.minCost,
      maxCost: this.maxCost,
      avgCost: this.avgCost,
      initialized: this.initialized,
      lastUpdated: this.lastUpdated,
      format: this.format
    };
  }
  
  /**
   * Conversion en format compact pour optimiser la taille
   * @returns {Object} - Format compact pour le stockage
   */
  toCompact() {
    // Structure de données plus efficace pour le stockage
    const compact = {
      meta: {
        minCost: this.minCost,
        maxCost: this.maxCost,
        avgCost: this.avgCost,
        initialized: this.initialized,
        lastUpdated: this.lastUpdated ? this.lastUpdated.toISOString() : null,
        format: 'compact'
      },
      // Dictionnaires pour éviter de répéter les chaînes de caractères
      ids: [],
      names: [],
      routes: [] // Format: [fromIndex, toIndex, distance, duration, cost, [coords]]
    };
    
    // Construire les dictionnaires d'IDs et de noms
    const idMap = new Map();
    const nameMap = new Map();
    
    Object.values(this.matrix).forEach(entry => {
      if (entry.fromId && !idMap.has(entry.fromId)) {
        idMap.set(entry.fromId, idMap.size);
        compact.ids.push(entry.fromId);
      }
      if (entry.toId && !idMap.has(entry.toId)) {
        idMap.set(entry.toId, idMap.size);
        compact.ids.push(entry.toId);
      }
      if (entry.fromName && !nameMap.has(entry.fromName)) {
        nameMap.set(entry.fromName, nameMap.size);
        compact.names.push(entry.fromName);
      }
      if (entry.toName && !nameMap.has(entry.toName)) {
        nameMap.set(entry.toName, nameMap.size);
        compact.names.push(entry.toName);
      }
    });
    
    // Conversion des routes au format compact
    Object.values(this.matrix).forEach(entry => {
      const fromIndex = idMap.get(entry.fromId);
      const toIndex = idMap.get(entry.toId);
      
      // Comprimer les coordonnées de géométrie si disponibles
      let compressedCoords = [];
      if (entry.geometry && entry.geometry.coordinates) {
        compressedCoords = entry.geometry.coordinates.map(coord => 
          [Math.round(coord[0] * 10000) / 10000, Math.round(coord[1] * 10000) / 10000]
        );
      }
      
      compact.routes.push([
        fromIndex, 
        toIndex, 
        Math.round(entry.distance), 
        Math.round(entry.duration), 
        Math.round(entry.cost * 100) / 100,
        compressedCoords
      ]);
    });
    
    return compact;
  }
  
  /**
   * Charge depuis un format compact
   * @param {Object} compact - Données au format compact
   */
  loadFromCompact(compact) {
    if (!compact || !compact.meta || !Array.isArray(compact.ids) || !Array.isArray(compact.routes)) {
      console.error('Invalid compact format');
      return false;
    }
    
    // Charger les métadonnées
    this.minCost = compact.meta.minCost || Infinity;
    this.maxCost = compact.meta.maxCost || 0;
    this.avgCost = compact.meta.avgCost || 0;
    this.initialized = compact.meta.initialized || false;
    this.lastUpdated = compact.meta.lastUpdated ? new Date(compact.meta.lastUpdated) : null;
    this.format = 'compact';
    
    // Reconstruire la matrice
    this.matrix = {};
    
    compact.routes.forEach(route => {
      const [fromIndex, toIndex, distance, duration, cost, coords] = route;
      
      const fromId = compact.ids[fromIndex];
      const toId = compact.ids[toIndex];
      const fromName = compact.names[fromIndex] || fromId;
      const toName = compact.names[toIndex] || toId;
      
      // Reconstruire la géométrie
      const geometry = {
        type: 'LineString',
        coordinates: coords || []
      };
      
      // Créer l'entrée
      const entry = {
        fromId,
        toId,
        fromName,
        toName,
        distance,
        duration,
        cost,
        geometry
      };
      
      // Ajouter à la matrice
      const key = this.getKey(fromId, toId);
      this.matrix[key] = entry;
    });
    
    console.log(`Loaded compact matrix with ${Object.keys(this.matrix).length} entries.`);
    return true;
  }
}

// Instance singleton de la matrice
const distanceMatrix = new DistanceMatrix();

/**
 * Calcule la matrice de distances pour tous les points stratégiques
 * @param {Array} strategicPoints - Points stratégiques
 * @returns {Promise<DistanceMatrix>} - Matrice calculée
 */
async function calculateMatrix(strategicPoints) {
  console.log('Calculating distance matrix...');

  // Extraction des points
  const points = strategicPoints.features.map(feature => ({
    id: feature.properties.id,
    name: feature.properties.name,
    coordinates: `${feature.geometry.coordinates[0]},${feature.geometry.coordinates[1]}`
  }));

  let totalCost = 0;
  let count = 0;

  // Calcul des routes entre paires de points
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      try {
        const from = points[i].coordinates;
        const to = points[j].coordinates;
        
        // Vérifier si cette entrée existe déjà
        if (distanceMatrix.getEntry(points[i].id, points[j].id)) {
          continue;
        }
        
        const route = await osrmService.getRoute(from, to);
        
        // Calcul du coût avec facteurs régionaux
        const [fromLon, fromLat] = from.split(',').map(parseFloat);
        const [toLon, toLat] = to.split(',').map(parseFloat);
        
        // Facteur régional (comme dans server.js)
        let regionFactor = 1.0;
        
        if (fromLat > 48 || toLat > 48) {
          regionFactor = 1.5;  // Nord
        } else if (fromLat < 44 || toLat < 44) {
          regionFactor = 1.2;  // Sud
        } else if (fromLon > 5 || toLon > 5) {
          regionFactor = 1.1;  // Est
        } else if (fromLon < 0 || toLon < 0) {
          regionFactor = 0.9;  // Ouest
        }
        
        // Calcul du coût final
        const cost = route.distance * regionFactor;
        
        // Ajout dans la matrice
        distanceMatrix.addEntry(points[i].id, points[j].id, { 
          ...route, 
          cost,
          fromId: points[i].id,
          toId: points[j].id,
          fromName: points[i].name,
          toName: points[j].name
        });
        
        totalCost += cost;
        count++;
        
        console.log(`Calculated route: ${points[i].name} to ${points[j].name}`);
      } catch (error) {
        console.error(`Error calculating route from ${points[i].name} to ${points[j].name}:`, error.message);
      }
    }
  }
  
  // Calcul du coût moyen
  distanceMatrix.avgCost = count > 0 ? totalCost / count : 0;
  distanceMatrix.initialized = true;
  
  console.log('Distance matrix calculation completed.');
  console.log(`Stats: Min cost: ${distanceMatrix.minCost.toFixed(2)}, Max cost: ${distanceMatrix.maxCost.toFixed(2)}, Avg: ${distanceMatrix.avgCost.toFixed(2)}`);
  
  // Sauvegarde automatique après le calcul
  await saveMatrixToFile();
  
  return distanceMatrix;
}

/**
 * Sauvegarde la matrice de distances dans un fichier
 * @param {string} format - Format de sauvegarde ('json', 'json-gz', 'compact')
 * @returns {Promise<boolean>} - Succès de la sauvegarde
 */
async function saveMatrixToFile(format = 'json-gz') {
  try {
    // Assurer que le dossier data existe
    try {
      await fs.mkdir(MATRIX_DIR, { recursive: true });
    } catch (err) {
      // Ignorer si le dossier existe déjà
      if (err.code !== 'EEXIST') throw err;
    }

    let data;
    let filePath;
    distanceMatrix.format = format;

    switch (format) {
      case 'json':
        // Format JSON standard
        data = JSON.stringify(distanceMatrix.toJSON(), null, 2);
        filePath = MATRIX_FILE_PATH;
        break;
      
      case 'json-gz':
        // JSON compressé avec gzip
        data = await gzip(JSON.stringify(distanceMatrix.toJSON()));
        filePath = MATRIX_GZ_FILE_PATH;
        break;
      
      case 'compact':
        // Format compact compressé
        data = await gzip(JSON.stringify(distanceMatrix.toCompact()));
        filePath = MATRIX_COMPACT_FILE_PATH;
        break;
      
      default:
        throw new Error(`Format non supporté: ${format}`);
    }

    // Écrire le fichier
    await fs.writeFile(filePath, data);
    
    // Mettre à jour la taille du fichier
    const stats = await fs.stat(filePath);
    distanceMatrix.fileSize = stats.size;
    
    const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);
    console.log(`Matrix saved to ${filePath} (${fileSizeMB} MB) in ${format} format`);
    
    return true;
  } catch (error) {
    console.error('Error saving matrix to file:', error);
    return false;
  }
}

/**
 * Charge la matrice de distances depuis un fichier
 * @param {string} format - Format de chargement ('json', 'json-gz', 'compact')
 * @returns {Promise<boolean>} - Succès du chargement
 */
async function loadMatrixFromFile(format = 'auto') {
  try {
    let filePath;
    let data;
    let parsedData;
    
    // Déterminer le format et le fichier à charger
    if (format === 'auto') {
      // Essayer dans l'ordre: compact, json-gz, json
      if (await fileExists(MATRIX_COMPACT_FILE_PATH)) {
        format = 'compact';
        filePath = MATRIX_COMPACT_FILE_PATH;
      } else if (await fileExists(MATRIX_GZ_FILE_PATH)) {
        format = 'json-gz';
        filePath = MATRIX_GZ_FILE_PATH;
      } else if (await fileExists(MATRIX_FILE_PATH)) {
        format = 'json';
        filePath = MATRIX_FILE_PATH;
      } else {
        console.log('No matrix file found in any format');
        return false;
      }
    } else {
      // Format spécifié
      switch (format) {
        case 'json':
          filePath = MATRIX_FILE_PATH;
          break;
        case 'json-gz':
          filePath = MATRIX_GZ_FILE_PATH;
          break;
        case 'compact':
          filePath = MATRIX_COMPACT_FILE_PATH;
          break;
        default:
          throw new Error(`Format non supporté: ${format}`);
      }
      
      // Vérifier l'existence du fichier
      if (!await fileExists(filePath)) {
        console.log(`Matrix file not found in ${format} format`);
        return false;
      }
    }
    
    // Mettre à jour la taille du fichier
    const stats = await fs.stat(filePath);
    distanceMatrix.fileSize = stats.size;
    
    // Lire et parser le fichier selon le format
    switch (format) {
      case 'json':
        data = await fs.readFile(filePath, 'utf8');
        parsedData = JSON.parse(data);
        distanceMatrix.loadFromData(parsedData);
        break;
      
      case 'json-gz':
        data = await fs.readFile(filePath);
        const decompressed = await gunzip(data);
        parsedData = JSON.parse(decompressed.toString());
        distanceMatrix.loadFromData(parsedData);
        break;
      
      case 'compact':
        data = await fs.readFile(filePath);
        const decompressedCompact = await gunzip(data);
        parsedData = JSON.parse(decompressedCompact.toString());
        distanceMatrix.loadFromCompact(parsedData);
        break;
    }
    
    const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);
    console.log(`Matrix loaded from ${filePath} (${fileSizeMB} MB) in ${format} format`);
    
    return true;
  } catch (error) {
    // Si le fichier n'existe pas, ce n'est pas une erreur critique
    if (error.code === 'ENOENT') {
      console.log('No existing matrix file found. Will create a new one.');
    } else {
      console.error('Error loading matrix from file:', error);
    }
    return false;
  }
}

/**
 * Vérifie si un fichier existe
 * @param {string} filePath - Chemin du fichier
 * @returns {Promise<boolean>} - Le fichier existe ou non
 */
async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch (error) {
    return false;
  }
}

/**
 * Convertit le format de la matrice
 * @param {string} targetFormat - Format cible ('json', 'json-gz', 'compact')
 * @returns {Promise<boolean>} - Succès de la conversion
 */
async function convertMatrixFormat(targetFormat) {
  if (!distanceMatrix.initialized) {
    console.log('Matrix not initialized, cannot convert');
    return false;
  }
  
  try {
    console.log(`Converting matrix to ${targetFormat} format...`);
    const success = await saveMatrixToFile(targetFormat);
    
    if (success) {
      console.log(`Matrix successfully converted to ${targetFormat} format`);
      return true;
    } else {
      console.error(`Failed to convert matrix to ${targetFormat} format`);
      return false;
    }
  } catch (error) {
    console.error('Error converting matrix format:', error);
    return false;
  }
}

module.exports = {
  distanceMatrix,
  calculateMatrix,
  saveMatrixToFile,
  loadMatrixFromFile,
  convertMatrixFormat
};
