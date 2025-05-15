// Import functions from city-picker module
import { pickCityPoints, CONFIG } from '/js/city-picker.js';
import { haversineDistance, gaussianRandom, rgbToHex, RNG, insideFrance, setFrancePolygon } from './utils.js';

// Map initialization code - only once
const map = L.map('map').setView([46.6, 2.3], 6);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Create LayerGroups for different point types
const hubsLayer = L.layerGroup().addTo(map);
const chargingLayer = L.layerGroup().addTo(map);
const deliveryLayer = L.layerGroup().addTo(map);
const pickupLayer = L.layerGroup().addTo(map);
const edgesLayer = L.layerGroup().addTo(map);

// Define colors for each category
const COLORS = {
  hubs: '#e74c3c',      // Red
  charging: '#2980b9',  // Blue
  delivery: '#27ae60',  // Green
  pickup: '#f39c12'     // Orange
};

// Only keeping useful parameter
const K_NEIGHBORS = 10;

// Global variables for graph representation
let allNodes = [];
let allEdges = [];
let windAngle = RNG() * 2 * Math.PI; // Random wind angle in radians but reproducible
let highlightedNodeId = null;
let francePoly;

// Global variables for drone settings, initialized with default values
window.batteryCapacity = 100;
window.maxPayload = 3;

// Build k-nearest neighbors graph from nodes
function buildGraph(nodes, k = K_NEIGHBORS) {
  const edges = [];
  
  for (let i = 0; i < nodes.length; i++) {
    // Calculate distances to all other nodes
    const distances = nodes.map((node, j) => ({
      index: j,
      distance: haversineDistance(
        nodes[i].lat, nodes[i].lng,
        node.lat, node.lng
      )
    })).sort((a, b) => a.distance - b.distance);
    
    // Connect to k nearest neighbors (skip the first one as it's the node itself)
    distances.slice(1, k + 1).forEach(({ index, distance }) => {
      edges.push({
        source: i,
        target: index,
        distance: distance
      });
    });
  }
  
  return edges;
}

// Annotate edges with wind effects and costs
function annotateEdges(edges, nodes, windAngle, alpha = 0.3, beta = 0.05) {
  return edges.map(edge => {
    const sourceNode = nodes[edge.source];
    const targetNode = nodes[edge.target];
    
    // Calculate angle of the edge
    const phi = Math.atan2(
      targetNode.lat - sourceNode.lat,
      targetNode.lng - sourceNode.lng
    );
    
    // Calculate wind effect (-1 to 1, -1 being headwind, 1 being tailwind)
    const windEffect = Math.cos(phi - windAngle);
    
    // Add random noise
    const noise = gaussianRandom(0, 0.05);
    
    // Calculate final cost - tailwind friendly formula
    const cost = edge.distance * (1 - alpha * windEffect + beta * noise);
    
    return {
      ...edge,
      windEffect,
      phi,
      cost
    };
  });
}

// Draw the graph on the map
function drawGraph(nodes, edges) {
  // Clear existing edges
  edgesLayer.clearLayers();
  
  // Calculate min, mean, and max costs for color scaling
  const costs = edges.map(e => e.cost);
  const minCost = Math.min(...costs);
  const maxCost = Math.max(...costs);
  const meanCost = costs.reduce((sum, cost) => sum + cost, 0) / costs.length;
  
  // Create a color scale function
  const getColor = (cost) => {
    if (cost <= meanCost) {
      // Scale from green to yellow
      const t = (cost - minCost) / (meanCost - minCost);
      return rgbToHex(
        Math.round(46 + t * (241 - 46)),
        Math.round(204 + t * (196 - 204)),
        Math.round(113 + t * (15 - 113))
      );
    } else {
      // Scale from yellow to red
      const t = (cost - meanCost) / (maxCost - meanCost);
      return rgbToHex(
        Math.round(241 - t * (241 - 231)),
        Math.round(196 - t * (196 - 76)),
        Math.round(15 - t * (15 - 60))
      );
    }
  };

  // Draw all edges
  edges.forEach(edge => {
    const source = nodes[edge.source];
    const target = nodes[edge.target];
    const color = getColor(edge.cost);
    
    // Edge thickness inversely proportional to cost but reduced overall
    const weight = 0.5 + 1.5 * (1 - (edge.cost - minCost) / (maxCost - minCost));
    
    const polyline = L.polyline(
      [[source.lat, source.lng], [target.lat, target.lng]], 
      {
        color: color,
        weight: weight,
        opacity: 0.1, // Further reduced opacity
        zIndex: 100  // Make sure edges are below nodes
      }
    ).addTo(edgesLayer);
    
    // Add arrow decoration with smaller size and reduced opacity
    const decorator = L.polylineDecorator(polyline, {
      patterns: [
        {
          offset: '70%',
          repeat: 0,
          symbol: L.Symbol.arrowHead({
            pixelSize: 3, // Even smaller arrow size
            polygon: true,
            pathOptions: { color: color, fillOpacity: 0.1 } // Further reduced opacity
          })
        }
      ]
    }).addTo(edgesLayer);
    
    // Add tooltip with edge information
    polyline.bindTooltip(`
      <b>${source.id} → ${target.id}</b><br>
      Distance: ${edge.distance.toFixed(2)} km<br>
      Wind effect: ${edge.windEffect.toFixed(2)}<br>
      Cost: ${edge.cost.toFixed(2)}
    `, { sticky: true });
    
    // Store edge data for highlighting
    polyline.sourceIndex = edge.source;
    polyline.targetIndex = edge.target;
  });
}

// Highlight edges connected to a node
function highlightNodeConnections(nodeIndex) {
  // Reset all edges to default style
  edgesLayer.eachLayer(layer => {
    if (layer instanceof L.Polyline) {
      const edge = allEdges.find(e => 
        e.source === layer.sourceIndex && e.target === layer.targetIndex
      );
      
      if (edge) {
        const costs = allEdges.map(e => e.cost);
        const minCost = Math.min(...costs);
        const maxCost = Math.max(...costs);
        const weight = 0.5 + 1.5 * (1 - (edge.cost - minCost) / (maxCost - minCost));
        
        layer.setStyle({
          weight: weight,
          opacity: nodeIndex === layer.sourceIndex ? 0.7 : 0.15 // Highlight active, keep others faded
        });
      }
    }
  });
  
  // If a node is selected, highlight its outgoing edges
  if (nodeIndex !== null) {
    edgesLayer.eachLayer(layer => {
      if (layer instanceof L.Polyline && layer.sourceIndex === nodeIndex) {
        layer.setStyle({
          weight: 2, // Slightly thicker for highlighted edges
          opacity: 0.8 // More visible but still not too bright
        });
        layer.bringToFront();
      }
    });
  }
}

// Update the wind direction compass
function updateWindCompass(angle) {
  const arrowEl = document.getElementById('wind-arrow');
  if (arrowEl) {
    arrowEl.style.transform = `translate(-50%, -50%) rotate(${angle * 180 / Math.PI}deg)`;
  }
}

// Fetch France GeoJSON and initialize the map
fetch('/data/metropole-version-simplifiee.geojson')
  .then(response => {
    if (!response.ok) {
      throw new Error(`Problème de chargement: ${response.status}`);
    }
    return response.json();
  })
  .then(geojson => {
    console.log("GeoJSON chargé avec succès");
    francePoly = geojson;
    setFrancePolygon(francePoly);
    
    // Display the outline on the map
    L.geoJSON(francePoly, { 
      weight: 2, 
      color: '#444', 
      fill: false 
    }).addTo(map);
    
    // Fit the map view to France's boundaries
    map.fitBounds(L.geoJSON(francePoly).getBounds());
    
    // Generate the network once the boundaries are loaded
    generateNetwork();
  })
  .catch(err => {
    console.error('Erreur de chargement des frontières:', err);
    alert('Problème avec le chargement des frontières françaises. Utilisation du mode dégradé.');
    
    // Fallback mode: use an approximate rectangle for France
    const franceBbox = {
      type: "Feature",
      properties: { name: "France (approximatif)" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [-5.2, 41.3], [9.5, 41.3], [9.5, 51.1], [-5.2, 51.1], [-5.2, 41.3]
        ]]
      }
    };
    
    francePoly = franceBbox;
    setFrancePolygon(francePoly);
    L.rectangle([[41.3, -5.2], [51.1, 9.5]], {color: "#444", weight: 1, fill: false}).addTo(map);
    generateNetwork();
  });

/**
 * Generate the delivery network
 */
async function generateNetwork() {
  if (!francePoly) {
    alert("Les frontières de la France ne sont pas encore chargées. Veuillez patienter.");
    return;
  }
  
  // Clear all existing layers
  hubsLayer.clearLayers();
  chargingLayer.clearLayers();
  deliveryLayer.clearLayers();
  pickupLayer.clearLayers();
  edgesLayer.clearLayers();
  
  // Get points using the city picker instead of Poisson generation
  const { hubs: hubPoints, charging: chargingPoints, delivery: deliveryPoints, pickup: pickupPoints } = await pickCityPoints();

  // Ajoute ce log pour vérifier la génération
  console.log("[app] hubs:", hubPoints.length, "delivery:", deliveryPoints.length, "pickup:", pickupPoints.length, "charging:", chargingPoints.length);
  
  // Log counts for debugging
  console.log("Point counts (hubs, delivery, pickup, charging):", 
              hubPoints.length, deliveryPoints.length, pickupPoints.length, chargingPoints.length);
  
  // Reset nodes array
  allNodes = [];
  
  // Add points to their respective layers with higher z-index
  hubPoints.forEach((latlng, index) => {
    if (!insideFrance(latlng[0], latlng[1])) return; // garde-fou ultime
    const id = `Hub ${index + 1}`;
    const marker = L.circleMarker(latlng, {
      color: COLORS.hubs,
      fillColor: COLORS.hubs,
      fillOpacity: 0.8,
      weight: 2,
      radius: 8,
      zIndex: 1000 // Ensure points appear above edges
    }).addTo(hubsLayer);
    marker.bindTooltip(id);
    
    // Add click event for highlighting
    marker.on('click', function() {
      const nodeIndex = allNodes.findIndex(n => n.id === id);
      highlightedNodeId = highlightedNodeId === nodeIndex ? null : nodeIndex;
      highlightNodeConnections(highlightedNodeId);
    });
    
    // Add to nodes array
    allNodes.push({
      id: id,
      lat: latlng[0],
      lng: latlng[1],
      type: 'hubs'
    });
  });
  
  // Similar handling for other point types
  chargingPoints.forEach((latlng, index) => {
    if (!insideFrance(latlng[0], latlng[1])) return;
    const id = `Charging ${index + 1}`;
    const marker = L.circleMarker(latlng, {
      color: COLORS.charging,
      fillColor: COLORS.charging,
      fillOpacity: 0.8,
      weight: 1,
      radius: 6,
      zIndex: 1000
    }).addTo(chargingLayer);
    marker.bindTooltip(id);
    
    marker.on('click', function() {
      const nodeIndex = allNodes.findIndex(n => n.id === id);
      highlightedNodeId = highlightedNodeId === nodeIndex ? null : nodeIndex;
      highlightNodeConnections(highlightedNodeId);
    });
    
    allNodes.push({
      id: id,
      lat: latlng[0],
      lng: latlng[1],
      type: 'charging'
    });
  });
  
  deliveryPoints.forEach((latlng, index) => {
    if (!insideFrance(latlng[0], latlng[1])) return;
    const id = `Delivery ${index + 1}`;
    const marker = L.circleMarker(latlng, {
      color: COLORS.delivery,
      fillColor: COLORS.delivery,
      fillOpacity: 0.8,
      weight: 1,
      radius: 5,
      zIndex: 1000
    }).addTo(deliveryLayer);
    marker.bindTooltip(id);
    
    marker.on('click', function() {
      const nodeIndex = allNodes.findIndex(n => n.id === id);
      highlightedNodeId = highlightedNodeId === nodeIndex ? null : nodeIndex;
      highlightNodeConnections(highlightedNodeId);
    });
    
    allNodes.push({
      id: id,
      lat: latlng[0],
      lng: latlng[1],
      type: 'delivery'
    });
  });
  
  pickupPoints.forEach((latlng, index) => {
    if (!insideFrance(latlng[0], latlng[1])) return;
    const id = `Pickup ${index + 1}`;
    const marker = L.circleMarker(latlng, {
      color: COLORS.pickup,
      fillColor: COLORS.pickup,
      fillOpacity: 0.8,
      weight: 1,
      radius: 5,
      zIndex: 1000
    }).addTo(pickupLayer);
    marker.bindTooltip(id);
    
    marker.on('click', function() {
      const nodeIndex = allNodes.findIndex(n => n.id === id);
      highlightedNodeId = highlightedNodeId === nodeIndex ? null : nodeIndex;
      highlightNodeConnections(highlightedNodeId);
    });
    
    allNodes.push({
      id: id,
      lat: latlng[0],
      lng: latlng[1],
      type: 'pickup'
    });
  });
  
  // Build graph if we have at least one node
  if (allNodes.length > 0) {
    // Fixed parameters for edge creation
    const alpha = 0.3;
    const beta = 0.05;
    
    // Build the k-nearest neighbors graph
    const rawEdges = buildGraph(allNodes, K_NEIGHBORS);
    
    // Add wind and cost effects
    allEdges = annotateEdges(rawEdges, allNodes, windAngle, alpha, beta);
    
    // Draw the graph
    drawGraph(allNodes, allEdges);

    // Affiche le coût maximal d'une arête dans la console
    const maxCost = Math.max(...allEdges.map(e => e.cost));
    console.log("Coût maximal d'une arête sur la carte :", maxCost);

    // Update the wind arrow display
    const windArrow = document.getElementById('wind-arrow');
    if (windArrow) {
      const degrees = Math.round((windAngle * 180 / Math.PI) % 360);
      windArrow.style.transform = `translate(-50%, -50%) rotate(${degrees}deg)`;
    }
  }
}

/**
 * Run the selected algorithm
 */
function runAlgorithm() {
  const algorithm = document.querySelector('.algorithm-toggle').getAttribute('data-selected');
  
  // Get drone settings from global variables set by UI components
  const batteryCapacity = window.batteryCapacity;
  const maxPayload = window.maxPayload;
  
  console.log(`Running ${algorithm.toUpperCase()} algorithm with:`, {
    batteryCapacity,
    maxPayload
  });
  
  // Make an AJAX call to the backend to run the algorithm
  fetch('/api/run-algorithm', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      algorithm,
      batteryCapacity,
      maxPayload,
      nodes: allNodes,
      edges: allEdges,
      windAngle
    })
  })
  .then(response => response.json())
  .then(data => {
    console.log('Success:', data);
    alert(`Algorithm completed! Distance: ${data.stats.distance} km, Battery used: ${data.stats.batteryUsed} units`);
  })
  .catch(error => {
    console.error('Error:', error);
    alert('Failed to run algorithm. Check the console for details.');
  });
}

// --- Wind compass interactive control ---
const windCompass = document.getElementById('wind-compass');
const windArrow = document.getElementById('wind-arrow');
const windAngleValue = document.getElementById('wind-angle-value');

function setWindAngleFromDegrees(degrees) {
  windAngle = (degrees * Math.PI / 180);
  if (windArrow) {
    windArrow.style.transform = `translate(-50%, -50%) rotate(${degrees}deg)`;
  }
  if (windAngleValue) {
    windAngleValue.textContent = `${Math.round(degrees)}°`;
  }
  // Recompute edge costs and redraw
  if (allNodes.length > 0 && allEdges.length > 0) {
    allEdges = annotateEdges(
      allEdges.map(e => ({ source: e.source, target: e.target, distance: e.distance })),
      allNodes,
      windAngle,
      0.3,
      0.05
    );
    drawGraph(allNodes, allEdges);
  }
}

let draggingWind = false;
function getAngleFromEvent(e) {
  const rect = windCompass.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - cx;
  const y = (e.touches ? e.touches[0].clientY : e.clientY) - cy;
  let angle = Math.atan2(x, -y) * 180 / Math.PI; // 0° = N, 90° = E
  if (angle < 0) angle += 360;
  return angle;
}

if (windCompass) {
  windCompass.addEventListener('mousedown', e => {
    draggingWind = true;
    setWindAngleFromDegrees(getAngleFromEvent(e));
  });
  windCompass.addEventListener('touchstart', e => {
    draggingWind = true;
    setWindAngleFromDegrees(getAngleFromEvent(e));
  });
  window.addEventListener('mousemove', e => {
    if (draggingWind) setWindAngleFromDegrees(getAngleFromEvent(e));
  });
  window.addEventListener('touchmove', e => {
    if (draggingWind) setWindAngleFromDegrees(getAngleFromEvent(e));
  });
  window.addEventListener('mouseup', () => { draggingWind = false; });
  window.addEventListener('touchend', () => { draggingWind = false; });
}

// Event listeners for UI controls
document.getElementById('toggle-hubs').addEventListener('change', function(e) {
  if (e.target.checked) {
    map.addLayer(hubsLayer);
  } else {
    map.removeLayer(hubsLayer);
  }
});

document.getElementById('toggle-charging').addEventListener('change', function(e) {
  if (e.target.checked) {
    map.addLayer(chargingLayer);
  } else {
    map.removeLayer(chargingLayer);
  }
});

document.getElementById('toggle-delivery').addEventListener('change', function(e) {
  if (e.target.checked) {
    map.addLayer(deliveryLayer);
  } else {
    map.removeLayer(deliveryLayer);
  }
});

document.getElementById('toggle-pickup').addEventListener('change', function(e) {
  if (e.target.checked) {
    map.addLayer(pickupLayer);
  } else {
    map.removeLayer(pickupLayer);
  }
});

document.getElementById('toggle-edges').addEventListener('change', function(e) {
  if (e.target.checked) {
    map.addLayer(edgesLayer);
  } else {
    map.removeLayer(edgesLayer);
  }
});

// Set up the run algorithm button
document.getElementById('run-algo').addEventListener('click', runAlgorithm);

// Map click handler
map.on('click', function(e) {
  // Reset highlighting when clicking on the map (not on a node)
  if (highlightedNodeId !== null) {
    highlightedNodeId = null;
    highlightNodeConnections(null);
  }
});
