// Import functions from city-picker module
// Importing CONFIG for potential future use and to document dependency
import { pickCityPoints, CONFIG } from '/js/city-picker.js';
import { haversineDistance, gaussianRandom, rgbToHex, RNG, insideFrance, setFrancePolygon } from './utils.js';

// Map initialization code - only once
const map = L.map('map').setView([46.6, 2.3], 6);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Create LayerGroups for different point types
const hubsLayer = L.layerGroup().addTo(map);
const chargingLayer = L.layerGroup();
const deliveryLayer = L.layerGroup().addTo(map);
const pickupLayer = L.layerGroup().addTo(map);
const edgesLayer = L.layerGroup();

// Remove chargingLayer and edgesLayer from map by default (since checkboxes are unchecked)
map.removeLayer(chargingLayer);
map.removeLayer(edgesLayer);

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
    
    // Export graph data for PPO training
    const graphData = {
      nodes: allNodes.map((node, index) => ({
        id: node.id,
        type: node.type,
        lat: node.lat,
        lng: node.lng,
        index: index
      })),
      edges: allEdges.map(edge => ({
        u: edge.source,        // 🔧 FIXED: Keep frontend format for compatibility
        v: edge.target,        // 🔧 FIXED: Keep frontend format for compatibility  
        dist: edge.distance,   // 🔧 FIXED: Keep frontend format for compatibility
        cost: edge.cost,
        // Also include training format for backward compatibility
        source: edge.source,
        target: edge.target,
        distance: edge.distance
      }))
    };
    
    // Save graph data to server
    fetch('/api/save-graph', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(graphData)
    }).catch(err => console.log('Graph save failed:', err));
    
    // Also log to console for debugging
    console.log('Graph exported:', graphData.nodes.length, 'nodes,', graphData.edges.length, 'edges');
    
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

    // Positionner les pins par défaut
    placeDefaultPins();
  }
}

// --- Algorithm toggle logic ---
const algoToggle = document.querySelector('.algorithm-toggle');
const algoOptions = document.querySelectorAll('.algorithm-option');
if (algoToggle && algoOptions.length) {
  algoOptions.forEach(opt => {
    opt.addEventListener('click', () => {
      algoOptions.forEach(o => o.classList.remove('active'));
      opt.classList.add('active');
      algoToggle.setAttribute('data-selected', opt.getAttribute('data-value'));
    });
  });
}

// --- Clean run algorithm function ---
function runAlgorithm() {
  const algorithm = document.querySelector('.algorithm-toggle').getAttribute('data-selected');
  const batteryCapacity = window.batteryCapacity;
  const maxPayload = window.maxPayload;
  
  if (!pickupNodeId || !deliveryNodeId) {
    alert('Please select both pickup and delivery points first!');
    return;
  }
  
  console.log(`Running ${algorithm}: ${pickupNodeId} -> ${deliveryNodeId}`);
  
  const runButton = document.getElementById('run-algo');
  runButton.disabled = true;
  runButton.innerHTML = 'Loading...';
  
  clearRouteVisualization();
  
  const endpoint = algorithm === 'ppo' ? '/api/run-ppo-inference' : '/api/run-algorithm';
  const requestBody = algorithm === 'ppo' ? {
    pickupNode: pickupNodeId,
    deliveryNode: deliveryNodeId,
    batteryCapacity,
    maxPayload
  } : {
    algorithm,
    batteryCapacity,
    maxPayload,
    startNode: pickupNodeId,
    endNode: deliveryNodeId
  };
  
  fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody)
  })
  .then(response => response.json())
  .then(data => {
    console.log('Algorithm response:', data);
    
    if (data.status === 'success') {
      const stats = data.stats || data.result || {};
      const routeIndices = data.route_indices || data.result?.route_indices || [];
      const routeNames = data.route_names || data.result?.route_names || [];
      const batteryHistory = data.battery_history || data.result?.battery_history || [];
      const actions = data.actions || data.result?.actions || [];
      const actionTypes = data.action_types || data.result?.action_types || [];
      
      // FIXED: Toujours afficher le chemin, succès ou échec
      if (routeIndices.length > 0) {
        const extraInfo = {
          algorithm: algorithm.toUpperCase(),
          failed: !stats.success,
          actions: actions,
          actionTypes: actionTypes,
          modelType: data.result?.model_type || 'PPO'
        };
        
        // Visualiser la route
        visualizeRoute(routeIndices, batteryHistory, extraInfo);
        
        // Message adapté selon le résultat
        if (stats.success) {
          const successMsg = `✅ Success!\nSteps: ${stats.steps || 'N/A'}\nBattery used: ${stats.batteryUsed || stats.battery_used || 'N/A'}%\nFinal battery: ${stats.battery_final || 'N/A'}%\nRoute: ${routeNames.join(' → ')}`;
          alert(successMsg);
        } else {
          const failMsg = `❌ Mission failed!\nReason: ${stats.termination_reason || 'Unknown'}\nSteps taken: ${stats.steps || 0}\nPickup done: ${stats.pickup_done ? 'Yes' : 'No'}\nDelivery done: ${stats.delivery_done ? 'Yes' : 'No'}\nPartial route: ${routeNames.join(' → ')}`;
          alert(failMsg);
        }
      } else {
        alert('No route data available to visualize');
      }
    } else {
      // Erreur - afficher message utile
      let errorMsg = `${algorithm.toUpperCase()} execution failed!\n${data.message || 'Unknown error'}`;
      
      if (algorithm === 'ppo' && data.help) {
        errorMsg += `\n\nSolution: ${data.help}`;
      }
      
      if (data.searched_paths) {
        console.log('Searched model paths:', data.searched_paths);
      }
      
      alert(errorMsg);
      console.error('Algorithm error:', data);
    }
  })
  .catch(error => {
    console.error('Error:', error);
    alert(`Algorithm execution failed!\n${error.message}`);
  })
  .finally(() => {
    runButton.disabled = false;
    runButton.innerHTML = 'Run';
  });
}

// Route visualization functions
let routeLayer = null;

function clearRouteVisualization() {
  if (routeLayer) {
    map.removeLayer(routeLayer);
    routeLayer = null;
  }
}

function visualizeRoute(routeIndices, batteryHistory = [], extraInfo = {}) {
  clearRouteVisualization();
  
  if (!allNodes || routeIndices.length < 1) {
    console.warn('Cannot visualize route: insufficient data');
    return;
  }
  
  routeLayer = L.layerGroup().addTo(map);
  
  // Couleurs selon succès/échec et algorithme
  const routeColor = extraInfo.failed 
    ? '#e74c3c'  // Rouge pour échec
    : (extraInfo.algorithm === 'PPO' ? '#3742fa' : '#2ecc71');  // Bleu PPO, vert autres
  
  const dashPattern = extraInfo.failed ? '10, 10' : (extraInfo.algorithm === 'PPO' ? '15, 5' : 'none');
  
  // Dessiner la route si plus d'un point
  if (routeIndices.length > 1) {
    const routeCoords = routeIndices.map(idx => {
      if (idx < allNodes.length) {
        return [allNodes[idx].lat, allNodes[idx].lng];
      }
      return null;
    }).filter(coord => coord !== null);
    
    if (routeCoords.length > 1) {
      const routeLine = L.polyline(routeCoords, {
        color: routeColor,
        weight: extraInfo.algorithm === 'PPO' ? 6 : 4,
        opacity: 0.8,
        dashArray: dashPattern
      }).addTo(routeLayer);
      
      // Flèches directionnelles
      const decorator = L.polylineDecorator(routeLine, {
        patterns: [{
          offset: 25,
          repeat: 50,
          symbol: L.Symbol.arrowHead({
            pixelSize: 12,
            polygon: true,
            pathOptions: { color: routeColor, fillOpacity: 0.8 }
          })
        }]
      }).addTo(routeLayer);
    }
  }
  
  // Marqueurs pour chaque étape
  routeIndices.forEach((nodeIdx, stepIdx) => {
    if (nodeIdx < allNodes.length) {
      const node = allNodes[nodeIdx];
      const battery = batteryHistory[stepIdx] || 100;
      
      // Couleur selon niveau de batterie
      let markerColor = '#27ae60'; // Vert
      if (battery < 20) markerColor = '#e74c3c'; // Rouge
      else if (battery < 50) markerColor = '#f39c12'; // Orange
      else if (battery < 80) markerColor = '#f1c40f'; // Jaune
      
      const stepMarker = L.circleMarker([node.lat, node.lng], {
        radius: stepIdx === 0 ? 12 : (stepIdx === routeIndices.length - 1 ? 10 : 8),
        fillColor: markerColor,
        color: stepIdx === 0 ? '#2c3e50' : '#fff',
        weight: stepIdx === 0 ? 3 : 2,
        fillOpacity: 0.9
      }).addTo(routeLayer);
      
      // Popup avec informations détaillées
      let popupContent = `
        <div style="min-width: 200px;">
          <h4 style="margin: 0 0 8px 0; color: ${routeColor};">
            ${stepIdx === 0 ? '🚁 START' : (stepIdx === routeIndices.length - 1 ? '🏁 END' : `Step ${stepIdx + 1}`)}
          </h4>
          <b>Node:</b> ${node.id}<br>
          <b>Type:</b> ${node.type}<br>
          <b>Battery:</b> ${battery.toFixed(1)}%<br>
      `;
      
      if (extraInfo.actions && extraInfo.actionTypes && stepIdx < extraInfo.actions.length) {
        const action = extraInfo.actions[stepIdx];
        const actionType = extraInfo.actionTypes[stepIdx];
        popupContent += `<b>Action:</b> ${action} (${actionType})<br>`;
      }
      
      if (extraInfo.algorithm) {
        popupContent += `<b>Algorithm:</b> ${extraInfo.algorithm}<br>`;
      }
      
      // Indicateurs spéciaux pour pickup/delivery
      if (node.type === 'pickup') {
        popupContent += `<span style="color: ${COLORS.pickup};">📦 Pickup Point</span><br>`;
      } else if (node.type === 'delivery') {
        popupContent += `<span style="color: ${COLORS.delivery};">🎯 Delivery Point</span><br>`;
      } else if (node.type === 'charging') {
        popupContent += `<span style="color: ${COLORS.charging};">⚡ Charging Station</span><br>`;
      } else if (node.type === 'hubs') {
        popupContent += `<span style="color: ${COLORS.hubs};">🏢 Hub</span><br>`;
      }
      
      popupContent += '</div>';
      stepMarker.bindPopup(popupContent);
      
      // Numéro d'étape sur le marqueur
      if (stepIdx > 0) {  // Pas de numéro sur le point de départ
        const stepLabel = L.divIcon({
          html: `<div style="color: white; font-weight: bold; font-size: 10px; text-align: center; line-height: 16px;">${stepIdx}</div>`,
          className: 'step-number-label',
          iconSize: [16, 16]
        });
        L.marker([node.lat, node.lng], { icon: stepLabel }).addTo(routeLayer);
      }
    }
  });
  
  // Badge informatif
  const badgeColor = extraInfo.failed ? '#e74c3c' : routeColor;
  const status = extraInfo.failed ? 'FAILED' : 'SUCCESS';
  
  const infoBadge = L.control({ position: 'topright' });
  infoBadge.onAdd = function() {
    const div = L.DomUtil.create('div', 'route-info-badge');
    div.innerHTML = `
      <div style="background: ${badgeColor}; color: white; padding: 8px 12px; border-radius: 20px; font-weight: bold; box-shadow: 0 2px 8px rgba(0,0,0,0.3); margin-bottom: 5px;">
        ${extraInfo.algorithm || 'Route'} - ${status}
      </div>
    `;
    return div;
  };
  infoBadge.addTo(map);
  
  // Supprimer le badge après 8 secondes
  setTimeout(() => {
    try {
      map.removeControl(infoBadge);
    } catch (e) {
      // Badge déjà supprimé
    }
  }, 8000);
  
  // Zoomer sur la route
  if (routeLayer.getLayers().length > 0) {
    const group = new L.featureGroup(routeLayer.getLayers());
    map.fitBounds(group.getBounds().pad(0.15));
  }
  
  console.log(`Route visualized: ${routeIndices.length} steps, ${extraInfo.algorithm || 'Algorithm'} ${status}`);
}

// Add CSS for route visualization
const style = document.createElement('style');
style.textContent = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  
  .step-number-label {
    background: none !important;
    border: none !important;
  }
`;
document.head.appendChild(style);

// --- Battery slider logic ---
const batterySlider = document.getElementById('battery-slider');
const batteryValue = document.getElementById('drone-battery-value');
const batteryFill = document.getElementById('battery-slider-fill');
if (batterySlider && batteryValue && batteryFill) {
  function updateBatteryUI(val) {
    batteryValue.textContent = val;
    window.batteryCapacity = parseInt(val, 10);
    const percent = (val - 10) / 90;
    batteryFill.style.width = (percent * 100) + '%';
    // Already styled in CSS for pro look
  }
  batterySlider.addEventListener('input', e => updateBatteryUI(e.target.value));
  updateBatteryUI(batterySlider.value);
}

// --- Package icons gradient fill (light to dark) ---
const packagesContainer = document.getElementById('packages-container');
const payloadValue = document.getElementById('drone-payload-value');
if (packagesContainer && payloadValue) {
  let maxPayload = 3;
  function updatePackagesUI(n) {
    Array.from(packagesContainer.children).forEach((el, idx) => {
      if (idx < n) el.classList.add('active');
      else el.classList.remove('active');
    });
    payloadValue.textContent = n;
    window.maxPayload = n;
  }
  Array.from(packagesContainer.children).forEach((el, idx) => {
    el.addEventListener('click', () => updatePackagesUI(idx + 1));
  });
  updatePackagesUI(maxPayload);
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

// --- Visibility toggles (only charging and edges) ---
document.getElementById('toggle-charging').addEventListener('change', function(e) {
  if (e.target.checked) {
    map.addLayer(chargingLayer);
  } else {
    map.removeLayer(chargingLayer);
  }
});
document.getElementById('toggle-edges').addEventListener('change', function(e) {
  if (e.target.checked) {
    map.addLayer(edgesLayer);
  } else {
    map.removeLayer(edgesLayer);
  }
});

// Map click handler
map.on('click', function(e) {
  // Reset highlighting when clicking on the map (not on a node)
  if (highlightedNodeId !== null) {
    highlightedNodeId = null;
    highlightNodeConnections(null);
  }
});

// --- Run button event listener ---
const runButton = document.getElementById('run-algo');
if (runButton) {
  runButton.addEventListener('click', runAlgorithm);
}

// === Ajout pour Draggable Map Pins ===

// Variables globales pour les marqueurs et ids sélectionnés
let pickupMarker = null;
let deliveryMarker = null;
let pickupNodeId = null;
let deliveryNodeId = null;

// Récupérer les labels (plus de boutons)
const pickupNodeLabel = document.getElementById('pickup-node-label');
const deliveryNodeLabel = document.getElementById('delivery-node-label');

// Fonction utilitaire pour trouver le nœud le plus proche d'une position
function findClosestNode(latlng, nodes) {
  let minDist = Infinity;
  let closestId = null;
  nodes.forEach(node => {
    const dist = map.distance(latlng, [node.lat, node.lng]);
    if (dist < minDist) {
      minDist = dist;
      closestId = node.id;
    }
  });
  return closestId;
}

// Positionne les pins sur des points aléatoires après la génération du réseau
function placeDefaultPins() {
  // Pickup
  const pickupNodes = allNodes.filter(n => n.type === 'pickup');
  if (pickupNodes.length > 0) {
    const randomPickup = pickupNodes[Math.floor(Math.random() * pickupNodes.length)];
    if (pickupMarker) map.removeLayer(pickupMarker);
    pickupMarker = L.marker([randomPickup.lat, randomPickup.lng], {
      draggable: true,
      icon: L.icon({
        iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-orange.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
      })
    }).addTo(map);
    pickupNodeId = randomPickup.id;
    pickupNodeLabel.textContent = pickupNodeId;
    pickupNodeLabel.style.color = COLORS.pickup;
    pickupMarker.on('dragend', function(e) {
      const pos = e.target.getLatLng();
      pickupNodeId = findClosestNode(pos, pickupNodes);
      pickupNodeLabel.textContent = pickupNodeId !== null ? pickupNodeId : '–';
      pickupNodeLabel.style.color = COLORS.pickup;
      const node = pickupNodes.find(n => n.id === pickupNodeId);
      if (node) {
        pickupMarker.setLatLng([node.lat, node.lng]);
      }
    });
    pickupMarker.fire('dragend');
  }

  // Delivery
  const deliveryNodes = allNodes.filter(n => n.type === 'delivery');
  if (deliveryNodes.length > 0) {
    const randomDelivery = deliveryNodes[Math.floor(Math.random() * deliveryNodes.length)];
    if (deliveryMarker) map.removeLayer(deliveryMarker);
    deliveryMarker = L.marker([randomDelivery.lat, randomDelivery.lng], {
      draggable: true,
      icon: L.icon({
        iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-green.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
      })
    }).addTo(map);
    deliveryNodeId = randomDelivery.id;
    deliveryNodeLabel.textContent = deliveryNodeId;
    deliveryNodeLabel.style.color = COLORS.delivery;
    deliveryMarker.on('dragend', function(e) {
      const pos = e.target.getLatLng();
      deliveryNodeId = findClosestNode(pos, deliveryNodes);
      deliveryNodeLabel.textContent = deliveryNodeId !== null ? deliveryNodeId : '–';
      deliveryNodeLabel.style.color = COLORS.delivery;
      const node = deliveryNodes.find(n => n.id === deliveryNodeId);
      if (node) {
        deliveryMarker.setLatLng([node.lat, node.lng]);
      }
    });
    deliveryMarker.fire('dragend');
  }
}

// Remplacer l'écouteur sur pickupLayer et deliveryLayer par le comportement suivant :
pickupLayer.on('click', (e) => {
  const pickupNodes = allNodes.filter(n => n.type === 'pickup');
  if (pickupMarker) map.removeLayer(pickupMarker);
  const latlng = e.latlng;
  pickupMarker = L.marker(latlng, {
    draggable: true,
    icon: L.icon({
      iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-orange.png',
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
    })
  }).addTo(map);
  pickupNodeId = findClosestNode(latlng, pickupNodes);
  pickupNodeLabel.textContent = pickupNodeId !== null ? pickupNodeId : '–';
  pickupNodeLabel.style.color = COLORS.pickup;
  pickupMarker.on('dragend', function(e) {
    const pos = e.target.getLatLng();
    pickupNodeId = findClosestNode(pos, pickupNodes);
    pickupNodeLabel.textContent = pickupNodeId !== null ? pickupNodeId : '–';
    pickupNodeLabel.style.color = COLORS.pickup;
    const node = pickupNodes.find(n => n.id === pickupNodeId);
    if (node) {
      pickupMarker.setLatLng([node.lat, node.lng]);
    }
  });
  pickupMarker.fire('dragend');
});

deliveryLayer.on('click', (e) => {
  const deliveryNodes = allNodes.filter(n => n.type === 'delivery');
  if (deliveryMarker) map.removeLayer(deliveryMarker);
  const latlng = e.latlng;
  deliveryMarker = L.marker(latlng, {
    draggable: true,
    icon: L.icon({
      iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-green.png',
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
    })
  }).addTo(map);
  deliveryNodeId = findClosestNode(latlng, deliveryNodes);
  deliveryNodeLabel.textContent = deliveryNodeId !== null ? deliveryNodeId : '–';
  deliveryNodeLabel.style.color = COLORS.delivery;
  deliveryMarker.on('dragend', function(e) {
    const pos = e.target.getLatLng();
    deliveryNodeId = findClosestNode(pos, deliveryNodes);
    deliveryNodeLabel.textContent = deliveryNodeId !== null ? deliveryNodeId : '–';
    deliveryNodeLabel.style.color = COLORS.delivery;
    const node = deliveryNodes.find(n => n.id === deliveryNodeId);
    if (node) {
      deliveryMarker.setLatLng([node.lat, node.lng]);
    }
  });
  deliveryMarker.fire('dragend');
});
