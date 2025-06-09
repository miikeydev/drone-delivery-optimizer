// Main application file - simplified and modular
import { pickCityPoints } from '/js/city-picker.js';
import { haversineDistance, RNG, insideFrance, setFrancePolygon } from './utils.js';
import { buildGraph, annotateEdges, drawGraph, highlightNodeConnections } from './graph-builder.js';
import { visualizeRoute, clearRouteVisualization } from './route-visualization.js';
import GeneticAlgorithm from './GA.js';

// Map and layers setup
const map = L.map('map').setView([46.6, 2.3], 6);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

const hubsLayer = L.layerGroup().addTo(map);
const chargingLayer = L.layerGroup();
const deliveryLayer = L.layerGroup().addTo(map);
const pickupLayer = L.layerGroup().addTo(map);
const edgesLayer = L.layerGroup();

map.removeLayer(chargingLayer);
map.removeLayer(edgesLayer);

const COLORS = {
  hubs: '#e74c3c',
  charging: '#2980b9',
  delivery: '#27ae60',
  pickup: '#f39c12'
};

// Global state
let allNodes = [];
let allEdges = [];
let windAngle = RNG() * 2 * Math.PI;
let highlightedNodeId = null;
let francePoly;
let pickupMarker = null;
let deliveryMarker = null;
let pickupNodeId = null;
let deliveryNodeId = null;

window.batteryCapacity = 100;
window.maxPayload = 3;

// Initialize map with France boundaries
fetch('/data/metropole-version-simplifiee.geojson')
  .then(response => response.json())
  .then(geojson => {
    francePoly = geojson;
    setFrancePolygon(francePoly);
    
    L.geoJSON(francePoly, { 
      weight: 2, 
      color: '#444', 
      fill: false 
    }).addTo(map);
    
    map.fitBounds(L.geoJSON(francePoly).getBounds());
    generateNetwork();
  })
  .catch(err => {
    console.error('Error loading France boundaries:', err);
    // Fallback mode
    francePoly = {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [[
          [-5.2, 41.3], [9.5, 41.3], [9.5, 51.1], [-5.2, 51.1], [-5.2, 41.3]
        ]]
      }
    };
    
    setFrancePolygon(francePoly);
    L.rectangle([[41.3, -5.2], [51.1, 9.5]], {color: "#444", weight: 1, fill: false}).addTo(map);
    generateNetwork();
  });

async function generateNetwork() {
  if (!francePoly) return;
  
  // Clear existing layers
  [hubsLayer, chargingLayer, deliveryLayer, pickupLayer, edgesLayer].forEach(layer => layer.clearLayers());
  
  // Get points from city picker
  const { hubs: hubPoints, charging: chargingPoints, delivery: deliveryPoints, pickup: pickupPoints } = await pickCityPoints();
  
  allNodes = [];
  
  // Create nodes for each type
  const nodeTypes = [
    { points: hubPoints, type: 'hubs', layer: hubsLayer, color: COLORS.hubs, radius: 8, prefix: 'Hub' },
    { points: chargingPoints, type: 'charging', layer: chargingLayer, color: COLORS.charging, radius: 6, prefix: 'Charging' },
    { points: deliveryPoints, type: 'delivery', layer: deliveryLayer, color: COLORS.delivery, radius: 5, prefix: 'Delivery' },
    { points: pickupPoints, type: 'pickup', layer: pickupLayer, color: COLORS.pickup, radius: 5, prefix: 'Pickup' }
  ];

  nodeTypes.forEach(({ points, type, layer, color, radius, prefix }) => {
    points.forEach((latlng, index) => {
      if (!insideFrance(latlng[0], latlng[1])) return;
      
      const id = `${prefix} ${index + 1}`;
      const marker = L.circleMarker(latlng, {
        color: color,
        fillColor: color,
        fillOpacity: 0.8,
        weight: type === 'hubs' ? 2 : 1,
        radius: radius,
        zIndex: 1000
      }).addTo(layer);
      
      marker.bindTooltip(id);
      
      marker.on('click', function() {
        const nodeIndex = allNodes.findIndex(n => n.id === id);
        highlightedNodeId = highlightedNodeId === nodeIndex ? null : nodeIndex;
        highlightNodeConnections(highlightedNodeId, edgesLayer, allEdges);
      });
      
      allNodes.push({
        id: id,
        index: allNodes.length,
        lat: latlng[0],
        lng: latlng[1],
        type: type
      });
    });
  });
  
  // Build and draw graph
  if (allNodes.length > 0) {
    const rawEdges = buildGraph(allNodes, 10);
    allEdges = annotateEdges(rawEdges, allNodes, windAngle);
    
    // Export graph data
    const graphData = {
      nodes: allNodes.map((node, index) => ({
        id: node.id,
        type: node.type,
        lat: node.lat,
        lng: node.lng,
        index: index
      })),
      edges: allEdges.map(edge => ({
        u: edge.source,
        v: edge.target,
        dist: edge.distance,
        cost: edge.cost,
        source: edge.source,
        target: edge.target,
        distance: edge.distance
      }))
    };
    
    fetch('/api/save-graph', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(graphData)
    }).catch(err => console.log('Graph save failed:', err));
    
    drawGraph(allNodes, allEdges, edgesLayer);
    
    // Update wind arrow
    const windArrow = document.getElementById('wind-arrow');
    if (windArrow) {
      const degrees = Math.round((windAngle * 180 / Math.PI) % 360);
      windArrow.style.transform = `translate(-50%, -50%) rotate(${degrees}deg)`;
    }

    placeDefaultPins();
  }
}

function placeDefaultPins() {
  const pickupNodes = allNodes.filter(n => n.type === 'pickup');
  const deliveryNodes = allNodes.filter(n => n.type === 'delivery');
  
  if (pickupNodes.length > 0) {
    const randomIndex = Math.floor(Math.random() * pickupNodes.length);
    const defaultPickup = pickupNodes[randomIndex];
    
    if (pickupMarker) map.removeLayer(pickupMarker);
    pickupMarker = L.marker([defaultPickup.lat, defaultPickup.lng], {
      draggable: true,
      icon: L.icon({
        iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-orange.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
      })
    }).addTo(map);

    pickupNodeId = defaultPickup.id;
    updatePickupDisplay(defaultPickup.id);

    pickupMarker.on('dragend', function(e) {
      const pos = e.target.getLatLng();
      const newPickupNode = findClosestNode(pos, pickupNodes);
      if (newPickupNode) {
        pickupNodeId = newPickupNode.id;
        updatePickupDisplay(newPickupNode.id);
        pickupMarker.setLatLng([newPickupNode.lat, newPickupNode.lng]);
      }
    });
  }

  if (deliveryNodes.length > 0) {
    let defaultDelivery;
    
    if (pickupNodes.length > 0 && deliveryNodes.length > 1) {
      const selectedPickup = pickupNodes.find(p => p.id === pickupNodeId);
      if (selectedPickup) {
        const distantDeliveries = deliveryNodes.filter(d => {
          const distance = haversineDistance(selectedPickup.lat, selectedPickup.lng, d.lat, d.lng);
          return distance > 50;
        });
        
        if (distantDeliveries.length > 0) {
          const randomIndex = Math.floor(Math.random() * distantDeliveries.length);
          defaultDelivery = distantDeliveries[randomIndex];
        } else {
          const randomIndex = Math.floor(Math.random() * deliveryNodes.length);
          defaultDelivery = deliveryNodes[randomIndex];
        }
      }
    } else {
      const randomIndex = Math.floor(Math.random() * deliveryNodes.length);
      defaultDelivery = deliveryNodes[randomIndex];
    }
    
    if (deliveryMarker) map.removeLayer(deliveryMarker);
    deliveryMarker = L.marker([defaultDelivery.lat, defaultDelivery.lng], {
      draggable: true,
      icon: L.icon({
        iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-green.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
      })
    }).addTo(map);

    deliveryNodeId = defaultDelivery.id;
    updateDeliveryDisplay(defaultDelivery.id);

    deliveryMarker.on('dragend', function(e) {
      const pos = e.target.getLatLng();
      const newDeliveryNode = findClosestNode(pos, deliveryNodes);
      if (newDeliveryNode) {
        deliveryNodeId = newDeliveryNode.id;
        updateDeliveryDisplay(newDeliveryNode.id);
        deliveryMarker.setLatLng([newDeliveryNode.lat, newDeliveryNode.lng]);
      }
    });
  }
}

function findClosestNode(latlng, nodes) {
  let minDist = Infinity;
  let closestNode = null;
  nodes.forEach(node => {
    const dist = map.distance(latlng, [node.lat, node.lng]);
    if (dist < minDist) {
      minDist = dist;
      closestNode = node;
    }
  });
  return closestNode;
}

function updatePickupDisplay(text) {
  const badge = document.querySelector('.point-badge.pickup .point-badge-text');
  if (badge) badge.textContent = text;
}

function updateDeliveryDisplay(text) {
  const badge = document.querySelector('.point-badge.delivery .point-badge-text');
  if (badge) badge.textContent = text;
}

// Algorithm execution
function runAlgorithm() {
  const algorithm = document.querySelector('.algorithm-toggle').getAttribute('data-selected');
  const batteryCapacity = window.batteryCapacity;
  const maxPayload = window.maxPayload;
  
  if (!pickupNodeId || !deliveryNodeId) {
    alert('Please select both pickup and delivery points first!');
    return;
  }
  
  const runButton = document.getElementById('run-algo');
  runButton.disabled = true;
  runButton.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="3" fill="currentColor"><animate attributeName="r" values="3;6;3" dur="1s" repeatCount="indefinite"/></circle></svg>';
  
  clearRouteVisualization(map);
  
  if (algorithm === 'ga') {
    try {
      const pickupNode = allNodes.find(n => n.id === pickupNodeId);
      const deliveryNode = allNodes.find(n => n.id === deliveryNodeId);
      
      if (!pickupNode || !deliveryNode) {
        throw new Error(`Could not find nodes: pickup=${pickupNodeId}, delivery=${deliveryNodeId}`);
      }
      
      const packages = [{
        pickup: pickupNode.index,
        delivery: deliveryNode.index,
        weight: 1
      }];
      
      const options = {
        batteryCapacity: batteryCapacity,
        maxPayload: maxPayload,
        populationSize: 30,
        generations: 30,
        crossoverRate: 0.8,
        mutationRate: 0.3
      };
      
      const ga = new GeneticAlgorithm(allNodes, allEdges, packages, options);
      const result = ga.run();
      
      if (result.success && result.route_indices.length > 0) {
        const extraInfo = {
          algorithm: 'GA',
          failed: !result.success,
          actions: result.route_names,
          actionTypes: result.route_names.map(() => 'move'),
          modelType: 'Genetic Algorithm'
        };
        
        visualizeRoute(result.route_indices, result.battery_history, extraInfo, allNodes, map);
        
        const successMsg = `✅ GA Success!\nFitness: ${result.fitness.toFixed(2)}\nSteps: ${result.stats.steps}\nBattery used: ${result.stats.battery_used.toFixed(1)}%\nFinal battery: ${result.stats.battery_final.toFixed(1)}%\nRoute: ${result.route_names.join(' → ')}`;
        alert(successMsg);
      } else {
        const failMsg = `❌ GA Failed!\nNo valid route found\nBest fitness: ${result.fitness.toFixed(2)}\nTry adjusting parameters or selecting different points.`;
        alert(failMsg);
        
        if (result.route_indices.length > 0) {
          const extraInfo = {
            algorithm: 'GA',
            failed: true,
            actions: result.route_names,
            actionTypes: result.route_names.map(() => 'move'),
            modelType: 'Genetic Algorithm'
          };
          visualizeRoute(result.route_indices, result.battery_history, extraInfo, allNodes, map);
        }
      }
    } catch (error) {
      console.error('[GA] Error:', error);
      alert(`GA execution failed!\n${error.message}`);
    }
    
    runButton.disabled = false;
    runButton.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><polygon points="8,5 19,12 8,19" fill="currentColor"/></svg>';
    return;
  }
  
  // Handle PPO and other algorithms via server
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
    if (data.status === 'success') {
      const stats = data.stats || data.result || {};
      const routeIndices = data.route_indices || data.result?.route_indices || [];
      const routeNames = data.route_names || data.result?.route_names || [];
      const batteryHistory = data.battery_history || data.result?.battery_history || [];
      const actions = data.actions || data.result?.actions || [];
      const actionTypes = data.action_types || data.result?.action_types || [];
      
      if (routeIndices.length > 0) {
        const extraInfo = {
          algorithm: algorithm.toUpperCase(),
          failed: !stats.success,
          actions: actions,
          actionTypes: actionTypes,
          modelType: data.result?.model_type || 'PPO'
        };
        
        visualizeRoute(routeIndices, batteryHistory, extraInfo, allNodes, map);
        
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
      let errorMsg = `${algorithm.toUpperCase()} execution failed!\n${data.message || 'Unknown error'}`;
      if (algorithm === 'ppo' && data.help) {
        errorMsg += `\n\nSolution: ${data.help}`;
      }
      alert(errorMsg);
    }
  })
  .catch(error => {
    alert(`Algorithm execution failed!\n${error.message}`);
  })
  .finally(() => {
    runButton.disabled = false;
    runButton.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><polygon points="8,5 19,12 8,19" fill="currentColor"/></svg>';
  });
}

// UI Event Handlers
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

// Battery slider
const batterySlider = document.getElementById('battery-slider');
const batteryValue = document.getElementById('drone-battery-value');
const batteryFill = document.getElementById('battery-slider-fill');
if (batterySlider && batteryValue && batteryFill) {
  function updateBatteryUI(val) {
    batteryValue.textContent = val;
    window.batteryCapacity = parseInt(val, 10);
    const percent = (val - 10) / 90;
    batteryFill.style.width = (percent * 100) + '%';
  }
  batterySlider.addEventListener('input', e => updateBatteryUI(e.target.value));
  updateBatteryUI(batterySlider.value);
}

// Package selection
const packagesContainer = document.getElementById('packages-container');
const payloadValue = document.getElementById('drone-payload-value');
if (packagesContainer && payloadValue) {
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
  updatePackagesUI(3);
}

// Wind compass
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
  if (allNodes.length > 0 && allEdges.length > 0) {
    allEdges = annotateEdges(
      allEdges.map(e => ({ source: e.source, target: e.target, distance: e.distance })),
      allNodes,
      windAngle
    );
    drawGraph(allNodes, allEdges, edgesLayer);
  }
}

let draggingWind = false;
function getAngleFromEvent(e) {
  const rect = windCompass.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - cx;
  const y = (e.touches ? e.touches[0].clientY : e.clientY) - cy;
  let angle = Math.atan2(x, -y) * 180 / Math.PI;
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

// Visibility toggles
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

// Map interactions
map.on('click', function(e) {
  if (highlightedNodeId !== null) {
    highlightedNodeId = null;
    highlightNodeConnections(null, edgesLayer, allEdges);
  }
});

// Run button event listener
const runButton = document.getElementById('run-algo');
if (runButton) {
  runButton.addEventListener('click', runAlgorithm);
}

// Layer click handlers for pin placement
pickupLayer.on('click', (e) => {
  const pickupNodes = allNodes.filter(n => n.type === 'pickup');
  if (pickupMarker) map.removeLayer(pickupMarker);
  
  pickupMarker = L.marker(e.latlng, {
    draggable: true,
    icon: L.icon({
      iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-orange.png',
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
    })
  }).addTo(map);
  
  const pickupNode = findClosestNode(e.latlng, pickupNodes);
  pickupNodeId = pickupNode ? pickupNode.id : null;
  updatePickupDisplay(pickupNodeId || '–');
  
  pickupMarker.on('dragend', function(e) {
    const pos = e.target.getLatLng();
    const pickupNode = findClosestNode(pos, pickupNodes);
    pickupNodeId = pickupNode ? pickupNode.id : null;
    updatePickupDisplay(pickupNodeId || '–');
    const node = pickupNodes.find(n => n.id === pickupNodeId);
    if (node) {
      pickupMarker.setLatLng([node.lat, node.lng]);
    }
  });
});

deliveryLayer.on('click', (e) => {
  const deliveryNodes = allNodes.filter(n => n.type === 'delivery');
  if (deliveryMarker) map.removeLayer(deliveryMarker);
  
  deliveryMarker = L.marker(e.latlng, {
    draggable: true,
    icon: L.icon({
      iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-green.png',
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
    })
  }).addTo(map);
  
  const deliveryNode = findClosestNode(e.latlng, deliveryNodes);
  deliveryNodeId = deliveryNode ? deliveryNode.id : null;
  updateDeliveryDisplay(deliveryNodeId || '–');
  
  deliveryMarker.on('dragend', function(e) {
    const pos = e.target.getLatLng();
    const deliveryNode = findClosestNode(pos, deliveryNodes);
    deliveryNodeId = deliveryNode ? deliveryNode.id : null;
    updateDeliveryDisplay(deliveryNodeId || '–');
    const node = deliveryNodes.find(n => n.id === deliveryNodeId);
    if (node) {
      deliveryMarker.setLatLng([node.lat, node.lng]);
    }
  });
});
