// Map initialization code
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

// Default parameters for the network
const DEFAULT_PARAMS = {
  hubCount: 50,
  chargingCount: 100,
  deliveryCount: 20,
  pickupCount: 20,
  kNeighbors: 10
};

// Global variables for graph representation
let allNodes = [];
let allEdges = [];
let windAngle = Math.random() * 2 * Math.PI; // Random wind angle in radians
let highlightedNodeId = null;
let francePoly;

// Global variables for drone settings, initialized with default values
window.batteryCapacity = 30;
window.maxPayload = 3;

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
    L.rectangle([[41.3, -5.2], [51.1, 9.5]], {color: "#444", weight: 1, fill: false}).addTo(map);
    generateNetwork();
  });

/**
 * Generate the delivery network
 */
function generateNetwork() {
  if (!francePoly) {
    alert("Les frontières de la France ne sont pas encore chargées. Veuillez patienter.");
    return;
  }
  
  // Use the manually set wind direction if available
  const windDirectionInput = document.getElementById('wind-direction-input');
  if (windDirectionInput) {
    // Get the angle in degrees from the input
    const degrees = parseInt(windDirectionInput.value) || 0;
    // Convert to radians
    windAngle = (degrees * Math.PI / 180);
  }
  
  // Clear all existing layers
  hubsLayer.clearLayers();
  chargingLayer.clearLayers();
  deliveryLayer.clearLayers();
  pickupLayer.clearLayers();
  edgesLayer.clearLayers();
  
  // Generate points with different minimum distances based on importance
  const hubPoints = generatePoissonPoints(DEFAULT_PARAMS.hubCount, 0.8);
  const chargingPoints = generatePoissonPoints(DEFAULT_PARAMS.chargingCount, 0.5);
  const deliveryPoints = generatePoissonPoints(DEFAULT_PARAMS.deliveryCount, 0.3);
  const pickupPoints = generatePoissonPoints(DEFAULT_PARAMS.pickupCount, 0.3);
  
  // Reset nodes array
  allNodes = [];
  
  // Add points to their respective layers with higher z-index
  hubPoints.forEach((latlng, index) => {
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
    const rawEdges = buildGraph(allNodes, DEFAULT_PARAMS.kNeighbors);
    
    // Add wind and cost effects
    allEdges = annotateEdges(rawEdges, allNodes, windAngle, alpha, beta);
    
    // Draw the graph
    drawGraph(allNodes, allEdges);
    
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
  
  // This is just a placeholder - the real implementation would be added later
  alert(`Running ${algorithm.toUpperCase()} algorithm with ${batteryCapacity} battery units and ${maxPayload} max payload capacity.`);
}

// Event listener for the wind direction input
document.getElementById('wind-direction-input').addEventListener('change', function() {
  const degrees = parseInt(this.value) || 0;
  // Normalize degrees to 0-359
  const normalizedDegrees = ((degrees % 360) + 360) % 360;
  this.value = normalizedDegrees;
  
  // Convert to radians
  windAngle = (normalizedDegrees * Math.PI / 180);
  
  // Update wind arrow display
  const windArrow = document.getElementById('wind-arrow');
  if (windArrow) {
    windArrow.style.transform = `translate(-50%, -50%) rotate(${normalizedDegrees}deg)`;
  }
  
  // If we have edges, recalculate costs with new wind direction and redraw
  if (allNodes.length > 0 && allEdges.length > 0) {
    const alpha = 0.3; // Default value 
    const beta = 0.05; // Default value
    
    // Recalculate edges with new wind angle
    allEdges = annotateEdges(
      allEdges.map(e => ({ source: e.source, target: e.target, distance: e.distance })),
      allNodes, 
      windAngle, 
      alpha, 
      beta
    );
    
    // Redraw the graph
    drawGraph(allNodes, allEdges);
  }
});

// Set up event listeners for the checkbox filters
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

// Generate the network on page load
document.addEventListener('DOMContentLoaded', function() {
  // Nothing to do here - network is generated after France GeoJSON is loaded
});
