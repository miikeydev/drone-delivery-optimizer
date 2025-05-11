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
const edgesLayer = L.layerGroup().addTo(map); // New layer for graph edges

// Define colors for each category
const COLORS = {
  hubs: '#e74c3c',      // Red
  charging: '#2980b9',  // Blue
  delivery: '#27ae60',  // Green
  pickup: '#f39c12'     // Orange
};

// Global variables for graph representation
let allNodes = [];
let allEdges = [];
let windAngle = Math.random() * 2 * Math.PI; // Random wind angle in radians
let highlightedNodeId = null;

// Variable pour stocker les frontières de la France
let francePoly;

// Charger le GeoJSON local de la France métropolitaine
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
    
    // Afficher le contour sur la carte
    L.geoJSON(francePoly, { 
      weight: 2, 
      color: '#444', 
      fill: false 
    }).addTo(map);
    
    // Ajuster la vue de la carte aux frontières de la France
    map.fitBounds(L.geoJSON(francePoly).getBounds());
    
    // Une fois la frontière prête, générer les points
    generateAll();
  })
  .catch(err => {
    console.error('Erreur de chargement des frontières:', err);
    alert('Problème avec le chargement des frontières françaises. Utilisation du mode dégradé.');
    
    // Mode dégradé: utiliser un rectangle approximatif pour la France
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
    generateAll();
  });

/**
 * Génère count points Poisson-disc, mais ne conserve
 * que ceux qui sont DANS la France.
 */
function generatePoissonPoints(count, minDist = 0.2) {
  if (!francePoly) return [];
  
  const bounds = L.geoJSON(francePoly).getBounds();
  const pts = [];

  while (pts.length < count) {
    const lat = Math.random() * (bounds.getNorth() - bounds.getSouth()) + bounds.getSouth();
    const lng = Math.random() * (bounds.getEast() - bounds.getWest()) + bounds.getWest();
    const pt = turf.point([lng, lat]);

    // Vérifier si le point est dans la France
    try {
      if (!turf.booleanPointInPolygon(pt, francePoly)) continue;
    } catch (e) {
      console.warn("Erreur lors du test point-in-polygon:", e);
      // En cas d'erreur, on accepte le point et continue
      pts.push([lat, lng]);
      continue;
    }

    // Vérifier la distance minimale avec les points existants
    let tooClose = false;
    for (const p of pts) {
      const distance = turf.distance(pt, turf.point([p[1], p[0]]));
      if (distance < minDist) {
        tooClose = true;
        break;
      }
    }
    
    if (!tooClose) {
      pts.push([lat, lng]);
    }
  }

  return pts;
}

/**
 * Calculate haversine distance between two points in km
 */
function haversineDistance(lat1, lon1, lat2, lon2) {
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
function gaussianRandom(mean = 0, stdev = 1) {
  const u = 1 - Math.random(); // Converting [0,1) to (0,1]
  const v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stdev + mean;
}

/**
 * Build k-nearest neighbors graph from nodes
 */
function buildGraph(nodes, k = 4) {
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

/**
 * Annotate edges with wind effects and costs
 */
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
    
    // Calculate final cost - new formula where tailwind reduces cost
    const cost = edge.distance * (1 - alpha * windEffect + beta * noise);
    
    return {
      ...edge,
      windEffect,
      phi,
      cost
    };
  });
}

/**
 * Draw the graph on the map
 */
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
    
    // Edge thickness inversely proportional to cost (thicker = better route)
    const weight = 1 + 3 * (1 - (edge.cost - minCost) / (maxCost - minCost));
    
    const polyline = L.polyline(
      [[source.lat, source.lng], [target.lat, target.lng]], 
      {
        color: color,
        weight: weight,
        opacity: 0.3, // Reduced from 0.6 to 0.3
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
            pixelSize: 6, // Reduced from 10 to 6
            polygon: true,
            pathOptions: { color: color, fillOpacity: 0.5 } // Reduced opacity
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

/**
 * Convert RGB to Hex color code
 */
function rgbToHex(r, g, b) {
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

/**
 * Highlight edges connected to a node
 */
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
        const weight = 1 + 3 * (1 - (edge.cost - minCost) / (maxCost - minCost));
        
        layer.setStyle({
          weight: weight,
          opacity: nodeIndex === layer.sourceIndex ? 1 : 0.6
        });
      }
    }
  });
  
  // If a node is selected, highlight its outgoing edges
  if (nodeIndex !== null) {
    edgesLayer.eachLayer(layer => {
      if (layer instanceof L.Polyline && layer.sourceIndex === nodeIndex) {
        layer.setStyle({
          weight: 4,
          opacity: 1
        });
        layer.bringToFront();
      }
    });
  }
}

/**
 * Generate and display all points on the map
 */
function generateAll() {
  if (!francePoly) {
    alert("Les frontières de la France ne sont pas encore chargées. Veuillez patienter.");
    return;
  }
  
  // Generate a new random wind angle
  windAngle = Math.random() * 2 * Math.PI;
  document.getElementById('wind-direction').innerHTML = `${(windAngle * 180 / Math.PI).toFixed(0)}°`;
  
  // Clear all existing layers
  hubsLayer.clearLayers();
  chargingLayer.clearLayers();
  deliveryLayer.clearLayers();
  pickupLayer.clearLayers();
  edgesLayer.clearLayers();
  
  // Get counts from inputs
  const hubCount = +document.getElementById('count-hubs').value;
  const chargingCount = +document.getElementById('count-charging').value;
  const deliveryCount = +document.getElementById('count-delivery').value;
  const pickupCount = +document.getElementById('count-pickup').value;
  
  // Get k-NN value
  const kNeighbors = +document.getElementById('k-neighbors').value;
  
  // Generate points with different minimum distances based on importance
  const hubPoints = generatePoissonPoints(hubCount, 0.8);
  const chargingPoints = generatePoissonPoints(chargingCount, 0.5);
  const deliveryPoints = generatePoissonPoints(deliveryCount, 0.3);
  const pickupPoints = generatePoissonPoints(pickupCount, 0.3);
  
  // Reset nodes array
  allNodes = [];
  
  // Add points to their respective layers and to the nodes array
  hubPoints.forEach((latlng, index) => {
    const id = `Hub ${index + 1}`;
    const marker = L.circleMarker(latlng, {
      color: COLORS.hubs,
      fillColor: COLORS.hubs,
      fillOpacity: 0.8,
      weight: 2,
      radius: 8
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
  
  chargingPoints.forEach((latlng, index) => {
    const id = `Charging ${index + 1}`;
    const marker = L.circleMarker(latlng, {
      color: COLORS.charging,
      fillColor: COLORS.charging,
      fillOpacity: 0.8,
      weight: 1,
      radius: 6
    }).addTo(chargingLayer);
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
      radius: 5
    }).addTo(deliveryLayer);
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
      radius: 5
    }).addTo(pickupLayer);
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
      type: 'pickup'
    });
  });
  
  // Build graph if we have at least one node
  if (allNodes.length > 0) {
    // Get parameters for edge creation
    const alpha = +document.getElementById('wind-factor').value;
    const beta = +document.getElementById('noise-factor').value;
    
    // Build the k-nearest neighbors graph
    const rawEdges = buildGraph(allNodes, kNeighbors);
    
    // Add wind and cost effects
    allEdges = annotateEdges(rawEdges, allNodes, windAngle, alpha, beta);
    
    // Draw the graph
    drawGraph(allNodes, allEdges);
    
    // Update the wind arrow on the compass
    updateWindCompass(windAngle);
  }
}

/**
 * Update the wind direction compass
 */
function updateWindCompass(angle) {
  const arrowEl = document.getElementById('wind-arrow');
  if (arrowEl) {
    arrowEl.style.transform = `rotate(${angle * 180 / Math.PI}deg)`;
  }
}

// Set up the legend
const legend = L.control({position: 'bottomright'});
legend.onAdd = function() {
  const div = L.DomUtil.create('div', 'legend');
  div.innerHTML = '<h4 class="legend-title">Point Types</h4>';
  
  const types = {
    'hubs': 'Hubs',
    'charging': 'Charging Stations',
    'delivery': 'Delivery Points',
    'pickup': 'Pickup Points'
  };
  
  for (const [key, label] of Object.entries(types)) {
    div.innerHTML += `
      <div class="legend-item">
        <span class="legend-color" style="background: ${COLORS[key]}"></span>
        ${label}
      </div>
    `;
  }
  
  // Add edge cost legend
  div.innerHTML += '<h4 class="legend-title" style="margin-top:10px;">Edge Cost</h4>';
  div.innerHTML += `
    <div style="display:flex;margin-bottom:5px;">
      <span style="flex:1;height:5px;background:linear-gradient(to right, #2ecc71, #f1c40f, #e74c3c);"></span>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:11px;">
      <span>Low Cost</span>
      <span>High Cost</span>
    </div>
  `;
  
  return div;
};
legend.addTo(map);

// Add wind compass
const windCompass = L.control({position: 'bottomleft'});
windCompass.onAdd = function() {
  const div = L.DomUtil.create('div', 'wind-compass');
  div.innerHTML = `
    <div style="background:white;padding:10px;border-radius:5px;box-shadow:0 0 5px rgba(0,0,0,0.2);">
      <h4 style="margin:0 0 5px 0;font-size:14px;">Wind Direction: <span id="wind-direction">0°</span></h4>
      <div style="position:relative;width:50px;height:50px;border:2px solid #ccc;border-radius:50%;margin:0 auto;">
        <div id="wind-arrow" style="
          position:absolute;
          top:50%;
          left:50%;
          width:0;
          height:0;
          transform:translate(-50%, -50%) rotate(0deg);
          border-left:5px solid transparent;
          border-right:5px solid transparent;
          border-bottom:20px solid #3498db;
        "></div>
      </div>
      <div style="text-align:center;font-size:10px;margin-top:5px;">
        Wind flowing toward arrow direction
      </div>
    </div>
  `;
  return div;
};
windCompass.addTo(map);

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

// Set up the generate button
document.getElementById('generate').addEventListener('click', generateAll);

// Listen for edge display toggle
document.getElementById('toggle-edges').addEventListener('change', function(e) {
  if (e.target.checked) {
    map.addLayer(edgesLayer);
  } else {
    map.removeLayer(edgesLayer);
  }
});

// Optional: Add a click handler to show routes between points
let selectedPoint = null;

map.on('click', function(e) {
  // Reset highlighting when clicking on the map (not on a node)
  if (highlightedNodeId !== null) {
    highlightedNodeId = null;
    highlightNodeConnections(null);
  }
});