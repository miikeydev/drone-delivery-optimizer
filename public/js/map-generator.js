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

// Define colors for each category
const COLORS = {
  hubs: '#e74c3c',      // Red
  charging: '#2980b9',  // Blue
  delivery: '#27ae60',  // Green
  pickup: '#f39c12'     // Orange
};

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
 * Generate and display all points on the map
 */
function generateAll() {
  if (!francePoly) {
    alert("Les frontières de la France ne sont pas encore chargées. Veuillez patienter.");
    return;
  }
  
  // Clear all existing layers
  hubsLayer.clearLayers();
  chargingLayer.clearLayers();
  deliveryLayer.clearLayers();
  pickupLayer.clearLayers();
  
  // Get counts from inputs
  const hubCount = +document.getElementById('count-hubs').value;
  const chargingCount = +document.getElementById('count-charging').value;
  const deliveryCount = +document.getElementById('count-delivery').value;
  const pickupCount = +document.getElementById('count-pickup').value;
  
  // Generate points with different minimum distances based on importance
  const hubPoints = generatePoissonPoints(hubCount, 0.8);
  const chargingPoints = generatePoissonPoints(chargingCount, 0.5);
  const deliveryPoints = generatePoissonPoints(deliveryCount, 0.3);
  const pickupPoints = generatePoissonPoints(pickupCount, 0.3);
  
  // Add points to their respective layers
  hubPoints.forEach((latlng, index) => {
    const marker = L.circleMarker(latlng, {
      color: COLORS.hubs,
      fillColor: COLORS.hubs,
      fillOpacity: 0.8,
      weight: 2,
      radius: 8
    }).addTo(hubsLayer);
    marker.bindTooltip(`Hub ${index + 1}`);
  });
  
  chargingPoints.forEach((latlng, index) => {
    const marker = L.circleMarker(latlng, {
      color: COLORS.charging,
      fillColor: COLORS.charging,
      fillOpacity: 0.8,
      weight: 1,
      radius: 6
    }).addTo(chargingLayer);
    marker.bindTooltip(`Charging Station ${index + 1}`);
  });
  
  deliveryPoints.forEach((latlng, index) => {
    const marker = L.circleMarker(latlng, {
      color: COLORS.delivery,
      fillColor: COLORS.delivery,
      fillOpacity: 0.8,
      weight: 1,
      radius: 5
    }).addTo(deliveryLayer);
    marker.bindTooltip(`Delivery Point ${index + 1}`);
  });
  
  pickupPoints.forEach((latlng, index) => {
    const marker = L.circleMarker(latlng, {
      color: COLORS.pickup,
      fillColor: COLORS.pickup,
      fillOpacity: 0.8,
      weight: 1,
      radius: 5
    }).addTo(pickupLayer);
    marker.bindTooltip(`Pickup Point ${index + 1}`);
  });
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
  
  return div;
};
legend.addTo(map);

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

// Optional: Add a click handler to show routes between points
let selectedPoint = null;

map.on('click', function(e) {
  // Here you could implement route finding logic between selected points
  console.log('Map clicked at:', e.latlng);
});