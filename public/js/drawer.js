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

/**
 * Update the wind direction compass
 */
function updateWindCompass(angle) {
  const arrowEl = document.getElementById('wind-arrow');
  if (arrowEl) {
    arrowEl.style.transform = `translate(-50%, -50%) rotate(${angle * 180 / Math.PI}deg)`;
  }
}

// Set up the legend
function setupLegend(map, COLORS) {
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
  return legend.addTo(map);
}

// Add wind compass
function setupWindCompass(map) {
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
  return windCompass.addTo(map);
}
