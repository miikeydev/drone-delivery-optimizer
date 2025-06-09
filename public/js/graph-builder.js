import { haversineDistance, gaussianRandom, rgbToHex } from './utils.js';

export function buildGraph(nodes, k = 10) {
  const edges = [];
  
  for (let i = 0; i < nodes.length; i++) {
    const distances = nodes.map((node, j) => ({
      index: j,
      distance: haversineDistance(
        nodes[i].lat, nodes[i].lng,
        node.lat, node.lng
      )
    })).sort((a, b) => a.distance - b.distance);
    
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

export function annotateEdges(edges, nodes, windAngle, alpha = 0.3, beta = 0.05) {
  return edges.map(edge => {
    const sourceNode = nodes[edge.source];
    const targetNode = nodes[edge.target];
    
    const phi = Math.atan2(
      targetNode.lat - sourceNode.lat,
      targetNode.lng - sourceNode.lng
    );
    
    const windEffect = Math.cos(phi - windAngle);
    const noise = gaussianRandom(0, 0.05);
    const cost = edge.distance * (1 - alpha * windEffect + beta * noise);
    
    return {
      ...edge,
      windEffect,
      phi,
      cost
    };
  });
}

export function drawGraph(nodes, edges, edgesLayer) {
  edgesLayer.clearLayers();
  
  const costs = edges.map(e => e.cost);
  const minCost = Math.min(...costs);
  const maxCost = Math.max(...costs);
  const meanCost = costs.reduce((sum, cost) => sum + cost, 0) / costs.length;
  
  const getColor = (cost) => {
    if (cost <= meanCost) {
      const t = (cost - minCost) / (meanCost - minCost);
      return rgbToHex(
        Math.round(46 + t * (241 - 46)),
        Math.round(204 + t * (196 - 204)),
        Math.round(113 + t * (15 - 113))
      );
    } else {
      const t = (cost - meanCost) / (maxCost - meanCost);
      return rgbToHex(
        Math.round(241 - t * (241 - 231)),
        Math.round(196 - t * (196 - 76)),
        Math.round(15 - t * (15 - 60))
      );
    }
  };

  edges.forEach(edge => {
    const source = nodes[edge.source];
    const target = nodes[edge.target];
    const color = getColor(edge.cost);
    const weight = 0.5 + 1.5 * (1 - (edge.cost - minCost) / (maxCost - minCost));
    
    const polyline = L.polyline(
      [[source.lat, source.lng], [target.lat, target.lng]], 
      {
        color: color,
        weight: weight,
        opacity: 0.1,
        zIndex: 100
      }
    ).addTo(edgesLayer);
    
    const decorator = L.polylineDecorator(polyline, {
      patterns: [{
        offset: '70%',
        repeat: 0,
        symbol: L.Symbol.arrowHead({
          pixelSize: 3,
          polygon: true,
          pathOptions: { color: color, fillOpacity: 0.1 }
        })
      }]
    }).addTo(edgesLayer);
    
    polyline.bindTooltip(`
      <b>${source.id} → ${target.id}</b><br>
      Distance: ${edge.distance.toFixed(2)} km<br>
      Wind effect: ${edge.windEffect.toFixed(2)}<br>
      Cost: ${edge.cost.toFixed(2)}
    `, { sticky: true });
    
    polyline.sourceIndex = edge.source;
    polyline.targetIndex = edge.target;
  });
}

export function highlightNodeConnections(nodeIndex, edgesLayer, allEdges) {
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
          opacity: nodeIndex === layer.sourceIndex ? 0.7 : 0.15
        });
      }
    }
  });
  
  if (nodeIndex !== null) {
    edgesLayer.eachLayer(layer => {
      if (layer instanceof L.Polyline && layer.sourceIndex === nodeIndex) {
        layer.setStyle({
          weight: 2,
          opacity: 0.8
        });
        layer.bringToFront();
      }
    });
  }
}
