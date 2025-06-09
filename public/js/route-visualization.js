/**
 * Route visualization utilities
 */

const COLORS = {
  hubs: '#e74c3c',
  charging: '#2980b9',
  delivery: '#27ae60',
  pickup: '#f39c12'
};

let routeLayer = null;

export function clearRouteVisualization(map) {
  if (routeLayer) {
    map.removeLayer(routeLayer);
    routeLayer = null;
  }
}

export function visualizeRoute(routeIndices, batteryHistory = [], extraInfo = {}, allNodes, map) {
  clearRouteVisualization(map);
  
  if (!allNodes || routeIndices.length < 1) {
    console.warn('Cannot visualize route: insufficient data');
    return;
  }
  
  routeLayer = L.layerGroup().addTo(map);
  
  const routeColor = extraInfo.failed 
    ? '#e74c3c'
    : (extraInfo.algorithm === 'PPO' ? '#666' : '#2ecc71');
  
  const dashPattern = extraInfo.failed ? '10, 10' : (extraInfo.algorithm === 'PPO' ? '15, 5' : 'none');
  
  // Draw route line
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
  
  // Draw step markers
  routeIndices.forEach((nodeIdx, stepIdx) => {
    if (nodeIdx < allNodes.length) {
      const node = allNodes[nodeIdx];
      const battery = batteryHistory[stepIdx] || 100;
      
      let markerColor = '#27ae60';
      if (battery < 20) markerColor = '#e74c3c';
      else if (battery < 50) markerColor = '#f39c12';
      else if (battery < 80) markerColor = '#f1c40f';
      
      const stepMarker = L.circleMarker([node.lat, node.lng], {
        radius: stepIdx === 0 ? 12 : (stepIdx === routeIndices.length - 1 ? 10 : 8),
        fillColor: markerColor,
        color: stepIdx === 0 ? '#2c3e50' : '#fff',
        weight: stepIdx === 0 ? 3 : 2,
        fillOpacity: 0.9
      }).addTo(routeLayer);
      
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
      
      if (stepIdx > 0) {
        const stepLabel = L.divIcon({
          html: `<div style="color: white; font-weight: bold; font-size: 10px; text-align: center; line-height: 16px;">${stepIdx}</div>`,
          className: 'step-number-label',
          iconSize: [16, 16]
        });
        L.marker([node.lat, node.lng], { icon: stepLabel }).addTo(routeLayer);
      }
    }
  });
  
  // Info badge
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
  
  setTimeout(() => {
    try {
      map.removeControl(infoBadge);
    } catch (e) {
      // Badge already removed
    }
  }, 8000);
  
  // Zoom to route
  if (routeLayer.getLayers().length > 0) {
    const group = new L.featureGroup(routeLayer.getLayers());
    map.fitBounds(group.getBounds().pad(0.15));
  }
  
  console.log(`Route visualized: ${routeIndices.length} steps, ${extraInfo.algorithm || 'Algorithm'} ${status}`);
}
