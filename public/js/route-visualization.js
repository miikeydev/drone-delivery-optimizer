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
  
  routeIndices.forEach((nodeIdx, stepIdx) => {
    if (nodeIdx < allNodes.length) {
      const node = allNodes[nodeIdx];
      const battery = batteryHistory[stepIdx] || 100;
      
      let markerColor = '#27ae60';
      if (battery < 20) markerColor = '#e74c3c';
      else if (battery < 50) markerColor = '#f39c12';
      else if (battery < 80) markerColor = '#f1c40f';
      
      const isStart = stepIdx === 0;
      const isEnd = stepIdx === routeIndices.length - 1;
      const radius = isStart ? 12 : (isEnd ? 10 : 8);
      const weight = isStart ? 3 : 2;
      const borderColor = isStart ? '#2c3e50' : '#fff';
      
      const stepMarker = L.circleMarker([node.lat, node.lng], {
        radius: radius,
        fillColor: markerColor,
        color: borderColor,
        weight: weight,
        fillOpacity: 0.9
      }).addTo(routeLayer);
      
      let distanceFromPrevious = 0;
      let energyUsed = 0;
      let timeElapsed = 0;
      
      if (stepIdx > 0 && routeIndices[stepIdx - 1] < allNodes.length) {
        const prevNode = allNodes[routeIndices[stepIdx - 1]];
        distanceFromPrevious = Math.sqrt(
          Math.pow((node.lat - prevNode.lat) * 111, 2) + 
          Math.pow((node.lng - prevNode.lng) * 111 * Math.cos(node.lat * Math.PI / 180), 2)
        );
        energyUsed = distanceFromPrevious * 0.8;
        timeElapsed = distanceFromPrevious / 60; 
      }
      
      let popupContent = `
        <div style="min-width: 250px;">
          <h4 style="margin: 0 0 8px 0; color: ${routeColor};">
            ${isStart ? 'START' : (isEnd ? 'END' : `Step ${stepIdx + 1}`)}
          </h4>
          <b>Node:</b> ${node.id}<br>
          <b>Type:</b> ${node.type}<br>
          <b>Position:</b> ${node.lat.toFixed(4)}, ${node.lng.toFixed(4)}<br>
          <b>Battery:</b> ${battery.toFixed(1)}%<br>
      `;
      
      if (stepIdx > 0) {
        const prevBattery = batteryHistory[stepIdx - 1] || 100;
        const batteryDrop = prevBattery - battery;
        popupContent += `<b>Distance from previous:</b> ${distanceFromPrevious.toFixed(2)} km<br>`;
        popupContent += `<b>Energy consumed:</b> ${batteryDrop.toFixed(1)}%<br>`;
        popupContent += `<b>Flight time:</b> ${timeElapsed.toFixed(1)} min<br>`;
      }
      
      if (extraInfo.actions && extraInfo.actionTypes && stepIdx < extraInfo.actions.length) {
        const action = extraInfo.actions[stepIdx];
        const actionType = extraInfo.actionTypes[stepIdx];
        popupContent += `<b>Action:</b> ${action} (${actionType})<br>`;
      }
      
      if (extraInfo.algorithm) {
        popupContent += `<b>Algorithm:</b> ${extraInfo.algorithm}<br>`;
      }
      
      if (node.type === 'pickup') {
        popupContent += `<span style="color: ${COLORS.pickup};">Pickup Point</span><br>`;
        popupContent += `<b>Status:</b> Package collected<br>`;
      } else if (node.type === 'delivery') {
        popupContent += `<span style="color: ${COLORS.delivery};">Delivery Point</span><br>`;
        popupContent += `<b>Status:</b> Package delivered<br>`;
      } else if (node.type === 'charging') {
        popupContent += `<span style="color: ${COLORS.charging};">Charging Station</span><br>`;
        if (stepIdx > 0 && stepIdx < routeIndices.length - 1) {
          const nextBattery = batteryHistory[stepIdx + 1] || battery;
          if (nextBattery > battery) {
            popupContent += `<b>Status:</b> Battery recharged to ${nextBattery.toFixed(1)}%<br>`;
          }
        }
      } else if (node.type === 'hubs') {
        if (isStart) {
          popupContent += `<span style="color: ${COLORS.hubs};">Departure Hub</span><br>`;
          popupContent += `<b>Status:</b> Mission started<br>`;
        } else if (isEnd) {
          popupContent += `<span style="color: ${COLORS.hubs};">Arrival Hub</span><br>`;
          popupContent += `<b>Status:</b> Mission completed<br>`;
        } else {
          popupContent += `<span style="color: ${COLORS.hubs};">Hub</span><br>`;
          popupContent += `<b>Status:</b> Transit point<br>`;
        }
      }
      
      // Add cumulative statistics
      if (stepIdx > 0) {
        const totalDistance = routeIndices.slice(0, stepIdx + 1).reduce((sum, _, idx) => {
          if (idx === 0) return 0;
          const curr = allNodes[routeIndices[idx]];
          const prev = allNodes[routeIndices[idx - 1]];
          return sum + Math.sqrt(
            Math.pow((curr.lat - prev.lat) * 111, 2) + 
            Math.pow((curr.lng - prev.lng) * 111 * Math.cos(curr.lat * Math.PI / 180), 2)
          );
        }, 0);
        
        const totalEnergyUsed = 100 - battery;
        const totalTime = totalDistance / 60;
        
        popupContent += `<hr style="margin: 8px 0;">`;
        popupContent += `<b>Cumulative distance:</b> ${totalDistance.toFixed(2)} km<br>`;
        popupContent += `<b>Total energy used:</b> ${totalEnergyUsed.toFixed(1)}%<br>`;
        popupContent += `<b>Total flight time:</b> ${totalTime.toFixed(1)} min<br>`;
      }
      
      popupContent += '</div>';
      stepMarker.bindPopup(popupContent);
      
      // Make the marker clickable for detailed info
      stepMarker.on('click', function() {
        stepMarker.openPopup();
      });
      
      if (stepIdx > 0 && !isEnd) {
        const stepLabel = L.divIcon({
          html: `<div style="color: white; font-weight: bold; font-size: 10px; text-align: center; line-height: 16px;">${stepIdx}</div>`,
          className: 'step-number-label',
          iconSize: [16, 16]
        });
        const labelMarker = L.marker([node.lat, node.lng], { icon: stepLabel }).addTo(routeLayer);
        
        // Also make the label clickable
        labelMarker.on('click', function() {
          stepMarker.openPopup();
        });
      }
    }
  });
  
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
    }
  }, 8000);
  
  if (routeLayer.getLayers().length > 0) {
    const group = new L.featureGroup(routeLayer.getLayers());
    map.fitBounds(group.getBounds().pad(0.15));
  }
  
  console.log(`Route visualized: ${routeIndices.length} steps, ${extraInfo.algorithm || 'Algorithm'} ${status}`);
}
