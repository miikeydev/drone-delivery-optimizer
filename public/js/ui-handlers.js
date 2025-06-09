export function setupAlgorithmToggle() {
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
}

export function setupBatterySlider() {
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
}

export function setupPackageSelection() {
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
}

export function setupWindCompass(windAngle, setWindAngleFromDegrees) {
  const windCompass = document.getElementById('wind-compass');
  const windArrow = document.getElementById('wind-arrow');
  const windAngleValue = document.getElementById('wind-angle-value');
  
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

  return {
    setWindAngleFromDegrees: function(degrees) {
      if (windArrow) {
        windArrow.style.transform = `translate(-50%, -50%) rotate(${degrees}deg)`;
      }
      if (windAngleValue) {
        windAngleValue.textContent = `${Math.round(degrees)}°`;
      }
    }
  };
}

export function setupVisibilityToggles(map, chargingLayer, edgesLayer) {
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
}

export function setupPinPlacements(map, allNodes, pickupMarker, deliveryMarker, pickupNodeId, deliveryNodeId, updatePickupDisplay, updateDeliveryDisplay, findClosestNode, pickupLayer, deliveryLayer) {
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
}
