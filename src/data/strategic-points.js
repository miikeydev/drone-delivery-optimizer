module.exports.strategicPoints = {
  type: "FeatureCollection",
  features: [
    { type: "Feature", properties: { id: "hub-1", name: "Paris Hub" },
      geometry: { type: "Point", coordinates: [2.3522, 48.8566] } },
    { type: "Feature", properties: { id: "depot-1", name: "Lyon Depot" },
      geometry: { type: "Point", coordinates: [4.8357, 45.7640] } },
    { type: "Feature", properties: { id: "delivery-1", name: "Marseille Delivery" },
      geometry: { type: "Point", coordinates: [5.3698, 43.2965] } },

  ]
};
