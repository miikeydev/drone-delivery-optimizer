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
