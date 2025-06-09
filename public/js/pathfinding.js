export function buildDistanceLookup(nodes, edges) {
  const map = new Map();
  
  for (const e of edges) {
    const u = e.u !== undefined ? e.u : e.source;
    const v = e.v !== undefined ? e.v : e.target;
    const distance = e.dist ?? e.distance ?? e.cost;
    
    if (u === undefined || v === undefined || distance === undefined) {
      console.warn('[Pathfinding] Invalid edge format:', e);
      continue;
    }
    
    const key1 = `${u}-${v}`;
    const key2 = `${v}-${u}`;
    map.set(key1, distance);
    map.set(key2, distance);
  }
  
  const adjacency = new Map();
  for (let i = 0; i < nodes.length; i++) {
    adjacency.set(i, []);
  }
  
  for (const e of edges) {
    const u = e.u !== undefined ? e.u : e.source;
    const v = e.v !== undefined ? e.v : e.target;
    const distance = e.dist ?? e.distance ?? e.cost;
    
    if (u === undefined || v === undefined || distance === undefined) {
      continue;
    }
    
    if (u >= 0 && u < nodes.length && v >= 0 && v < nodes.length) {
      adjacency.get(u).push({ node: v, dist: distance });
      adjacency.get(v).push({ node: u, dist: distance });
    }
  }
  
  return function (u, v) {
    if (u === v) return 0;
    
    if (u < 0 || u >= nodes.length || v < 0 || v >= nodes.length) {
      return Infinity;
    }
    
    const directKey = `${u}-${v}`;
    if (map.has(directKey)) {
      return map.get(directKey);
    }
    
    return dijkstraDistance(u, v, adjacency);
  };
}

export function dijkstraDistance(start, end, adjacency) {
  if (start === end) return 0;
  
  const distances = new Map();
  const visited = new Set();
  const queue = [{ node: start, dist: 0 }];
  
  distances.set(start, 0);
  
  while (queue.length > 0) {
    queue.sort((a, b) => a.dist - b.dist);
    const { node: current, dist: currentDist } = queue.shift();
    
    if (visited.has(current)) continue;
    visited.add(current);
    
    if (current === end) {
      return currentDist;
    }
    
    const neighbors = adjacency.get(current) || [];
    for (const { node: neighbor, dist: edgeDist } of neighbors) {
      if (visited.has(neighbor)) continue;
      
      const newDist = currentDist + edgeDist;
      if (!distances.has(neighbor) || newDist < distances.get(neighbor)) {
        distances.set(neighbor, newDist);
        queue.push({ node: neighbor, dist: newDist });
      }
    }
  }
  
  return Infinity;
}

export function findGraphPath(start, end, adjacencyList) {
  if (start === end) return [start];
  
  const distances = new Map();
  const previous = new Map();
  const visited = new Set();
  const queue = [{ node: start, dist: 0 }];
  
  distances.set(start, 0);
  
  while (queue.length > 0) {
    queue.sort((a, b) => a.dist - b.dist);
    const { node: current, dist: currentDist } = queue.shift();
    
    if (visited.has(current)) continue;
    visited.add(current);
    
    if (current === end) {
      const path = [];
      let node = end;
      while (node !== undefined) {
        path.unshift(node);
        node = previous.get(node);
      }
      return path;
    }
    
    const neighbors = adjacencyList.get(current) || [];
    for (const { node: neighbor, dist: edgeDist } of neighbors) {
      if (visited.has(neighbor)) continue;
      
      const newDist = currentDist + edgeDist;
      if (!distances.has(neighbor) || newDist < distances.get(neighbor)) {
        distances.set(neighbor, newDist);
        previous.set(neighbor, current);
        queue.push({ node: neighbor, dist: newDist });
      }
    }
  }
  
  return [start, end];
}
