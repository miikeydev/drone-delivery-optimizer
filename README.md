# Drone Delivery Optimizer

A lightweight prototype for drone delivery route visualization using the public OSRM (Open Source Routing Machine) API for road network routing.

## Prerequisites

- Node.js 14+ and npm
- Internet connection (for OSRM public API access)

## Installing and Running the Application

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the server:
   ```bash
   npm start
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

## Application Structure

- `server.js`: Express server with API endpoints
- `src/services/osrm-service.js`: Service for interacting with OSRM
- `src/data/strategic-points.js`: GeoJSON data for strategic delivery points
- `src/graph.js`: Graph building and conversion utilities
- `public/index.html`: Web interface for visualizing routes

## API Endpoints

- `GET /api/strategic-points`: Returns all strategic points as GeoJSON
- `GET /api/national-graph/geojson`: Returns a graph of connections as GeoJSON
- `GET /api/route?from=lon,lat&to=lon,lat`: Returns an OSRM route between two points
- `GET /api/status`: Returns the status of the OSRM service

## Usage

1. Open the web application in your browser
2. Click on two strategic points on the map
3. The application will automatically calculate and display the optimal route
4. Use the "Reset Selection" button to clear the route and select new points
