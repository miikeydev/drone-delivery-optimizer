# Drone Delivery Optimizer

This project is an interactive web application to simulate and optimize parcel delivery by drone across a map of France. The user can configure various parameters and visualize the optimal path for a drone between a hub, a pickup point, a delivery point, and optionally a return to the hub.

## Main Features

- **Interactive map**: Visualization of hubs, relay points, delivery points, and charging stations.
- **User selection** :
  - Choose pickup and delivery points on the map.
  - Set the drone's initial battery.
  - Select the number of packages (payloads).
  - Set wind azimuth.
  - Show or hide edges and charging stations.
- **Available algorithms** :
  - **Genetic Algorithm (GA)** : Works very well to find optimal paths.
  - **Proximal Policy Optimization (PPO)** : Experimental reinforcement learning agent, results need improvement.
- **Path visualization** : Display of the computed route, steps, battery consumption, and detailed information.

## Algorithm Details

### Genetic Algorithm (GA)

- When a user selects points and parameters, the GA searches for the optimal path for the drone :
  - Automatic departure from the hub closest to the pickup.
  - Passes through pickup, then delivery, then ideally returns to the closest hub.
  - Takes into account battery, number of packages, wind, and charging stations.
- The GA is robust and fast.

### PPO (Reinforcement Learning)

- A PPO implementation is available but remains experimental.
- Current results are not satisfactory, but RL remains a possible avenue for future improvements.
- To use PPO, you must install the necessary Python dependencies and train a model (see below).

## Technologies Used

- **Frontend** : JavaScript (ES modules), Leaflet.js for the map, HTML/CSS.
- **Backend** : Node.js (Express), REST API to launch algorithms.
- **Python** : For PPO (Stable Baselines3, sb3-contrib, PyTorch).
- **Other** : Turf.js for geolocation, graph generation scripts, etc.

## Installation and Launch

### Prerequisites

- Node.js (v16+ recommended)
- Python 3.8+ (for PPO only, optional)

### Installation

```bash
# Install Node.js dependencies
npm install
```

### Start the server

```bash
npm start
```
or 

```bash
node server.js
```

The server will be available at [http://localhost:8000](http://localhost:8000).

### (Optional) Using PPO

1. Install Python dependencies in the `python/` folder :
   ```bash
   cd python
   pip install -r requirements.txt
   ```
2. Generate a graph :
   - Use the web interface to generate and save a graph (`/api/save-graph`).
3. Train a PPO model :
   ```bash
   python train.py --graph ../data/graph.json --timesteps 100000
   ```
4. Run PPO inference from the interface (requires a trained model).

## Usage

- Select pickup and delivery points on the map.
- Set battery, number of packages, and wind azimuth.
- Click "Run GA" to launch the genetic algorithm and visualize the route.
- "Run PPO" is experimental and requires a trained model.

## Limitations and Improvements

- The PPO algorithm is not yet fully operational.
- The interface can be improved for advanced use cases.

## Credits

Project developed by two students and as part of a drone delivery optimization project for school.

---
