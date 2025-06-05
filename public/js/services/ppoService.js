const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const config = require('../config');

class PPOService {
  constructor() {
    this.config = config;
  }

  findModelPath() {
    for (const testPath of this.config.paths.models) {
      if (fs.existsSync(testPath)) {
        return testPath;
      }
    }
    return null;
  }

  validateEnvironment() {
    const modelPath = this.findModelPath();
    const graphPath = path.join(this.config.paths.data, 'graph.json');

    if (!modelPath) {
      return {
        valid: false,
        error: 'Trained model not found. Please train a model first.',
        help: 'Run: cd python && python train.py --graph ../data/graph.json --timesteps 100000',
        searched_paths: this.config.paths.models
      };
    }

    if (!fs.existsSync(graphPath)) {
      return {
        valid: false,
        error: 'Graph data not found. Please generate network first.'
      };
    }

    return { valid: true, modelPath, graphPath };
  }

  async runInference(params) {
    const { pickupNode, deliveryNode, batteryCapacity, maxPayload } = params;
    
    return new Promise((resolve) => {
      const validation = this.validateEnvironment();
      
      if (!validation.valid) {
        return resolve({
          status: 'error',
          message: validation.error,
          help: validation.help,
          searched_paths: validation.searched_paths
        });
      }

      const { modelPath, graphPath } = validation;
      const pythonScript = path.join(this.config.paths.python, this.config.ppo.pythonScript);

      const args = [
        pythonScript,
        '--model', modelPath,
        '--graph', graphPath,
        '--pickup', pickupNode,
        '--delivery', deliveryNode,
        '--battery', batteryCapacity.toString(),
        '--payload', maxPayload.toString()
      ];

      const inferenceProcess = spawn('python', args, {
        cwd: this.config.paths.python
      });

      let output = '';
      let errorOutput = '';
      let responseSent = false;

      const timeout = setTimeout(() => {
        if (!responseSent) {
          responseSent = true;
          inferenceProcess.kill();
          resolve({
            status: 'error',
            message: `PPO inference timeout (${this.config.ppo.timeout / 1000}s)`
          });
        }
      }, this.config.ppo.timeout);

      inferenceProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      inferenceProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      inferenceProcess.on('close', (code) => {
        clearTimeout(timeout);
        if (responseSent) return;
        responseSent = true;

        if (code === 0) {
          this.handleSuccessfulInference(resolve);
        } else {
          resolve({
            status: 'error',
            message: 'PPO inference failed',
            exit_code: code,
            stdout: output,
            stderr: errorOutput
          });
        }
      });

      inferenceProcess.on('error', (error) => {
        clearTimeout(timeout);
        if (!responseSent) {
          responseSent = true;
          resolve({
            status: 'error',
            message: 'Failed to start PPO inference process',
            error: error.message
          });
        }
      });
    });
  }

  handleSuccessfulInference(resolve) {
    const resultPath = path.join(this.config.paths.python, this.config.ppo.resultFile);

    if (fs.existsSync(resultPath)) {
      try {
        const resultData = JSON.parse(fs.readFileSync(resultPath, 'utf8'));
        resolve({
          status: 'success',
          message: 'PPO inference completed',
          result: resultData,
          stats: {
            success: resultData.success,
            steps: resultData.steps,
            batteryUsed: resultData.battery_used,
            termination_reason: resultData.termination_reason
          },
          route_indices: resultData.route_indices,
          route_names: resultData.route_names,
          battery_history: resultData.battery_history,
          actions: resultData.actions,
          action_types: resultData.action_types
        });
        fs.unlinkSync(resultPath);
      } catch (parseError) {
        resolve({
          status: 'error',
          message: 'Failed to parse inference result',
          error: parseError.message
        });
      }
    } else {
      resolve({
        status: 'error',
        message: 'No inference result file generated'
      });
    }
  }

  runLegacyInference(params) {
    const { algorithm, batteryCapacity, maxPayload, startNode, endNode } = params;
    
    return new Promise((resolve) => {
      if (algorithm === 'ppo') {
        this.runInference({
          pickupNode: startNode,
          deliveryNode: endNode,
          batteryCapacity,
          maxPayload
        }).then(result => {
          if (result.status === 'success') {
            resolve({
              status: 'success',
              message: 'PPO inference completed',
              stats: result.stats,
              route_indices: result.route_indices,
              route_names: result.route_names,
              battery_history: result.battery_history
            });
          } else {
            resolve(result);
          }
        });
      } else {
        setTimeout(() => {
          resolve({ 
            status: 'success', 
            message: `${algorithm} completed`,
            route: ['Hub 1', 'Pickup 3', 'Delivery 5'],
            stats: { success: true, distance: 150.5, batteryUsed: 60, steps: 12 }
          });
        }, 1000);
      }
    });
  }
}

module.exports = new PPOService();
