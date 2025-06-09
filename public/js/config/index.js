import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let dotenvLoaded = false;
try {
  console.log('Using default configuration values');
  dotenvLoaded = true;
} catch (e) {
  console.log('dotenv not found, using default values');
}

const config = {
  server: {
    port: process.env.PORT || 8000,
    jsonLimit: process.env.JSON_LIMIT || '50mb'
  },
  
  paths: {
    public: path.join(__dirname, '..', '..', '..', 'public'),
    data: path.join(__dirname, '..', '..', '..', 'data'),
    python: path.join(__dirname, '..', '..', '..', 'python'),
    models: [
      path.join(__dirname, '..', '..', '..', 'models', 'best_model.zip'),
      path.join(__dirname, '..', '..', '..', 'python', 'models', 'best_model.zip')
    ]
  },

  ppo: {
    timeout: parseInt(process.env.PPO_TIMEOUT) || 30000,
    pythonScript: 'live_inference.py',
    resultFile: 'inference_result.json'
  }
};

export default config;
