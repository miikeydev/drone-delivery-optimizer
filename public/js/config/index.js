const path = require('path');

// Try to load dotenv, but don't fail if it's not available
try {
  require('dotenv').config();
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
      path.join(__dirname, '..', '..', '..', 'python', 'models', 'drone_ppo_enhanced_final.zip'),
      path.join(__dirname, '..', '..', '..', 'models', 'drone_ppo_enhanced_final.zip'),
      path.join(__dirname, '..', '..', '..', 'python', 'models', 'best_model.zip'),
      path.join(__dirname, '..', '..', '..', 'models', 'best_model.zip')
    ]
  },

  ppo: {
    timeout: parseInt(process.env.PPO_TIMEOUT) || 30000,
    pythonScript: 'live_inference.py',
    resultFile: 'inference_result.json'
  }
};

module.exports = config;
