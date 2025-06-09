// server.js
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import config from './public/js/config/index.js';
import apiRoutes from './public/js/routes/api.js';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

app.use(express.json({ limit: config.server.jsonLimit }));
app.use(express.urlencoded({ limit: config.server.jsonLimit, extended: true }));

app.use(express.static(config.paths.public));
app.use('/data', express.static(config.paths.data));

app.use('/api', apiRoutes);

app.get('/', (req, res) => {
  res.sendFile(path.join(config.paths.public, 'index.html'));
});

app.listen(config.server.port, () => {
  console.log(`Server started on http://localhost:${config.server.port}`);
});


