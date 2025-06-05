// server.js
const express = require('express');
const path = require('path');
const config = require('./public/js/config');
const apiRoutes = require('./public/js/routes/api');

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


