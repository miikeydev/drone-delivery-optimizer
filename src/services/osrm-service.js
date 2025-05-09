const axios = require('axios');
const BASE = 'https://router.project-osrm.org';

/**
 * @param {string} from - The starting point in "longitude,latitude" format
 * @param {string} to - The destination point in "longitude,latitude" format
 * @returns {Promise<Object>} - The OSRM route response
 */
module.exports.getRoute = async (from, to) => {
  const url = `${BASE}/route/v1/driving/${from};${to}?geometries=geojson&overview=full`;
  const { data } = await axios.get(url);
  if (data.code !== 'Ok') throw new Error('OSRM error ' + data.code);
  return data.routes[0];
};
