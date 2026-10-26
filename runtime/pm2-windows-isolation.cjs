// PM2 7.0.4 on Windows hardcodes global named pipes even with PM2_HOME.
// Preload in EVERY dedicated PM2 CLI/daemon Node invocation (--require), and
// propagate through NODE_OPTIONS when allowing PM2 to spawn its daemon.
// No installed PM2 files or other daemons are modified.
'use strict';
if (process.platform === 'win32') {
  const path = require('node:path');
  const crypto = require('node:crypto');
  if (!process.env.PM2_HOME || !process.env.PM2_PACKAGE_ROOT) {
    throw new Error('Dedicated PM2_HOME and PM2_PACKAGE_ROOT are required');
  }
  const home = path.resolve(process.env.PM2_HOME);
  const file = require.resolve(path.join(process.env.PM2_PACKAGE_ROOT, 'paths.js'));
  const original = require(file);
  const namespace = crypto.createHash('sha256').update(home.toLowerCase()).digest('hex').slice(0, 20);
  require.cache[file].exports = function dedicatedPaths(requestedHome) {
    const result = original(requestedHome);
    if (path.resolve(result.PM2_HOME).toLowerCase() !== home.toLowerCase()) {
      throw new Error('Refusing unexpected PM2 home');
    }
    result.DAEMON_RPC_PORT = '\\\\.\\pipe\\crib-' + namespace + '-rpc';
    result.DAEMON_PUB_PORT = '\\\\.\\pipe\\crib-' + namespace + '-pub';
    result.INTERACTOR_RPC_PORT = '\\\\.\\pipe\\crib-' + namespace + '-interactor';
    return result;
  };
}
