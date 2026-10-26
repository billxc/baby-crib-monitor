// Copy outside the repository, replace every placeholder, use a dedicated PM2_HOME.
// Invoke PM2 with an absolute Node path and absolute PM2 CLI path.
module.exports = { apps: [{
  name: 'crib-monitor',
  cwd: '/absolute/private/work-directory',
  script: '/absolute/release/runtime/live.py',
  interpreter: '/absolute/isolated/python',
  interpreter_args: '-u',
  args: ['/absolute/private/config.json'],
  exec_mode: 'fork', instances: 1,
  autorestart: true, restart_delay: 10000,
  min_uptime: '30s', max_restarts: 10,
  time: true, merge_logs: true,
  out_file: '/absolute/private/logs/worker-out.log',
  error_file: '/absolute/private/logs/worker-error.log',
  env: { PYTHONUNBUFFERED: '1', OMP_NUM_THREADS: '2', MKL_NUM_THREADS: '2',
         OPENBLAS_NUM_THREADS: '1', YOLO_CONFIG_DIR: '/absolute/private/yolo-settings' }
}] };
