"""Low-cardinality Prometheus exporter. HTTP reads memory, never state files."""
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time


class Metrics:
    def __init__(self, stale_seconds=90, clock=time.time):
        self.clock, self.stale_seconds = clock, stale_seconds
        self.started = clock()
        self.heartbeat = 0
        self.last_sample = self.last_success = 0
        self.raw = self.filtered = 'missing'
        self.roi_valid = self.confidence = 0
        self.counts = Counter()
        self.durations = dict.fromkeys(('capture', 'detect', 'qwen', 'total'), 0)
        self.lock = threading.Lock()

    def beat(self):
        with self.lock:
            self.heartbeat = self.clock()

    def begin(self, epoch):
        with self.lock:
            self.last_sample = epoch

    def gap(self):
        with self.lock:
            self.raw = self.filtered = 'missing'
            self.roi_valid = self.confidence = 0

    def complete(self, record):
        with self.lock:
            self.raw, self.filtered = record['raw'], record['filtered']
            result = 'error' if record['error'] else record['raw']
            self.counts[result] += 1
            if not record['error'] and record['image_count'] == 1 and record['raw'] != 'missing':
                self.last_success = record['completed_epoch']
            self.roi_valid = int(record.get('roi_valid', False))
            self.confidence = record.get('detection', {}).get('confidence', 0) if self.roi_valid else 0
            self.durations = record['stage_duration_seconds'].copy()

    def render(self):
        with self.lock:
            state = {'yes': 1, 'no': 0}
            values = [
                ('crib_monitor_state', 'gauge', [('kind="raw"', state.get(self.raw, -1)),
                                                ('kind="filtered"', state.get(self.filtered, -1))]),
                ('crib_monitor_up', 'gauge', [('', int(self.heartbeat > 0 and self.clock()-self.heartbeat <= self.stale_seconds))]),
                ('crib_monitor_last_sample_timestamp_seconds', 'gauge', [('', self.last_sample)]),
                ('crib_monitor_last_success_timestamp_seconds', 'gauge', [('', self.last_success)]),
                ('crib_monitor_samples_total', 'counter', [(f'result="{r}"', self.counts[r]) for r in ('yes', 'no', 'unclear', 'missing', 'error')]),
                ('crib_monitor_stage_duration_seconds', 'gauge', [(f'stage="{s}"', v) for s, v in self.durations.items()]),
                ('crib_monitor_yolo_confidence', 'gauge', [('', self.confidence)]),
                ('crib_monitor_roi_valid', 'gauge', [('', self.roi_valid)]),
                ('crib_monitor_process_start_time_seconds', 'gauge', [('', self.started)])]
            lines = []
            for name, kind, samples in values:
                lines.append(f'# TYPE {name} {kind}')
                for labels, value in samples:
                    suffix = '{'+labels+'}' if labels else ''
                    lines.append(f'{name}{suffix} {value}')
            return ('\n'.join(lines)+'\n').encode('ascii')


@contextmanager
def serve(metrics, config):
    if not config:
        yield
        return
    allowed = set(config['allowed_clients'])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Defense in depth in addition to a deployment firewall allowlist.
            if self.client_address[0] not in allowed:
                self.send_error(403)
                return
            if self.path != '/metrics':
                self.send_error(404)
                return
            body = metrics.render()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((config['host'], config['port']), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
