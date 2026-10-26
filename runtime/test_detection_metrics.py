from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

from PIL import Image
from detection import YoloCropper, padded_box
from filtering import OccupancyFilter
from live import observe
from metrics import Metrics, serve


class DetectionMetricsTest(unittest.TestCase):
    def test_padding_clip_and_real_synthetic_crop(self):
        self.assertEqual(padded_box([2, 4, 92, 44], (100, 50), .04), (0, 2, 96, 46))
        detector = YoloCropper.__new__(YoloCropper)
        detector.config = {}
        class Conf:
            def argmax(self):
                return SimpleNamespace(item=lambda: 1)
            def __getitem__(self, index):
                return SimpleNamespace(item=lambda: .9)
        class Boxes:
            conf = Conf()
            xyxy = [None, SimpleNamespace(tolist=lambda: [2, 4, 92, 44])]
            def __len__(self):
                return 2
        detector.model = Mock()
        detector.model.predict.return_value = [SimpleNamespace(boxes=Boxes())]
        with tempfile.TemporaryDirectory() as tmp:
            full, frame = Path(tmp)/'full.jpg', Path(tmp)/'crop.jpg'
            Image.new('RGB', (100, 50), 'red').save(full)
            result = detector.crop(full, frame, 48)
            with Image.open(frame) as image:
                image.load()
                self.assertEqual(image.size, (48, 22))
                self.assertGreater(image.getpixel((10, 10))[0], 240)
            self.assertEqual(result['crop_xyxy'], [0, 2, 96, 46])
            self.assertEqual(result['crop_size'], [96, 44])
            self.assertEqual(len(result['image_sha256']), 64)
        kwargs = detector.model.predict.call_args.kwargs
        self.assertTrue(kwargs['nms'])
        self.assertEqual((kwargs['imgsz'], kwargs['conf'], kwargs['iou'], kwargs['max_det'], kwargs['classes']),
                         (512, .6, .7, 10, [0]))

    def test_no_box_never_requests_qwen_then_recovers_and_resets_filter(self):
        detector = Mock()
        detector.crop.side_effect = [None, {'image_size': [10, 20], 'confidence': .8}]
        smoother = OccupancyFilter()
        smoother.update('yes')
        metrics = Metrics()
        response = {'choices': [{'message': {'content': '{"in_crib":"no"}'}}]}
        with tempfile.TemporaryDirectory() as tmp, patch('live.capture', return_value=[100, 50]), \
                patch('live.infer', return_value=('no', .1, response)) as request:
            first = observe({}, None, Path(tmp), 1, smoother, None, detector, metrics)
            request.assert_not_called()
            second = observe({}, None, Path(tmp), 2, smoother, None, detector, metrics)
            request.assert_called_once()
        self.assertEqual((first['raw'], first['filtered'], first['missing_reason']),
                         ('missing', 'unclear', 'no_crib_detection'))
        self.assertEqual((second['raw'], second['filtered']), ('no', 'no'))
        text = metrics.render().decode()
        self.assertIn('crib_monitor_samples_total{result="missing"} 1', text)
        self.assertIn('crib_monitor_samples_total{result="no"} 1', text)

    def test_metrics_heartbeat_errors_unclear_and_http_surface(self):
        now = [100]
        metrics = Metrics(clock=lambda: now[0])
        metrics.beat()
        metrics.begin(100)
        base = dict(raw='unclear', filtered='yes', error=None, image_count=1,
                    completed_epoch=102, roi_valid=True, detection={'confidence': .8},
                    stage_duration_seconds=dict(capture=1, detect=.1, qwen=2, total=3.1))
        metrics.complete(base)
        metrics.complete(dict(base, error={'stage': 'inference'}, completed_epoch=110))
        text = metrics.render().decode()
        self.assertIn('crib_monitor_last_success_timestamp_seconds 102', text)
        self.assertIn('crib_monitor_state{kind="raw"} -1', text)
        self.assertIn('crib_monitor_state{kind="filtered"} 1', text)
        self.assertIn('crib_monitor_samples_total{result="error"} 1', text)
        self.assertIn('crib_monitor_samples_total{result="unclear"} 1', text)
        now[0] = 191
        self.assertIn('crib_monitor_up 0', metrics.render().decode())
        metrics.gap()
        self.assertIn('crib_monitor_roi_valid 0', metrics.render().decode())
        # Bind an ephemeral test port without exposing any file routes.
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        config = dict(host='127.0.0.1', port=port, allowed_clients=['127.0.0.1'])
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with serve(metrics, config):
            with http.open(f'http://127.0.0.1:{port}/metrics') as response:
                self.assertEqual(response.status, 200)
            for path in ('/', '/current.json', '/full/example.jpg', '/metrics?config=1'):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    http.open(f'http://127.0.0.1:{port}{path}')
                self.assertEqual(error.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
