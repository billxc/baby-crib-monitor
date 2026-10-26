import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from filtering import OccupancyFilter
from inference import infer
from live import capture, exclusive, next_tick, observe, run


class LiveTest(unittest.TestCase):
    def test_next_tick_skips_overruns_not_backlog(self):
        self.assertEqual(next_tick(0, 5, 60), (60, 0))
        self.assertEqual(next_tick(0, 60, 60), (120, 1))
        self.assertEqual(next_tick(0, 135, 60), (180, 2))

    def test_one_image_and_fixed_inference_protocol(self):
        class HTTP:
            def open(self, request, timeout):
                self.payload = json.loads(request.data)
                return io.BytesIO(json.dumps({'choices': [{'message': {
                    'content': '{"in_crib":"yes"}'}}]}).encode())
        http = HTTP()
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp)/'synthetic.jpg'
            Image.new('RGB', (5, 6)).save(frame)
            raw, _, _ = infer(http, frame, {'model': 'local-model',
                                           'model_base_url': 'http://example.invalid/v1'})
        self.assertEqual(raw, 'yes')
        payload = http.payload
        self.assertEqual(len(payload['messages'][1]['content']), 1)
        self.assertEqual(payload['messages'][1]['content'][0]['type'], 'image_url')
        self.assertEqual((payload['temperature'], payload['seed'],
                          payload['max_completion_tokens']), (0, 42, 32))
        self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])

    def test_capture_zero_exit_requires_actual_image(self):
        with tempfile.TemporaryDirectory() as tmp, patch('live.subprocess.run') as process:
            with self.assertRaisesRegex(ValueError, 'no_frame_output'):
                capture({'stream_url': 'rtsp://example.invalid/demo', 'ffmpeg': 'ffmpeg'},
                        {}, Path(tmp)/'full.jpg', Path(tmp)/'frame.jpg')
            command = process.call_args.args[0]
            self.assertIn('yuvj420p', command)
            self.assertIn('pc', command)
            self.assertIn('tcp', command)

    def test_inference_failure_is_unknown_and_next_observation_recovers(self):
        smoother = OccupancyFilter()
        with tempfile.TemporaryDirectory() as tmp, patch('live.capture', return_value=[5, 6]), \
                patch('live.infer', side_effect=[TimeoutError('test timeout'),
                                                ('no', .1, {'choices': [{'message': {
                                                    'content': '{"in_crib":"no"}'}}]})]):
            first = observe({}, {}, Path(tmp), 1, smoother, None)
            second = observe({}, {}, Path(tmp), 2, smoother, None)
        self.assertEqual((first['raw'], first['filtered']), ('unclear', 'unclear'))
        self.assertEqual(first['error']['stage'], 'inference')
        self.assertEqual((second['raw'], second['filtered']), ('no', 'no'))
        self.assertIsNone(second['error'])

    def test_real_loop_uses_cadence_and_recovers_after_missing_image(self):
        now, starts = [0.0], []
        def sleep(seconds):
            now[0] += seconds
        def fake_capture(*args):
            starts.append(now[0])
            if len(starts) == 2:
                raise ValueError('synthetic missing frame')
            return [5, 6]
        response = {'choices': [{'message': {'content': '{"in_crib":"yes"}'}}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'crop.json').write_text('{}')
            config = root/'config.json'
            config.write_text(json.dumps({'crop_config': 'crop.json',
                                          'output_dir': str(root/'data')}))
            with patch('live.capture', side_effect=fake_capture), \
                    patch('live.infer', return_value=('yes', .1, response)):
                session = run(config, cycles=3, clock=lambda: now[0], sleep=sleep)
            records = [json.loads(line) for line in (session/'results.jsonl').read_text().splitlines()]
            self.assertEqual(starts, [0, 60, 120])
            self.assertEqual([r['raw'] for r in records], ['yes', 'missing', 'yes'])
            self.assertEqual([r['filtered'] for r in records], ['yes', 'unclear', 'yes'])
            self.assertEqual(json.loads((root/'data'/'heartbeat.json').read_text())['phase'], 'finished')

    def test_lock_prevents_overlapping_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            with exclusive(Path(tmp)/'runner.lock'):
                with self.assertRaises(OSError):
                    with exclusive(Path(tmp)/'runner.lock'):
                        self.fail('lock acquired twice')


if __name__ == '__main__':
    unittest.main()
