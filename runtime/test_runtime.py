import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image
from filtering import OccupancyFilter
from imaging import transform
from runner import extract_crop


class RuntimeTest(unittest.TestCase):
    def filtered(self, values):
        f = OccupancyFilter()
        return [f.update(x) for x in values]

    def test_isolated_flip_and_confirmed_transition(self):
        self.assertEqual(self.filtered(['yes', 'yes', 'no', 'yes', 'no', 'no']),
                         ['yes', 'yes', 'yes', 'yes', 'no', 'no'])

    def test_unknown_does_not_vote_and_gap_resets(self):
        self.assertEqual(self.filtered(['unclear', 'yes', 'unclear', 'no', 'missing', 'no']),
                         ['unclear', 'yes', 'yes', 'yes', 'unclear', 'no'])

    def test_zero_exit_without_frame_is_recorded_as_unavailable(self):
        http = SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(b'video'))
        with tempfile.TemporaryDirectory() as directory, patch('runner.subprocess.run'):
            output = Path(directory)
            available = extract_crop(http, {'video_url': 'https://example.invalid/clip',
                                           'offset_seconds': 9},
                                     output/'crop.jpg', {}, 'ffmpeg', output)
            self.assertFalse(available)
            self.assertFalse((output/'crop.jpg').exists())

    def test_crop_coordinates_are_relative_to_previous_output(self):
        im = Image.new('RGB', (20, 10), 'white')
        config = {'source_size': [20, 10], 'operations': [
            {'type': 'rotate', 'clockwise_degrees': 0, 'expand': True, 'fill': '#333333'},
            {'type': 'crop', 'x': 4, 'y': 2, 'width': 5, 'height': 6}],
            'output_size': [5, 6]}
        self.assertEqual(transform(im, config).size, (5, 6))


if __name__ == '__main__':
    unittest.main()
