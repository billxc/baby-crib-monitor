"""Synthetic-only checks. Run on the same isolated execution host as the PoC."""
import unittest

import cv2
import numpy as np

from localize import Localizer, oriented_crop, polygon_iou, project
from evaluate import old_polygon


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026)
        self.image = np.zeros((480, 640, 3), np.uint8)
        self.image[80:380, 180:460] = rng.integers(0, 256, (300, 280, 3), dtype=np.uint8)
        self.roi = [[180, 80], [460, 80], [460, 380], [180, 380]]
        regions = [[[185, 85], [310, 85], [310, 175], [185, 175]],
                   [[330, 85], [455, 85], [455, 175], [330, 175]],
                   [[185, 180], [230, 180], [230, 280], [185, 280]],
                   [[185, 285], [310, 285], [310, 375], [185, 375]],
                   [[330, 285], [455, 285], [455, 375], [330, 375]]]
        self.ref = dict(id='synthetic', image=self.image, roi=self.roi, mask_regions=regions)

    def test_independent_similarity_both_descriptors(self):
        matrix = cv2.getRotationMatrix2D((320, 240), 7, 0.95)
        matrix[:, 2] += [17, -9]
        target = cv2.warpAffine(self.image, matrix, (640, 480))
        for algorithm in ['SIFT', 'ORB']:
            loc = Localizer([self.ref], algorithm)
            result = loc.locate(target)
            self.assertTrue(result['accepted'], result)
            np.testing.assert_allclose(result['roi'], project(self.roi, matrix), atol=3)
            self.assertFalse(loc.locate(np.zeros_like(target))['accepted'])
            self.assertTrue(loc.locate(target)['accepted'])

    def test_affine(self):
        matrix = np.float64([[1, 0.06, -12], [0.03, 1, -10]])
        target = cv2.warpAffine(self.image, matrix, (640, 480))
        result = Localizer([self.ref], model='affine').locate(target)
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['roi'], project(self.roi, matrix), atol=3)

    def test_local_support_is_not_full_target(self):
        target = self.image.copy()
        target[180:] = 0
        for algorithm in ['SIFT', 'ORB']:
            self.assertFalse(Localizer([self.ref], algorithm).locate(target)['accepted'])

    def test_crop_preserves_points(self):
        pixels, matrix = oriented_crop(self.image, self.roi)
        points = project(self.roi, matrix)
        self.assertEqual(pixels.shape, (770, 562, 3))
        self.assertTrue(np.all(points > 0))
        self.assertTrue(np.all(points[:, 0] < 562))
        self.assertTrue(np.all(points[:, 1] < 770))

    def test_concave_envelope_is_convexified(self):
        ref = dict(self.ref, roi=[[180, 80], [460, 80], [320, 220],
                                  [460, 380], [180, 380]])
        loc = Localizer([ref])
        self.assertTrue(cv2.isContourConvex(loc.references[0]['roi']))
        self.assertTrue(loc.locate(self.image)['accepted'])

    def test_convex_metric(self):
        self.assertAlmostEqual(polygon_iou(self.roi, self.roi), 1)
        self.assertEqual(polygon_iou(self.roi, np.asarray(self.roi)+1000), 0)

    def test_ordered_crop_inverse(self):
        config = dict(source_size=[640, 480], operations=[
            dict(type='rotate', clockwise_degrees=30),
            dict(type='crop', x=200, y=100, width=120, height=180)])
        polygon = old_polygon(config)
        self.assertEqual(polygon.shape, (4, 2))
        self.assertAlmostEqual(abs(cv2.contourArea(polygon)), 120*180, delta=0.1)


if __name__ == '__main__':
    unittest.main()
