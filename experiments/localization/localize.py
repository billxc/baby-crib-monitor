"""Offline, independent-frame masked multi-reference localization (no model/service).

Coordinates in the public API are original image pixels. A mask region contains
only stable target structure; callers own annotation and privacy. An accepted
transform is an approximate region locator, not a planar reconstruction or an
occlusion/occupancy decision. No temporal state or static-ROI fallback.
"""
import time

import cv2
import numpy as np


def project(points, matrix):
    return cv2.transform(np.asarray(points, np.float32).reshape(-1, 1, 2),
                         np.asarray(matrix, np.float64)).reshape(-1, 2)


def polygon_iou(a, b):
    """Convex polygon metric; annotations must be convex and consistently ordered."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    area_a, area_b = abs(cv2.contourArea(a)), abs(cv2.contourArea(b))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    return float(intersection / max(1e-9, area_a + area_b - intersection))


def oriented_crop(image, polygon, margin=0.12, output_size=(562, 770)):
    """Rotate a padded minimum-area rectangle, never perspective-flatten people.

    Return pixels and original->output affine matrix. Black outside the source
    is padding, not recovered information; the caller must separately gate it.
    """
    rect = cv2.minAreaRect(np.asarray(polygon, np.float32))
    center, (w, h), angle = rect
    if w > h:
        w, h, angle = h, w, angle + 90
    w, h = w * (1 + 2 * margin), h * (1 + 2 * margin)
    ow, oh = output_size
    scale = min(ow / w, oh / h)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[:, 2] += np.array([ow / 2, oh / 2]) - np.array(center)
    return cv2.warpAffine(image, matrix, output_size, flags=cv2.INTER_CUBIC), matrix


class Localizer:
    def __init__(self, references, algorithm='SIFT', model='similarity', long_edge=1600,
                 threads=1, min_inliers=12, ratio=0.75):
        if algorithm not in ('SIFT', 'ORB') or model not in ('similarity', 'affine'):
            raise ValueError('Unsupported algorithm/model')
        cv2.setNumThreads(threads)
        cv2.ocl.setUseOpenCL(False)
        self.algorithm, self.model, self.long_edge = algorithm, model, long_edge
        self.min_inliers, self.ratio = min_inliers, ratio
        self.detector = (cv2.SIFT_create(contrastThreshold=0.02) if algorithm == 'SIFT'
                         else cv2.ORB_create(nfeatures=12000, fastThreshold=10))
        self.matcher = cv2.BFMatcher(cv2.NORM_L2 if algorithm == 'SIFT' else cv2.NORM_HAMMING)
        self.references = []
        for ref in references:
            gray, scale = self.prepare(ref['image'])
            regions = np.zeros(gray.shape, np.uint8)
            for i, polygon in enumerate(ref['mask_regions'], 1):
                cv2.fillPoly(regions, [np.round(np.asarray(polygon) * scale).astype(np.int32)], i)
            kp, desc = self.detector.detectAndCompute(gray, (regions > 0).astype(np.uint8) * 255)
            points = np.array([p.pt for p in kp], np.float32).reshape(-1, 2)
            labels = [int(regions[min(int(p[1]), regions.shape[0]-1),
                                  min(int(p[0]), regions.shape[1]-1)]) for p in points]
            self.references.append(dict(id=ref['id'], scale=scale, points=points,
                                        desc=desc, labels=np.array(labels),
                                        roi=cv2.convexHull(np.asarray(ref['roi'], np.float32)
                                                           * scale).reshape(-1, 2)))

    def prepare(self, image):
        scale = min(1.0, self.long_edge / max(image.shape[:2]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if scale != 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray, scale

    def locate(self, image):
        start = time.perf_counter()
        gray, scale = self.prepare(image)
        kp, desc = self.detector.detectAndCompute(gray, None)
        feature_end = time.perf_counter()
        points = np.array([p.pt for p in kp], np.float32).reshape(-1, 2)
        candidates, diagnostics = [], []
        for ref in self.references:
            diag = dict(reference=ref['id'], template_features=len(ref['points']),
                        matches=0, inliers=0, accepted=False, reason='insufficient_matches')
            diagnostics.append(diag)
            if desc is None or ref['desc'] is None or len(desc) < 2 or len(ref['desc']) < 2:
                continue
            forward = self.matcher.knnMatch(ref['desc'], desc, k=2)
            reverse = self.matcher.match(desc, ref['desc'])
            reverse = {m.queryIdx: m.trainIdx for m in reverse}
            good = [m for pair in forward if len(pair) == 2 for m, n in [pair]
                    if m.distance < self.ratio * n.distance and reverse.get(m.trainIdx) == m.queryIdx]
            diag['matches'] = len(good)
            if len(good) < self.min_inliers:
                continue
            src = np.array([ref['points'][m.queryIdx] for m in good], np.float32)
            dst = np.array([points[m.trainIdx] for m in good], np.float32)
            cv2.setRNGSeed(42)
            estimator = cv2.estimateAffinePartial2D if self.model == 'similarity' else cv2.estimateAffine2D
            matrix, mask = estimator(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3,
                                     maxIters=2000, confidence=0.995, refineIters=10)
            if matrix is None:
                diag['reason'] = 'ransac_failed'
                continue
            inside = mask.ravel().astype(bool)
            support = src[inside]
            n = len(support)
            rate = n / len(good)
            error = float(np.median(np.linalg.norm(project(src, matrix)[inside] - dst[inside], axis=1)))
            region_labels = set(int(ref['labels'][m.queryIdx]) for m, yes in zip(good, inside) if yes)
            region_count = len(region_labels)
            parts_supported = any(i <= 2 for i in region_labels) and any(i >= 4 for i in region_labels)
            hull_area = abs(cv2.contourArea(cv2.convexHull(support))) if n >= 3 else 0
            roi_area = abs(cv2.contourArea(ref['roi']))
            span = np.ptp(support, axis=0) / np.maximum(1, np.ptp(ref['roi'], axis=0))
            singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
            pred = project(ref['roi'], matrix) / scale
            h, w = image.shape[:2]
            coverage, _ = cv2.intersectConvexConvex(pred.astype(np.float32),
                                                  np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
            coverage = float(coverage / max(1, abs(cv2.contourArea(pred))))
            diag.update(inliers=n, inlier_ratio=rate, median_residual_px=error,
                        region_count=region_count, parts_supported=parts_supported,
                        hull_fraction=hull_area/max(1, roi_area),
                        span=span.tolist(), singular_values=singular.tolist(), frame_coverage=coverage)
            reason = None
            if n < self.min_inliers or rate < 0.4 or error > 3:
                reason = 'weak_consensus'
            elif not parts_supported or region_count < 2 or hull_area / max(1, roi_area) < 0.12 or min(span) < 0.35 or max(span) < 0.45:
                reason = 'local_only_support'
            elif np.linalg.det(matrix[:, :2]) <= 0 or min(singular) < 0.65 or max(singular) > 1.5 or max(singular)/min(singular) > 1.35:
                reason = 'implausible_transform'
            elif coverage < 0.95:
                reason = 'insufficient_frame_coverage_not_a_truncation_diagnosis'
            diag.update(accepted=reason is None, reason=reason)
            if reason:
                continue
            original = matrix.copy()
            original[:, :2] *= ref['scale'] / scale
            original[:, 2] /= scale
            candidates.append(dict(reference=ref['id'], roi=pred.tolist(), matrix=original.tolist(),
                                   score=n * min(1, hull_area/max(1, roi_area)),
                                   inlier_source=(support/ref['scale']).tolist(),
                                   inlier_target=(dst[inside]/scale).tolist()))
        candidates.sort(key=lambda x: x['score'], reverse=True)
        result = dict(accepted=False, reason='no_valid_reference', diagnostics=diagnostics,
                      query_features=len(kp), feature_seconds=feature_end-start)
        if candidates:
            if any(polygon_iou(candidates[0]['roi'], c['roi']) < 0.65 for c in candidates[1:]):
                result['reason'] = 'conflicting_references'
            else:
                result.update(candidates[0], accepted=True, reason=None)
        result['localization_seconds'] = time.perf_counter() - start
        return result
