"""Offline private-data evaluator. All inputs/outputs are caller-owned files.

Dataset is a list of {id, file, split}; annotation document has annotations[id]
with keypoints (named point or null), observability and optional mask_regions/roi.
Reference masks: first two regions are far-end structure; fourth and subsequent
regions are near-end structure. Third region may cover the connecting side.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from localize import Localizer, oriented_crop, project


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def distribution(values):
    return dict(n=len(values), median=float(np.median(values)), p95=float(np.percentile(values, 95))) if values else None


def old_polygon(config):
    """Invert the crop tool's ordered affine coordinates, including expand rounding."""
    matrix, size = np.eye(3), config['source_size']
    for op in config['operations']:
        if op['type'] == 'rotate':
            angle = np.deg2rad(op['clockwise_degrees'])
            c, s = np.cos(angle), np.sin(angle)
            w, h = size
            width = int(np.ceil(abs(w*c)+abs(h*s)))
            height = int(np.ceil(abs(w*s)+abs(h*c)))
            step = np.array([[c, -s, width/2-c*w/2+s*h/2],
                             [s, c, height/2-s*w/2-c*h/2], [0, 0, 1]])
            size = [width, height]
        elif op['type'] == 'crop':
            step = np.array([[1, 0, -op['x']], [0, 1, -op['y']], [0, 0, 1]])
            size = [op['width'], op['height']]
        else:
            raise ValueError(op['type'])
        matrix = step @ matrix
    w, h = size
    return project([[0, 0], [w, 0], [w, h], [0, h]], np.linalg.inv(matrix)[:2])


def measure(result, annotation, reference_annotation):
    pairs = [(reference_annotation['keypoints'][key], value)
             for key, value in annotation['keypoints'].items()
             if value is not None and reference_annotation['keypoints'].get(key) is not None]
    if len(pairs) < 3:
        return dict(outcome='accepted_unscorable', keypoints=len(pairs))
    source, truth = np.float32([p[0] for p in pairs]), np.float32([p[1] for p in pairs])
    span = float(np.max(np.linalg.norm(truth[:, None] - truth[None, :], axis=2)))
    errors = np.linalg.norm(project(source, result['matrix']) - truth, axis=1)
    median, maximum = float(np.median(errors)/span), float(np.max(errors)/span)
    return dict(outcome='success' if median <= 0.03 and maximum <= 0.06 else 'false_accept',
                keypoints=len(pairs), anchor_span_px=span, errors_px=errors.tolist(),
                median_error_px=float(np.median(errors)), median_normalized_error=median,
                max_normalized_error=maximum)


def preview(image, annotation, result, old, target):
    canvas = image.copy()
    cv2.polylines(canvas, [np.int32(old)], True, (0, 0, 255), 5)
    for name, point in annotation['keypoints'].items():
        if point is None:
            continue
        x, y = map(int, point)
        cv2.circle(canvas, (x, y), 12, (0, 255, 0), 4)
        cv2.putText(canvas, name, (x+12, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if 'mask_regions' in annotation:
        mask = np.zeros(image.shape[:2], np.uint8)
        for polygon in annotation['mask_regions']:
            cv2.fillPoly(mask, [np.int32(polygon)], 255)
        color = canvas.copy()
        color[mask > 0] = (255, 0, 255)
        canvas = cv2.addWeighted(canvas, 0.45, color, 0.55, 0)
    if result.get('accepted'):
        cv2.polylines(canvas, [np.int32(result['roi'])], True, (255, 255, 0), 5)
        for p in result['inlier_target']:
            cv2.circle(canvas, tuple(map(int, p)), 5, (255, 0, 255), -1)
    # Full-width top area, so a moved target cannot be cut off by a preview window.
    canvas = cv2.resize(canvas[:1600], (1440, 600))
    cv2.putText(canvas, target.stem+' '+str(result.get('outcome', 'reference')), (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(target), canvas)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--split', choices=['reference', 'development', 'test'], required=True)
    p.add_argument('--algorithm', choices=['SIFT', 'ORB'], default='SIFT')
    p.add_argument('--model', choices=['similarity', 'affine'], default='similarity')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--long-edge', type=int, default=1600)
    p.add_argument('--old-crop', required=True)
    args = p.parse_args()
    root = args.root
    dataset = load(root/'dataset.json')
    annotations = load(root/'annotations.json')['annotations']
    out = root/(args.split+'-'+args.algorithm+'-'+args.model+'-'+str(args.long_edge))
    out.mkdir(exist_ok=True)
    old = old_polygon(load(root/args.old_crop))
    if args.split == 'reference':
        for row in dataset:
            if row['split'] == 'reference':
                preview(cv2.imread(str(root/row['file'])), annotations[row['id']], {}, old, out/(row['id']+'.jpg'))
        return
    start = time.perf_counter()
    refs = [dict(annotations[row['id']], id=row['id'], image=cv2.imread(str(root/row['file'])))
            for row in dataset if row['split'] == 'reference']
    loc = Localizer(refs, args.algorithm, args.model, long_edge=args.long_edge)
    setup = time.perf_counter()-start
    # Warmup on a separate development frame; never retained as a measurement.
    warm = next(row for row in dataset if row['split'] == 'development')
    loc.locate(cv2.imread(str(root/warm['file'])))
    results = []
    for row in dataset:
        if row['split'] != args.split:
            continue
        annotation = annotations[row['id']]
        loc_times, e2e_times, decode_times, crop_times = [], [], [], []
        decisions = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            image = cv2.imread(str(root/row['file']))
            if image is None:
                raise ValueError('Cannot decode '+row['id'])
            decoded = time.perf_counter()
            result = loc.locate(image)
            crop_start = time.perf_counter()
            pixels = None
            if result['accepted']:
                pixels, _ = oriented_crop(image, result['roi'])
            end = time.perf_counter()
            loc_times.append(result['localization_seconds'])
            e2e_times.append(end-start)
            decode_times.append(decoded-start)
            crop_times.append(end-crop_start)
            decisions.append([result['accepted'], result.get('reference')])
        if len({str(d) for d in decisions}) != 1:
            raise AssertionError('Non-deterministic frame result')
        result.update(id=row['id'], timestamp=row.get('timestamp'), split=args.split,
                      algorithm=args.algorithm, model=args.model,
                      localization_timings=loc_times, read_decode_crop_timings=e2e_times,
                      decode_timings=decode_times, crop_timings=crop_times,
                      observability=annotation['observability'], truncation=annotation['truncation'])
        if result['accepted']:
            result.update(measure(result, annotation, annotations[result['reference']]))
            cv2.imwrite(str(out/(row['id']+'-crop.jpg')), pixels)
        else:
            result['outcome'] = 'reject'
        truth_points = np.float32([v for v in annotation['keypoints'].values() if v is not None])
        result['old_anchor_coverage'] = sum(cv2.pointPolygonTest(old, tuple(map(float, v)), False) >= 0 for v in truth_points)/len(truth_points)
        if len(truth_points) == 4:
            hull = cv2.convexHull(truth_points)
            intersection, _ = cv2.intersectConvexConvex(hull, old)
            result['old_structural_envelope_coverage'] = float(intersection/max(1, cv2.contourArea(hull)))
        result['usable_crop_conservative'] = result['outcome']=='success' and annotation['observability']=='clear'
        preview(image, annotation, result, old, out/(row['id']+'-alignment.jpg'))
        results.append(result)
        save(out/'results.json', results)
        print(row['id'], result['outcome'], result.get('reference'), result.get('median_normalized_error'), flush=True)
    counts = dict(Counter(r['outcome'] for r in results))
    summary = dict(counts=counts, total=len(results), setup_seconds=setup,
                   usable_crops=sum(r['usable_crop_conservative'] for r in results),
                   observability=dict(Counter(r['observability'] for r in results)),
                   normalized_error_accepted=distribution([r['median_normalized_error'] for r in results if 'median_normalized_error' in r]),
                   false_accept_rate_all=counts.get('false_accept', 0)/len(results),
                   false_accept_rate_accepted=(counts.get('false_accept', 0)/sum(r['accepted'] for r in results)
                                               if any(r['accepted'] for r in results) else None),
                   timing={k:distribution([t for r in results for t in r[k]]) for k in
                           ['localization_timings','read_decode_crop_timings','decode_timings','crop_timings']},
                   timing_by_outcome={status:distribution([t for r in results if r['outcome']==status for t in r['localization_timings']]) for status in counts},
                   environment=dict(machine=platform.node(), processor=platform.processor(),
                                    python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                                    threads=cv2.getNumThreads(), opencl=cv2.ocl.useOpenCL(),
                                    long_edge=args.long_edge, references=len(refs), warmup_frames=1,
                                    repeats=args.repeats, query_decode='cv2.imread', template_setup_in_timing=False,
                                    preview_and_jpeg_write_in_timing=False),
                   input_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['dataset.json','annotations.json',args.old_crop]})
    save(out/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
