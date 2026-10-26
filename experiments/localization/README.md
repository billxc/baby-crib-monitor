# Offline localization experiment

This is a deliberately small, **offline-only** masked multi-reference SIFT/ORB
PoC, not a replacement for the static-ROI live runner. It has no network client,
occupancy inference, temporal tracker, automatic occlusion detector or service.
A successful numerical fit is not evidence that a target is observable.

Keep **all** real images, source manifests, masks, annotations, calibration,
results and reports outside this repository. No real-data example is included.
Install `requirements.txt` into an isolated experiment environment; do not
change an existing live environment. With that environment on the execution
host:

```sh
python -m unittest discover -s experiments/localization -p 'test_*.py'
python experiments/localization/evaluate.py /path/to/private/experiment \
  --split reference --old-crop old-crop.json
python experiments/localization/evaluate.py /path/to/private/experiment \
  --split development --algorithm SIFT --model similarity --old-crop old-crop.json
python experiments/localization/evaluate.py /path/to/private/experiment \
  --split test --algorithm ORB --model similarity --old-crop old-crop.json
```

Run synthetic tests and all image processing on the authorized execution host.
Commands above describe usage, not verification of any particular machine.
Review template previews and freeze data splits/thresholds before evaluating
test images. A rerun writes the same output directory: archive a previous run
first if its provenance must be retained.

## Private input contract

- `dataset.json`: list of objects with `id`, `file` (relative to the experiment
  root), `split` (`reference`, `development`, `test`), optional `timestamp` and
  caller-owned provenance. The script does not fetch source recordings.
- `annotations.json`: object with `annotations[id]`; each annotation has
  `keypoints` (named original-image `[x,y]` pairs, or `null` for invisible points),
  manual `observability` (`clear`, `partial`, `severe`) and `truncation`.
- References additionally have `roi` (manual approximate output envelope) and
  `mask_regions` (polygons restricting **keypoint centers** to stable target
  material). The first two mask polygons represent the far end, the third may
  represent the connecting side, and polygons four onward represent the near
  end. Both ends must support an accepted fit. Exclude movable contents,
  people, holes and room background. Descriptors have a surrounding receptive
  field; restricting centers does not prove every contributing pixel belongs
  to the target. Inspect real inlier previews, not just counts.
- `old-crop.json`: the crop tool's ordered rotate/crop configuration, used only
  as a fixed-ROI geometric comparison, not a localization fallback.

`Localizer` searches each image independently against every precomputed
reference, uses reciprocal nearest matches plus a ratio test and RANSAC, and
rejects weak, local-only, implausible, out-of-image or conflicting fits. The
similarity and affine models are separate experiments, not automatic
fallbacks. Template ROI envelopes are convexified for geometric checks. This
is an approximate three-dimensional target-region locator, **not** exact
planar rectification. Homography is intentionally absent.

Outputs include per-frame diagnostics, inlier coordinates, transforms, manual
point errors, overlays and accepted crops. Crops use a padded minimum-area
rectangle and an affine warp, preserving natural perspective. Rotation is
axis-aligned but its 180-degree direction is not semantically head-normalized;
black padding is not recovered image content. Crop margin and orientation
require separate downstream validation before any deployment.

## Evaluation caveats

The sample point criterion is median error <=3% and maximum error <=6% of the
maximum span of visible corresponding manual points, with at least three
points. These are illustrative engineering gates, not universal guarantees.
Invisible points are not invented. Accepted frames with fewer points are
`accepted_unscorable`, neither successes nor proven false accepts.
`usable_crop_conservative` requires geometric success **and manual** `clear`;
it is not an automatically available decision. The fixed ROI comparison is
coverage of the manual **structural anchor envelope**, not mattress IoU.

Timing includes all test frames, including rejects, with one development
warmup and three repetitions per image. Template setup is separate. Localization
includes resize, features, all-template matching and geometric gates. A second
timer includes image read/decode and accepted crop generation, but excludes
preview/JPEG writes and source video download/extraction. OpenCV uses one CPU
thread and OpenCL is disabled. Consecutive repeats are warm-cache measurements,
not independent real-time samples; no speed guarantee follows from this tool.

An observed zero false-accept count is not a proof of reliability, especially
when most frames are rejected. No accepted frames means conditional false-
accept rate is undefined (`null`), not zero. Report denominators, uncertain
annotations, severe occlusion and untested real truncation separately. Do not
treat a rejected fit as an out-of-frame diagnosis or an occupancy observation.
