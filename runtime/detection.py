"""Optional Ultralytics CPU localization. Weights and settings are private inputs.

Ultralytics and its derived weights have separate AGPL-3.0 licensing; the
repository's MIT license does not relicense either dependency or weights.
"""
import hashlib
import math
from pathlib import Path

from PIL import Image


def padded_box(xyxy, size, padding):
    if len(xyxy) != 4 or not all(math.isfinite(x) for x in xyxy):
        raise ValueError('invalid_detection_box')
    x1, y1, x2, y2 = xyxy
    if x2 <= x1 or y2 <= y1 or not 0 <= padding <= 1:
        raise ValueError('invalid_detection_box')
    dx, dy = (x2-x1)*padding, (y2-y1)*padding
    box = (max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
           min(size[0], math.ceil(x2+dx)), min(size[1], math.ceil(y2+dy)))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('empty_detection_crop')
    return box


class YoloCropper:
    def __init__(self, config):
        # Lazy import: static/offline users need not install Ultralytics.
        import cv2
        import torch
        from ultralytics import YOLO
        self.config = config
        torch.set_num_threads(config.get('threads', 2))
        cv2.setNumThreads(1)
        checkpoint = Path(config['checkpoint'])
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if config.get('checkpoint_sha256', self.checkpoint_sha256) != self.checkpoint_sha256:
            raise ValueError('checkpoint_hash_mismatch')
        self.model = YOLO(str(checkpoint))

        def configure_cpu(predictor):
            # select_device() resets Torch threads during lazy backend setup.
            # Set the requested budget AFTER that setup, before warmup/inference.
            torch.set_num_threads(config.get('threads', 2))
            cv2.setNumThreads(1)
            if predictor.device.type != 'cpu' or predictor.model.end2end:
                raise ValueError('expected_cpu_one_to_many_backend')
        self.model.add_callback('on_predict_start', configure_cpu)
        if self.model.names.get(config.get('class_id', 0)) != config.get('class_name', 'crib'):
            raise ValueError('checkpoint_class_mismatch')

    def crop(self, full, frame, max_edge=None):
        c = self.config
        with Image.open(full) as source:
            source.load()
            image = source.convert('RGB')
        result = self.model.predict(source=image, device='cpu', imgsz=c.get('imgsz', 512),
                                    conf=c.get('conf', .6), iou=c.get('iou', .7),
                                    max_det=c.get('max_det', 10), classes=[c.get('class_id', 0)],
                                    nms=True, rect=True, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or not len(boxes):
            return None
        index = int(boxes.conf.argmax().item())
        xyxy = boxes.xyxy[index].tolist()
        box = padded_box(xyxy, image.size, c.get('padding', .04))
        cropped = image.crop(box)
        original_size = list(cropped.size)
        if max_edge is not None:
            if not isinstance(max_edge, int) or max_edge < 1:
                raise ValueError('invalid_qwen_max_edge')
            cropped.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        cropped.save(frame, quality=95)
        with Image.open(frame) as check:
            check.load()
            size = list(check.size)
        return {'bbox_xyxy': xyxy, 'crop_xyxy': list(box),
                'confidence': float(boxes.conf[index].item()),
                'full_size': list(image.size), 'crop_size': original_size,
                'image_size': size, 'image_sha256': hashlib.sha256(frame.read_bytes()).hexdigest()}
