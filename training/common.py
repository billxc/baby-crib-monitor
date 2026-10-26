"""Original MIT helper candidates; no model, image, or runtime dependencies."""
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def loss_values(value):
    """Handle 8.4.x dict branches and legacy tensor/list loss, failing on NaN."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): loss_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [loss_values(item) for item in value]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('nonfinite_loss')
    return number


def check_groups(rows):
    """Audit user-assigned groups/splits; never randomly split neighbouring frames."""
    groups, ids, hashes, counts = {}, set(), {}, {}
    for row in rows:
        identifier, group, split = (row.get(key) for key in ('id', 'group', 'split'))
        if not all(isinstance(x, str) and x.strip() for x in (identifier, group, split)):
            raise ValueError('id_group_split_required')
        if split not in {'train', 'val', 'test', 'challenge', 'quarantine'}:
            raise ValueError('invalid_split')
        if identifier in ids:
            raise ValueError('duplicate_id')
        ids.add(identifier)
        if group in groups and groups[group] != split:
            raise ValueError('group_crosses_splits')
        groups[group] = split
        digest = row.get('sha256')
        if digest:
            if digest in hashes and hashes[digest] != split:
                raise ValueError('identical_content_crosses_splits')
            hashes[digest] = split
        counts[split] = counts.get(split, 0) + 1
    if not rows:
        raise ValueError('empty_manifest')
    return counts


def select_threshold(rows):
    """Val-only precomputed TP/FP-box counts; tie -> higher confidence.

    Input rows: split, threshold, tp, fp_boxes, n. This is NOT test mAP.
    A caller computes counts with a preregistered matching/top1 policy.
    """
    if not rows or any(row['split'] != 'val' for row in rows):
        raise ValueError('validation_only')
    candidates = []
    for row in rows:
        threshold, n = float(row['threshold']), row['n']
        tp, fp = row['tp'], row['fp_boxes']
        if (not math.isfinite(threshold) or not 0 <= threshold <= 1
                or any(type(x) is not int for x in (n, tp, fp))
                or n <= 0 or not 0 <= tp <= n or fp < 0):
            raise ValueError('invalid_threshold_counts')
        candidates.append(((tp - fp) / n, threshold))
    return max(candidates)[1]


def verify_freeze(path):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if record.get('schema') != 1 or not record.get('files'):
        raise ValueError('invalid_freeze')
    for name, digest in record['files'].items():
        if not Path(name).is_absolute() or sha256(name) != digest:
            raise ValueError('frozen_input_changed')
    return record
