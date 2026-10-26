"""Offline historical sampling: exactly one image per inference request."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import imageio_ffmpeg
from PIL import Image
from filtering import OccupancyFilter
from imaging import transform

from inference import PROMPT, infer


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def summarize(output, records, samples, interval, finished=False):
    value = {'finished': finished, 'completed': len(records), 'total': len(samples),
             'interval_seconds': interval,
             'raw_counts': dict(Counter(r['raw'] for r in records)),
             'missing_reasons': dict(Counter(r.get('missing_reason') or 'no_recording'
                                            for r in records if r['raw'] == 'missing')),
             'filtered_counts': dict(Counter(r['filtered'] for r in records)),
             'held_unknown_samples': sum(r['held_unknown'] for r in records),
             'window_start': samples[0]['timestamp'],
             'last_completed': records[-1]['timestamp'] if records else None,
             'updated_epoch': time.time()}
    save_json(output/'progress.json', value)
    if not finished:
        return
    save_json(output/'summary.json', value)
    with (output/'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'timestamp', 'raw', 'filtered', 'held_unknown', 'missing_reason', 'inference_seconds'])
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in writer.fieldnames} for row in records)
    segments = []
    for record in records:
        if not segments or segments[-1]['label'] != record['filtered']:
            segments.append({'start_epoch': record['epoch'], 'end_epoch': record['epoch'] + interval,
                             'label': record['filtered'], 'samples': 1,
                             'held_unknown_samples': int(record['held_unknown'])})
        else:
            segments[-1]['end_epoch'] = record['epoch'] + interval
            segments[-1]['samples'] += 1
            segments[-1]['held_unknown_samples'] += int(record['held_unknown'])
    save_json(output/'segments.json', segments)


def extract_crop(http, sample, frame, crop, ffmpeg, output):
    video, full = output/'current.mp4', output/'current-full.jpg'
    full.unlink(missing_ok=True)
    with http.open(sample['video_url'], timeout=60) as source, video.open('wb') as target:
        shutil.copyfileobj(source, target)
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-threads', '2',
                    '-ss', str(sample['offset_seconds']), '-i', str(video),
                    '-pix_fmt', 'yuvj420p', '-color_range', 'pc',
                    '-frames:v', '1', '-update', '1', '-y', str(full)],
                   stdout=subprocess.DEVNULL, check=True, timeout=60)
    # A segment's indexed end can extend past its last decodable video frame.
    available = full.exists()
    if available:
        with Image.open(full) as image:
            transform(image.convert('RGB'), crop).save(frame, quality=95)
        full.unlink()
    video.unlink()
    return available


def run(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    root = config_path.parent
    samples = json.loads((root/config['manifest']).read_text(encoding='utf-8'))
    crop = json.loads((root/config['crop_config']).read_text(encoding='utf-8'))
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    frames = output/'frames'
    frames.mkdir(exist_ok=True)
    snapshot = {'config': config, 'crop': crop, 'samples': samples, 'prompt': PROMPT}
    frozen = output/'run-config.json'
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding='utf-8')) == snapshot, 'Run inputs changed'
    else:
        save_json(frozen, snapshot)
    result_path = output/'results.jsonl'
    records = [json.loads(x) for x in result_path.read_text(encoding='utf-8').splitlines()] if result_path.exists() else []
    assert [r['id'] for r in records] == [s['id'] for s in samples[:len(records)]]
    smoother = OccupancyFilter()
    for record in records:
        assert smoother.update(record['raw']) == record['filtered']
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    interval = config['interval_seconds']
    summarize(output, records, samples, interval)
    for sample in samples[len(records):]:
        raw, seconds, response = 'missing', None, None
        missing_reason = 'no_recording'
        frame = frames/(sample['id']+'.jpg')
        available = False
        if sample['video_url']:
            available = frame.exists() or extract_crop(http, sample, frame, crop, ffmpeg, output)
            missing_reason = 'no_frame_at_sample_time' if not available else None
        if available:
            raw, seconds, response = infer(http, frame, config)
        filtered = smoother.update(raw)
        record = {'id': sample['id'], 'epoch': sample['epoch'], 'timestamp': sample['timestamp'],
                  'raw': raw, 'filtered': filtered,
                  'held_unknown': raw == 'unclear' and filtered != 'unclear',
                  'inference_seconds': seconds,
                  'missing_reason': missing_reason,
                  'usage': response.get('usage') if response else None}
        with result_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
            f.flush()
        records.append(record)
        summarize(output, records, samples, interval)
        print(f"{len(records)}/{len(samples)} {sample['timestamp']} raw={raw} filtered={filtered} {seconds}s", flush=True)
        if response:
            time.sleep(.1)
    summarize(output, records, samples, interval, finished=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
