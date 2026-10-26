"""Offline historical sampling: exactly one image per inference request."""
import base64
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

PROMPT = ('判断图中婴儿床内是否有婴儿或小孩，不判断睡眠。床外大人不算。'
          '孩子被被褥部分遮挡但仍能确认时算yes；仅有被褥或玩具不算孩子；'
          '关键区域被遮挡到无法确认时返回unclear。只输出JSON，字段in_crib，'
          '值为yes、no或unclear。')
SCHEMA = {'type': 'json_schema', 'json_schema': {
    'name': 'crib_occupancy', 'strict': True, 'schema': {
        'type': 'object', 'properties': {'in_crib': {
            'type': 'string', 'enum': ['yes', 'no', 'unclear']}},
        'required': ['in_crib'], 'additionalProperties': False}}}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def summarize(output, records, samples, interval, finished=False):
    value = {'finished': finished, 'completed': len(records), 'total': len(samples),
             'interval_seconds': interval,
             'raw_counts': dict(Counter(r['raw'] for r in records)),
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
        writer = csv.DictWriter(f, fieldnames=['id', 'timestamp', 'raw', 'filtered', 'held_unknown', 'inference_seconds'])
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
        if sample['video_url']:
            frame = frames/(sample['id']+'.jpg')
            if not frame.exists():
                video = output/'current.mp4'
                full = output/'current-full.jpg'
                with http.open(sample['video_url'], timeout=60) as source, video.open('wb') as target:
                    shutil.copyfileobj(source, target)
                subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-threads', '2',
                                '-ss', str(sample['offset_seconds']), '-i', str(video),
                                '-frames:v', '1', '-update', '1', '-y', str(full)],
                               capture_output=True, check=True, timeout=60)
                with Image.open(full) as image:
                    transform(image.convert('RGB'), crop).save(frame, quality=95)
                full.unlink()
                video.unlink()
            content = [{'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,'+base64.b64encode(frame.read_bytes()).decode()}}]
            payload = {'model': config['model'], 'messages': [
                {'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': content}],
                'temperature': 0, 'seed': 42, 'max_completion_tokens': 32,
                'response_format': SCHEMA,
                'chat_template_kwargs': {'enable_thinking': False}}
            request = urllib.request.Request(config['model_base_url'].rstrip('/')+'/chat/completions',
                                             data=json.dumps(payload).encode(),
                                             headers={'Content-Type': 'application/json'})
            started = time.perf_counter()
            with http.open(request, timeout=120) as source:
                response = json.load(source)
            seconds = round(time.perf_counter()-started, 3)
            raw = json.loads(response['choices'][0]['message']['content'])['in_crib']
            assert raw in ('yes', 'no', 'unclear')
        filtered = smoother.update(raw)
        record = {'id': sample['id'], 'epoch': sample['epoch'], 'timestamp': sample['timestamp'],
                  'raw': raw, 'filtered': filtered,
                  'held_unknown': raw == 'unclear' and filtered != 'unclear',
                  'inference_seconds': seconds,
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
