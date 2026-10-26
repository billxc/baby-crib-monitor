"""Serial live sampling. All endpoints, calibration and data stay in private config."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid

import imageio_ffmpeg
from PIL import Image
from filtering import OccupancyFilter
from imaging import transform
from inference import infer, PROMPT
from runner import save_json


def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


@contextmanager
def exclusive(path):
    """OS-owned lock, released on crash; never truncate another process's lock."""
    with path.open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def capture(config, crop, full, frame):
    subprocess.run([config.get('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe(),
                    '-hide_banner', '-loglevel', 'error', '-nostdin', '-threads', '2',
                    '-rtsp_transport', 'tcp', '-i', config['stream_url'],
                    '-map', '0:v:0', '-an', '-frames:v', '1',
                    '-pix_fmt', 'yuvj420p', '-color_range', 'pc',
                    '-update', '1', '-y', str(full)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                   timeout=config.get('capture_timeout_seconds', 20))
    # Exit code alone is not proof of an extracted frame.
    if not full.is_file() or not full.stat().st_size:
        raise ValueError('no_frame_output')
    with Image.open(full) as image:
        image.load()
        transform(image.convert('RGB'), crop).save(frame, quality=95)
    with Image.open(frame) as image:
        image.load()
        if list(image.size) != crop['output_size']:
            raise ValueError('crop_size_mismatch')
        return list(image.size)


def observe(config, crop, session, index, smoother, http):
    epoch = time.time()
    name = f'{index:08d}'
    full, frame = session/'full'/(name+'.jpg'), session/'frames'/(name+'.jpg')
    record = {'id': name, 'epoch': epoch, 'timestamp': stamp(epoch),
              'roi_mode': 'static', 'raw': 'missing', 'missing_reason': None,
              'error': None, 'inference_seconds': None, 'image_count': 0,
              'frame': None, 'image_size': None}
    try:
        record['image_size'] = capture(config, crop, full, frame)
        record['frame'] = str(frame.relative_to(session))
    except Exception as exc:
        # Avoid echoing private URLs or FFmpeg command lines into public consoles.
        record['missing_reason'] = 'capture_or_decode_failed'
        record['error'] = {'stage': 'capture', 'type': type(exc).__name__,
                           'detail': str(exc)}
    else:
        try:
            record['image_count'] = 1
            raw, seconds, response = infer(http, frame, config,
                                          config.get('inference_timeout_seconds', 35))
            record.update(raw=raw, inference_seconds=seconds, usage=response.get('usage'),
                          model_response=response['choices'][0]['message']['content'])
        except Exception as exc:
            # An available image with failed inference is unknown, not a gap in images.
            record['raw'] = 'unclear'
            record['error'] = {'stage': 'inference', 'type': type(exc).__name__,
                               'detail': str(exc)}
    record['filtered'] = smoother.update(record['raw'])
    record['held_unknown'] = record['raw'] == 'unclear' and record['filtered'] != 'unclear'
    record['completed_timestamp'] = stamp(time.time())
    return record


def next_tick(due, now, interval):
    """Advance to the next future tick; never queue missed work."""
    candidate = due + interval
    skipped = max(0, math.floor((now-candidate)/interval)+1) if now >= candidate else 0
    return candidate + skipped*interval, skipped


def run(config_path, cycles=None, clock=time.monotonic, sleep=time.sleep):
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    interval = config.get('interval_seconds', 60)
    if interval != 60:
        raise ValueError('Live sampling requires interval_seconds=60')
    crop = json.loads((path.parent/config['crop_config']).read_text(encoding='utf-8'))
    root = Path(config['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    with exclusive(root/'runner.lock'):
        session = root/'sessions'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
        for name in ('frames', 'full'):
            (session/name).mkdir(parents=True, exist_ok=True)
        save_json(session/'run-config.json', {'config': config, 'crop': crop, 'prompt': PROMPT,
                                             'roi_mode': 'static', 'filter_window_samples': 3})
        smoother = OccupancyFilter()
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        due, index, last, skipped = clock(), 0, None, 0

        def heartbeat(phase):
            save_json(root/'heartbeat.json', {'timestamp': stamp(time.time()), 'pid': os.getpid(),
                      'phase': phase, 'session': str(session), 'completed': index,
                      'interval_seconds': interval, 'roi_mode': 'static'})

        while cycles is None or index < cycles:
            while clock() < due:
                heartbeat('waiting')
                sleep(min(5, due-clock()))
            heartbeat('sampling')
            record = observe(config, crop, session, index+1, smoother, http)
            record['skipped_ticks_before'] = skipped
            with (session/'results.jsonl').open('a', encoding='utf-8') as out:
                out.write(json.dumps(record, ensure_ascii=False)+'\n')
                out.flush()
            index += 1
            last = dict(record, session=str(session), interval_seconds=interval)
            save_json(root/'current.json', last)
            heartbeat('waiting')
            print(json.dumps({'timestamp': record['timestamp'], 'raw': record['raw'],
                              'filtered': record['filtered'], 'error': record['error']},
                             ensure_ascii=False), flush=True)
            due, skipped = next_tick(due, clock(), interval)
            if skipped:
                # Missing wall-clock observations must not carry stale votes forward.
                smoother.update('missing')
                save_json(root/'current.json', dict(last, filtered='unclear',
                          missing_reason='scheduler_overrun', skipped_ticks=skipped))
        heartbeat('finished')
        return session


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--cycles', type=int, help='Finite foreground validation (normal cadence)')
    args = parser.parse_args()
    if args.cycles is not None and args.cycles < 1:
        parser.error('--cycles must be positive')
    run(args.config, args.cycles)
