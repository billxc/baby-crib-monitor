"""Shared local-compatible inference: exactly one image, no reference images."""
import base64
import json
import time
import urllib.request

PROMPT = ('判断图中婴儿床内是否有婴儿或小孩，不判断睡眠。床外大人不算。'
          '孩子被被褥部分遮挡但仍能确认时算yes；仅有被褥或玩具不算孩子；'
          '关键区域被遮挡到无法确认时返回unclear。只输出JSON，字段in_crib，'
          '值为yes、no或unclear。')
SCHEMA = {'type': 'json_schema', 'json_schema': {
    'name': 'crib_occupancy', 'strict': True, 'schema': {
        'type': 'object', 'properties': {'in_crib': {
            'type': 'string', 'enum': ['yes', 'no', 'unclear']}},
        'required': ['in_crib'], 'additionalProperties': False}}}


def infer(http, frame, config, timeout=120):
    content = [{'type': 'image_url', 'image_url': {
        'url': 'data:image/jpeg;base64,' + base64.b64encode(frame.read_bytes()).decode()}}]
    payload = {'model': config['model'], 'messages': [
        {'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': content}],
        'temperature': 0, 'seed': 42, 'max_completion_tokens': 32,
        'response_format': SCHEMA, 'chat_template_kwargs': {'enable_thinking': False}}
    request = urllib.request.Request(config['model_base_url'].rstrip('/')+'/chat/completions',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with http.open(request, timeout=timeout) as source:
        response = json.load(source)
    seconds = round(time.perf_counter()-started, 3)
    raw = json.loads(response['choices'][0]['message']['content'])['in_crib']
    if raw not in ('yes', 'no', 'unclear'):
        raise ValueError('Invalid in_crib label')
    return raw, seconds, response
