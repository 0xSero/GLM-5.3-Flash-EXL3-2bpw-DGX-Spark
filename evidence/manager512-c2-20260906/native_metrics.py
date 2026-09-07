"""Restricted vLLM request-timing snapshots; no URLs, headers or labels saved."""
import json, math, re, urllib.request

BASES = ('vllm:request_prefill_time_seconds', 'vllm:request_decode_time_seconds', 'vllm:request_prompt_tokens', 'vllm:request_generation_tokens')
NAMES = {base+suffix for base in BASES for suffix in ('_sum','_count')}
LINE = re.compile(r'^([^\s{]+)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$')
MODEL = re.compile(r'(?:^|,)\s*model_name=("(?:[^"\\]|\\.)*")(?=,|$)')

def parse(text, model):
    values = {}
    for line in text.splitlines():
        match = LINE.fullmatch(line)
        if not match or match[1] not in NAMES: continue
        label = MODEL.search(match[2] or '')
        if not label or json.loads(label[1]) != model: continue
        value = float(match[3])
        if not math.isfinite(value) or value < 0: raise ValueError('Invalid native metric')
        values[match[1]] = values.get(match[1], 0.0) + value
    if set(values) != NAMES: raise ValueError('Required native timing metrics unavailable for served model')
    return values

def snapshot(url, model, token=None):
    headers = {'Authorization':'Bearer '+token} if token else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as response:
        data = response.read(8*1024*1024+1)
    if len(data)>8*1024*1024: raise ValueError('Metrics response too large')
    return parse(data.decode(), model)

def compare(before, after, rows):
    if not rows or not all(row['success'] for row in rows): raise ValueError('Failed request cell')
    delta = {name:after[name]-before[name] for name in NAMES}
    if any(not math.isfinite(v) or v < 0 for v in delta.values()): raise ValueError('Metrics reset or invalid delta')
    if any(delta[base+'_count'] != len(rows) for base in BASES):
        raise ValueError('Native observation count does not match isolated request cell')
    prompt = sum(row['usage']['prompt_tokens'] for row in rows)
    generated = sum(row['usage']['completion_tokens'] for row in rows)
    if delta[BASES[2]+'_sum'] != prompt or delta[BASES[3]+'_sum'] != generated:
        raise ValueError('Native token counts differ from client usage')
    prefill = delta[BASES[0]+'_sum']; decode = delta[BASES[1]+'_sum']
    if prefill <= 0 or decode <= 0: raise ValueError('No positive native timing interval')
    return {'status':'matched_isolated_cell', 'request_count':len(rows),
            'sum_request_prefill_seconds':prefill, 'sum_request_decode_seconds':decode,
            'mean_request_prefill_seconds':prefill/len(rows), 'mean_request_decode_seconds':decode/len(rows),
            'prompt_tokens_per_sum_request_prefill_second':prompt/prefill,
            'reported_counter_deltas':delta,
            'scope':'vLLM request-level timing observations. Concurrent request durations overlap; these sums are not GPU-exclusive time or aggregate throughput. No native decode rate inferred because speculative first-token accounting differs.'}
