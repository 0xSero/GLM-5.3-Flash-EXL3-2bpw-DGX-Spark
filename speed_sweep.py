#!/usr/bin/env python3
"""Cold-prefix natural-stop OpenAI streaming sweeps. Client metrics are estimates."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, statistics, threading, time, urllib.error, urllib.request, uuid
from datetime import datetime, timezone
from pathlib import Path

FILLER='The archive describes a numbered cache entry. A reader checks its revision before reuse. A writer publishes a complete snapshot after verification. Independent workers preserve unrelated data.\n'
TASK='\nWrite 160 numbered Python comment lines explaining practical software debugging techniques. Each line must contain a distinct, concrete tip. End after line 160. Do not summarize the archive.\n'

def prepare_prompt(tokenizer, target, nonce):
    prefix='Independent request '+nonce+'. The following archive is context only.\n'
    # Build near target with the actual tokenizer; server chat-template overhead is reported separately.
    ids=tokenizer.encode(FILLER,add_special_tokens=False)
    if not ids:raise ValueError('Tokenizer produced no tokens')
    overhead=len(tokenizer.encode(prefix+TASK,add_special_tokens=False))
    budget=max(0,target-overhead)
    body=tokenizer.decode((ids*((budget+len(ids)-1)//len(ids)))[:budget],skip_special_tokens=False)
    prompt=prefix+body+TASK
    return prompt,len(tokenizer.encode(prompt,add_special_tokens=False))

def request(base, model, prompt, timeout, token=None, template=None):
    payload={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
    if template is not None:payload['chat_template_kwargs']=template
    headers={'Content-Type':'application/json'}
    if token:headers['Authorization']='Bearer '+token
    started=time.monotonic(); first=None;last=None;usage=None;finish=None;events=[];text=[];reasoning=[];done=False
    row={'success':False,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'started_monotonic':started}
    try:
        req=urllib.request.Request(base.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=timeout) as response:
            for raw in response:
                now=time.monotonic()
                if now-started>timeout:raise TimeoutError('wall deadline')
                line=raw.decode('utf-8').strip()
                if not line.startswith('data:'):continue
                chunk=line[5:].strip()
                if chunk=='[DONE]':done=True;break
                event=json.loads(chunk)
                if event.get('error'):raise ValueError('stream error')
                if event.get('usage'):usage=event['usage']
                for choice in event.get('choices',[]):
                    if choice.get('index',0)!=0:continue
                    delta=choice.get('delta') or {}
                    content=delta.get('content') or '';reason=delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if content or reason:
                        first=now if first is None else first;last=now
                        text.append(content);reasoning.append(reason)
                        events.append({'seconds':now-started,'content':content,'reasoning':reason})
                    if choice.get('finish_reason') is not None:finish=choice['finish_reason']
        ended=time.monotonic()
        row.update(elapsed_seconds=ended-started,finish_reason=finish,done=done,usage=usage,events=events,content=''.join(text),reasoning=''.join(reasoning),first_monotonic=first,last_monotonic=last)
        if first is None or last is None:raise ValueError('no generated content')
        if not usage or not isinstance(usage.get('completion_tokens'),int) or not isinstance(usage.get('prompt_tokens'),int):raise ValueError('missing token usage')
        if finish!='stop' or not done:raise ValueError('non-natural or incomplete stream')
        if usage['completion_tokens']<=1 or last<=first:raise ValueError('insufficient timed decode')
        cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        row.update(success=True,ttft_seconds=first-started,decode_seconds=last-first,
                   decode_tokens_per_second_estimate=(usage['completion_tokens']-1)/(last-first),
                   input_tokens_per_ttft_estimate=usage['prompt_tokens']/(first-started),
                   output_tokens_per_wall_second=usage['completion_tokens']/(ended-started),cached_tokens=cached,
                   timing_note='Client estimate: first stream event can contain multiple speculative tokens; input/TTFT includes queue/network/tokenization. Native engine timing must be published separately.')
    except Exception as e:
        row.update(success=False,error_type=type(e).__name__,elapsed_seconds=time.monotonic()-started,events=events,usage=usage,finish_reason=finish)
        if isinstance(e,urllib.error.HTTPError):row['http_status']=e.code
    return row

def summarize(rows,wall):
    good=[r for r in rows if r['success']]
    out={'requests':len(rows),'successes':len(good),'failures':len(rows)-len(good),'wall_seconds':wall}
    # A failed cell has no accepted aggregate score; keep raw successful observations.
    if good and len(good)==len(rows):
        out.update(mean_decode_tokens_per_second_estimate=statistics.fmean(r['decode_tokens_per_second_estimate'] for r in good),
                   median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in good),
                   aggregate_output_tokens_per_wall_second=sum(r['usage']['completion_tokens'] for r in good)/wall)
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url',required=True,help='Private/local API base ending /v1; never saved in results')
    ap.add_argument('--model',required=True);ap.add_argument('--tokenizer',required=True,help='Local tokenizer directory or pinned public ID')
    ap.add_argument('--tokenizer-revision');ap.add_argument('--lengths',default='1024,8192,32768,65536,131072,200000')
    ap.add_argument('--concurrencies',default='1,2,4,8');ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--timeout',type=int,default=3600);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--metrics-url',help='Optional isolated vLLM /metrics endpoint; never saved')
    ap.add_argument('--template-json',default='{}');ap.add_argument('--no-warmup',action='store_true')
    a=ap.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer,revision=a.tokenizer_revision,trust_remote_code=False)
    lengths=[int(x) for x in a.lengths.split(',')];cs=[int(x) for x in a.concurrencies.split(',')]
    if min(lengths+cs+[a.repeats,a.timeout])<=0:raise ValueError('Positive lengths, concurrencies, repeats and timeout required')
    a.output.mkdir(parents=True,exist_ok=True)
    run=uuid.uuid4().hex
    metadata={'run_id':run,'utc':datetime.now(timezone.utc).isoformat(),'model':a.model,'tokenizer_revision':a.tokenizer_revision,'lengths':lengths,'incoming_concurrencies':cs,'repeats':a.repeats,'template':json.loads(a.template_json),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'metrics':'Client estimates; server-native prefill/decode timing is separate.'}
    metadata['native_metrics_enabled']=bool(a.metrics_url)
    if a.metrics_url:
        metadata['native_metrics_source_sha256']=hashlib.sha256(Path(__file__).with_name('native_metrics.py').read_bytes()).hexdigest()
    (a.output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    key=os.environ.get('MODEL_API_KEY');template=metadata['template']
    if not a.no_warmup:
        prompt,_=prepare_prompt(tok,1024,uuid.uuid4().hex)
        warm=request(a.base_url,a.model,prompt,a.timeout,key,template)
        (a.output/'warmup.json').write_text(json.dumps(warm,indent=2)+'\n')
        if not warm['success']:raise SystemExit('Warmup failed; inspect local result')
    summaries=[]
    for length in lengths:
        for concurrency in cs:
            for repeat in range(a.repeats):
                prepared=[prepare_prompt(tok,length,uuid.uuid4().hex) for _ in range(concurrency)]
                barrier=threading.Barrier(concurrency)
                def worker(item):
                    prompt,count=item;barrier.wait();r=request(a.base_url,a.model,prompt,a.timeout,key,template)
                    r['input_text_tokens']=count;return r
                native_before=None
                if a.metrics_url:
                    import native_metrics
                    native_before=native_metrics.snapshot(a.metrics_url,a.model,key)
                start=time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:rows=list(pool.map(worker,prepared))
                summary=summarize(rows,time.monotonic()-start)
                if native_before is not None:
                    deadline=time.monotonic()+5
                    while True:
                        try:
                            summary['native_timing']=native_metrics.compare(native_before,native_metrics.snapshot(a.metrics_url,a.model,key),rows)
                            break
                        except Exception:
                            if time.monotonic()>=deadline:
                                summary['native_timing']={'status':'invalid_or_unmatched','scope':'Missing, reset, delayed or contaminated server observations; no native rate accepted.'}
                                break
                            time.sleep(0.5)
                summary.update(target_input_tokens=length,incoming_concurrency=concurrency,repeat=repeat)
                filename=f'input-{length}-c{concurrency}-r{repeat}.json'
                (a.output/filename).write_text(json.dumps({'summary':summary,'requests':rows},indent=2)+'\n')
                summaries.append(summary);(a.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
                print(json.dumps(summary),flush=True)
                if summary.get('native_timing',{}).get('status')=='invalid_or_unmatched':raise SystemExit('Native metrics failed to match request cell; preserve evidence and diagnose')
                if summary['failures']:raise SystemExit('Sweep stopped on failed cell; preserve evidence and diagnose before increasing load')
if __name__=='__main__':main()
