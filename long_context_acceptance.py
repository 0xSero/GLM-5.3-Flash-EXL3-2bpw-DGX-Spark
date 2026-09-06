#!/usr/bin/env python3
"""Synthetic multi-position retrieval with actual server token-count checks."""
import argparse, hashlib, json, os, secrets
from pathlib import Path
from speed_sweep import FILLER, request
from vision_acceptance import score


def build(tokenizer, target):
    expected={name:secrets.token_hex(8) for name in ('amber','birch','cobalt','dahlia')}
    text='Unique retrieval trial '+secrets.token_hex(16)+'.\n'
    filler=tokenizer.encode(FILLER,add_special_tokens=False)
    if not filler:raise ValueError('Empty filler tokens')
    def padding(count):
        return tokenizer.decode((filler*((count+len(filler)-1)//len(filler)))[:count],skip_special_tokens=False)
    positions={}
    for fraction,(name,value) in zip((0.05,0.35,0.65,0.95),expected.items()):
        current=len(tokenizer.encode(text,add_special_tokens=False))
        text+=padding(max(0,int(target*fraction)-current))
        positions[name]=len(tokenizer.encode(text,add_special_tokens=False))
        text+=f'\nRegistry record: {name} has retrieval value {value}.\n'
    question='\nReturn only a JSON object with keys amber, birch, cobalt, dahlia and their exact retrieval values from the registry records.\n'
    remaining=target-len(tokenizer.encode(text+question,add_special_tokens=False))
    text+=padding(max(0,remaining))+question
    count=len(tokenizer.encode(text,add_special_tokens=False))
    return text,expected,positions,count


def evaluate(row,expected,minimum):
    matched,actual=score(row.get('content',''),expected)
    usage=row.get('usage') or {}
    cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    count=usage.get('prompt_tokens')
    return {'passed':bool(row.get('success') and matched and isinstance(count,int) and count>=minimum and cached in (None,0)),
            'exact_retrieval':matched,'actual':actual,'actual_prompt_tokens':count,'reported_cached_tokens':cached,
            'cache_note':'Unique nonce begins each prompt. Missing cached-token usage is unknown; preserve separate runtime prefix-cache configuration evidence.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url',required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--tokenizer',required=True);ap.add_argument('--tokenizer-revision')
    ap.add_argument('--input-tokens',type=int,default=200000);ap.add_argument('--timeout',type=int,default=7200)
    ap.add_argument('--template-json',default='{}');ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.input_tokens<200000:raise ValueError('This acceptance requires at least200000input tokens')
    if args.output.exists():raise ValueError('Use a fresh output directory')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,revision=args.tokenizer_revision,trust_remote_code=False)
    prompt,expected,positions,count=build(tokenizer,args.input_tokens)
    args.output.mkdir(parents=True)
    (args.output/'prompt.txt').write_text(prompt)
    receipt={'model':args.model,'tokenizer_revision':args.tokenizer_revision,'input_text_tokens':count,
             'target_input_tokens':args.input_tokens,'marker_token_positions':positions,'expected':expected,
             'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'scope':'Four synthetic records at separated positions; not broad long-context reasoning or benchmark quality.'}
    (args.output/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
    row=request(args.base_url,args.model,prompt,args.timeout,os.environ.get('MODEL_API_KEY'),json.loads(args.template_json))
    result={**receipt,'evaluation':evaluate(row,expected,args.input_tokens),'response':row}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['evaluation']),flush=True)
    raise SystemExit(not result['evaluation']['passed'])
if __name__=='__main__':main()
