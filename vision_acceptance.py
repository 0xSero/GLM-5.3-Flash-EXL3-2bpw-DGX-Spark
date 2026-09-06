#!/usr/bin/env python3
"""Run synthetic paired image/video checks; never send expected answers."""
import argparse,base64,hashlib,json,os,time,urllib.error,urllib.request
from pathlib import Path

def payload(case,root,model):
 p=(root/case['file']).resolve()
 if not p.is_relative_to(root.resolve()):raise ValueError('Unsafe fixture path')
 data=p.read_bytes()
 if hashlib.sha256(data).hexdigest()!=case['sha256']:raise ValueError('Fixture hash mismatch')
 kind=case['kind'];mime='image/png' if kind=='image' else 'video/mp4';field='image_url' if kind=='image' else 'video_url'
 return {'model':model,'temperature':0,'messages':[{'role':'user','content':[{'type':'text','text':case['prompt']},{'type':field,field:{'url':'data:'+mime+';base64,'+base64.b64encode(data).decode()}}]}]}

def score(text,expected):
 text=text.strip()
 if text.startswith('```'):
  lines=text.splitlines();text='\n'.join(lines[1:-1]) if lines[-1].strip()=='```' else text
 try:actual=json.loads(text)
 except ValueError:return False,None
 # Strict types prevent True matching an integer count.
 passed=isinstance(actual,dict) and actual.keys()==expected.keys() and all(type(actual[k]) is type(v) and actual[k]==v for k,v in expected.items())
 return passed,actual

def report(model,rows,expected_ids):
 complete=len(rows)==len(expected_ids) and [r['id'] for r in rows]==expected_ids
 return {'model':model,'complete':complete,'expected_cases':len(expected_ids),'completed_cases':len(rows),'all_selected_pass':bool(rows) and complete and all(r['passed'] for r in rows),'scope':'Synthetic paired visual acceptance only; not broad visual quality evaluation.','results':rows}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base-url',required=True);ap.add_argument('--model',required=True)
 ap.add_argument('--fixtures',type=Path,default=Path(__file__).parent/'vision-fixtures');ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--timeout',type=int,default=1800);ap.add_argument('--kind',choices=['image','video','all'],default='all');ap.add_argument('--template-json')
 a=ap.parse_args()
 if a.output.exists():raise ValueError('Use a fresh visual result path')
 manifest=json.loads((a.fixtures/'cases.json').read_text());rows=[];key=os.environ.get('MODEL_API_KEY')
 expected_ids=[c['id'] for c in manifest['cases'] if a.kind=='all' or a.kind==c['kind']]
 for case in manifest['cases']:
  if a.kind!='all' and a.kind!=case['kind']:continue
  body=payload(case,a.fixtures,a.model)
  if a.template_json:body['chat_template_kwargs']=json.loads(a.template_json)
  headers={'Content-Type':'application/json'}
  if key:headers['Authorization']='Bearer '+key
  row={'id':case['id'],'kind':case['kind'],'fixture_sha256':case['sha256'],'passed':False};start=time.monotonic()
  try:
   req=urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers=headers)
   with urllib.request.urlopen(req,timeout=a.timeout) as response:d=json.load(response)
   choice=d['choices'][0];content=choice['message'].get('content') or '';passed,actual=score(content,case['expected'])
   row.update(passed=passed and choice.get('finish_reason')=='stop',content=content,actual=actual,expected=case['expected'],finish_reason=choice.get('finish_reason'),usage=d.get('usage'))
  except Exception as e:
   row['error_type']=type(e).__name__
   if isinstance(e,urllib.error.HTTPError):row['http_status']=e.code
  row['seconds']=time.monotonic()-start;rows.append(row)
  a.output.parent.mkdir(parents=True,exist_ok=True)
  temporary=a.output.with_name(a.output.name+'.tmp')
  temporary.write_text(json.dumps(report(a.model,rows,expected_ids),indent=2)+'\n')
  os.replace(temporary,a.output)
  print(json.dumps({'id':row['id'],'passed':row['passed'],'seconds':row['seconds']}),flush=True)
 raise SystemExit(not rows or not all(r['passed'] for r in rows))
if __name__=='__main__':main()
