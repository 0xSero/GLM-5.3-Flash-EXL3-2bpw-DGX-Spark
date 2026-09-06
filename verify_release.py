#!/usr/bin/env python3
"""Validate release evidence locally. Anonymous remote verification remains separate."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

GATES={'structure','held_out_quality','cuda_graphs','text','tools','image','video','long_context','sustained_decode','speed_matrix','anonymous_model_download','anonymous_image_pull','clean_deployment','public_documentation','secret_audit'}

def verify(root):
    root=Path(root).resolve(); errors=[]
    def need(ok,msg):
        if not ok: errors.append(msg)
    def artifact(path):
        if not isinstance(path,str) or not path:
            return None
        p=(root/path).resolve()
        if not p.is_relative_to(root) or not p.is_file():return None
        return p
    try:d=json.loads((root/'release.json').read_text())
    except (OSError,ValueError) as e:return ['Unreadable release.json: '+type(e).__name__]
    need(d.get('schema_version')==1,'Unsupported schema')
    need(d.get('status')=='accepted','Release is not accepted')
    rel=d.get('release',{}); req=d.get('requirements',{}); ms=d.get('measurements',{})
    for key in ['model_revision','github_revision']:
        need(bool(re.fullmatch(r'[a-f0-9]{40}',str(rel.get(key,'')))),key+' must be immutable commit')
    need(bool(re.fullmatch(r'sha256:[a-f0-9]{64}',str(rel.get('docker_digest','')))),'Docker digest missing')
    need(isinstance(rel.get('docker_image'),str) and bool(rel['docker_image']),'Docker image missing')
    need(bool(re.fullmatch(r'[\w.-]+/[\w.-]+',str(rel.get('model_repository','')))),'Model repository missing')
    need(bool(re.fullmatch(r'https://github.com/[\w.-]+/[\w.-]+',str(rel.get('github_repository','')))),'GitHub URL missing')
    need('linux/arm64' in rel.get('docker_platforms',[]),'ARM64 image missing')
    if d.get('model')=='DeepSeek-V4-Flash Vision':
        need(len(set(rel.get('docker_platforms',[])))>=2,'Required multi-platform Docker manifest missing')
        need(ms.get('fixed_draft')=='K6' and ms.get('graph_verifier_size')==7,'DeepSeek draft/verifier contract not met')
    need(artifact(rel.get('sbom')) is not None,'SBOM missing')
    files=d.get('files',[]); indexed={}
    need(bool(files),'Empty evidence manifest')
    for f in files:
        path=f.get('path'); p=artifact(path)
        need(p is not None,'Missing or unsafe artifact: '+str(path))
        need(path not in indexed,'Duplicate artifact: '+str(path));indexed[path]=f
        if p:
            need(p.stat().st_size==f.get('bytes'),'Size mismatch: '+str(path))
            need(hashlib.sha256(p.read_bytes()).hexdigest()==f.get('sha256'),'Hash mismatch: '+str(path))
    gates=d.get('gates',{})
    need(GATES<=gates.keys(),'Required gates missing')
    for name in GATES:
        gate=gates.get(name,{})
        need(gate.get('status')=='pass',name+' has not passed')
        need(gate.get('evidence') in indexed and artifact(gate.get('evidence')) is not None,name+' lacks manifested evidence')
    need(ms.get('gpu_count')==1,'Observed GPU count must be one')
    need(ms.get('native_vision') is True,'Native vision not verified')
    need(ms.get('cuda_graphs') is True,'CUDA graphs not verified')
    need(isinstance(ms.get('cold_input_tokens'),int) and ms['cold_input_tokens']>=max(200000,req.get('cold_input_tokens',200000)),'Cold context test below requirement')
    need(isinstance(ms.get('kv_tokens'),int) and ms['kv_tokens']>=max(200000,req.get('context_target',200000)),'KV allocation below requirement')
    for key in ['weight_bytes','peak_unified_memory_bytes']:
        need(isinstance(ms.get(key),int) and ms[key]>0,key+' missing')
    trials=ms.get('speed_trials',[])
    need(len(trials)>=max(5,req.get('sustained_trials',5)),'Insufficient sustained trials')
    floor=max(25,req.get('stricter_decode_minimum') or req.get('decode_target_tokens_per_second',[25])[0])
    for trial in trials:
        need(trial.get('finish_reason')=='stop' and trial.get('success') is True,'Unsuccessful sustained trial')
        need(isinstance(trial.get('decode_tokens_per_second'),(int,float)) and math.isfinite(trial['decode_tokens_per_second']) and trial['decode_tokens_per_second']>=floor,'Decode target not met')
        need(trial.get('evidence') in indexed,'Unmanifested sustained trial')
        need(isinstance(trial.get('decode_seconds'),(int,float)) and math.isfinite(trial['decode_seconds']) and trial['decode_seconds']>=30,'Sustained trial must contain at least 30 seconds of measured decode')
    matrix=ms.get('speed_matrix',[])
    need(bool(matrix),'Speed matrix empty')
    expected={(n,c) for n in [1024,8192,32768,65536,131072,200000] for c in [1,2,4,8]}
    actual={(x.get('target_input_tokens'),x.get('incoming_concurrency')) for x in matrix}
    need(expected<=actual,'Speed matrix coverage incomplete')
    need(len(actual)==len(matrix),'Duplicate speed matrix cells')
    need(any(x.get('status')=='pass' and x.get('target_input_tokens')==200000 and x.get('incoming_concurrency')==1 for x in matrix),'200k C1 speed cell must pass')
    for row in matrix:
        need(row.get('status') in ['pass','unsupported'],'Unresolved speed matrix row')
        need(row.get('evidence') in indexed,'Unmanifested speed matrix row')
        if row.get('status')=='pass':need(row.get('successful_repeats',0)>=3,'Insufficient sweep repeats')
        else:need(bool(row.get('reason')),'Unsupported row needs reason')
    # Never print matching text: failures disclose only file path and class.
    patterns={'credential':r'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)',
              'private infrastructure':r'(?:/Users/[^/\s]+/|/home/[^/\s]+/|\.tail[a-z0-9]+\.ts\.net|100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.\d+\.\d+)'}
    for p in root.rglob('*'):
        if not p.is_file() or '.git' in p.parts or p.suffix not in {'.md','.json','.yaml','.yml','.toml','.env'}:continue
        txt=p.read_text(errors='replace')
        for name,pat in patterns.items():need(not re.search(pat,txt),name+' found in '+str(p.relative_to(root)))
    return errors

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args()
    errors=verify(a.root)
    print(json.dumps({'local_evidence_pass':not errors,'errors':errors},indent=2))
    raise SystemExit(bool(errors))
