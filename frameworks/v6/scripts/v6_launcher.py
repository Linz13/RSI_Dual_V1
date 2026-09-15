"""V6 launch contract: CPU checks do not perform CUDA inference or call APIs."""
import argparse
import atexit
import importlib
import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dual_isl_train.config import load_config, public_config, validate_config, ENV_OVERRIDES, INTEGER_ENV_OVERRIDES
from dual_isl_train.io import load_yaml, atomic_json
from dual_isl_train.data_v6 import build_records

ROOT=Path(__file__).resolve().parents[1]


def share(root):
    if os.getenv('DUALISL_SHARED_WRITABLE')!='1':return
    failures=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        for p in [Path(directory),*(Path(directory)/n for n in files)]:
            try:
                if p.is_symlink():continue
                bits=0o777 if p.is_dir() else 0o666
                mode=stat.S_IMODE(p.stat().st_mode)
                if mode|bits!=mode:p.chmod(mode|bits)
            except OSError:failures+=1
    if failures:print(f'Shared permission updates failed for {failures} paths',file=sys.stderr)


def configuration(mode,root):
    frozen=root/'resolved_config.yaml'
    if mode=='resume' or (mode in ('prepare','train') and frozen.exists()):
        cfg=load_yaml(frozen)
        if Path(cfg['run']['output_dir']).resolve()!=root:raise ValueError('Resume requires original run directory')
        if len(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(','))!=cfg['distributed']['world_size']:
            raise ValueError('Resume requires same number of visible GPUs; config is frozen')
        validate_config(cfg);return cfg
    cfg=public_config(load_config(os.getenv('DUALISL_V6_CONFIG',ROOT/'configs/v6.yaml')))
    cfg['run']['output_dir']=str(root)
    cfg['labeling']['cache_dir']=str(root/'label_cache');cfg['tts']['codec_cache_dir']=str(root/'codec_cache')
    cfg['training']['group_size']=int(os.getenv('DUALISL_GROUP_SIZE',8))
    if mode=='gpu-smoke':
        cfg['training']['rounds']=1;cfg['data']['max_records']=int(os.getenv('DUALISL_SMOKE_RECORDS',16));cfg['run']['smoke']=True
    batch=int(os.getenv('DUALISL_INFERENCE_BATCH',8))
    cfg['captioner']['generation']['rollout_batch_size']=int(os.getenv('DUALISL_CAPTION_BATCH',batch))
    cfg['tts']['generation']['synthesis_batch_size']=int(os.getenv('DUALISL_TTS_BATCH',batch))
    cfg['labeling']['api_workers']=int(os.getenv('DUALISL_API_WORKERS',cfg['labeling']['api_workers']))
    if os.getenv('DUALISL_GPU_MEMORY_GIB'):cfg['gpu_memory_gib']=float(os.environ['DUALISL_GPU_MEMORY_GIB'])
    ids=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7').split(',')
    if len(set(ids))!=len(ids) or any(not x.strip() for x in ids):raise ValueError('GPU list must contain unique nonempty IDs')
    cfg['distributed'].update(enabled=len(ids)>1,world_size=len(ids))
    validate_config(cfg);return cfg


def preflight(cfg):
    roles={role:cfg[role] for role in ('captioner','tts')}
    roles.update({role:cfg['labeling'][role] for role in ('asr','emotion')})
    required={f'{role}.{k}':v[k] for role,v in roles.items() for k in ('python','model_path')}
    required.update(tokenizer=cfg['tts']['tokenizer_path'],labeling_source=cfg['labeling']['source_root'])
    missing={k:p for k,p in required.items() if not Path(p).exists()}
    if missing:raise ValueError('Missing runtime resources: '+json.dumps(missing))
    modules={'captioner':['torch','transformers','peft','yaml','jsonschema','soundfile'],
             'tts':['torch','qwen_tts','peft','yaml','jsonschema','soundfile'],
             'asr':['torch','qwen_asr','soundfile'],'emotion':['torch','funasr','soundfile']}
    def check(role):
        code='import importlib.util; assert all(importlib.util.find_spec(m) is not None for m in '+repr(modules[role])+')'
        p=subprocess.run([roles[role]['python'],'-c',code],capture_output=True,text=True)
        return role,p.returncode==0
    with ThreadPoolExecutor(max_workers=4) as pool:deps=dict(pool.map(check,roles))
    if not all(deps.values()):raise ValueError('Missing runtime dependencies '+str(deps))
    sys.path.insert(0,str(Path(cfg['labeling']['source_root'])/'Experiment'))
    credentials=importlib.import_module('labeling2.run_with_local_api_key')
    for model in ('gemini','qwen35'):
        if not os.getenv(credentials.PROVIDERS[model][0]) and not credentials.load_key(model):
            raise ValueError('Missing credential for '+model)
    rows,data=build_records(cfg['data'])
    return {'cpu_preflight':'passed','gpu_execution':'not_performed','api_requests':0,
            'audio_records':len(rows),'source_records':data['source_records'],
            'world_size':cfg['distributed']['world_size'],'rounds':cfg['training']['rounds'],
            'group_size':cfg['training']['group_size'],'caption_batch':cfg['captioner']['generation']['rollout_batch_size'],
            'tts_batch':cfg['tts']['generation']['synthesis_batch_size'],'gpu_memory_gib':cfg.get('gpu_memory_gib'),
            'updates':['caption_grpo','tts_sft'],'paired_anchor':False,'dependencies':deps,'run_dir':cfg['run']['output_dir']}


def snapshot(root):
    out={}
    for name in ('status','latest'):
        p=root/(name+'.json');out[name]=json.loads(p.read_text()) if p.exists() else None
    out['completed_rounds']=len(list(root.glob('round_*/commit.json')))
    state=root/'run_state.json'
    if state.exists():
        data=json.loads(state.read_text());out['stages']={n:s['status'] for n,s in data.get('stages',{}).items()}
    return out


def dashboard(root):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import html
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body=('<html><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>DualRSI V6</title>'
                  '<h1>DualRSI V6 训练进度</h1><pre>'+html.escape(json.dumps(snapshot(root),ensure_ascii=False,indent=2))+'</pre></html>').encode()
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    port=int(os.getenv('DUALISL_DASHBOARD_PORT',6007));print(f'Dashboard http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('0.0.0.0',port),Handler).serve_forever()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('check','prepare','batch-smoke','gpu-smoke','train','resume','verify','status','dashboard'));p.add_argument('run_dir',type=Path)
    args=p.parse_args();root=args.run_dir.resolve();mode=args.mode
    if mode=='status':print(json.dumps(snapshot(root),ensure_ascii=False,indent=2));return
    if mode=='dashboard':dashboard(root);return
    if mode=='verify':
        from scripts.verify_v6 import verify
        report=verify(root);print(json.dumps(report,ensure_ascii=False,indent=2));raise SystemExit(0 if report['ok'] else 1)
    if mode in ('gpu-smoke','batch-smoke') and (root/'run_state.json').exists():raise ValueError('Run exists; use resume or new directory')
    if mode=='train' and list(root.glob('round_*')):raise ValueError('Training already started; use resume')
    cfg=configuration(mode,root)
    for name in (*ENV_OVERRIDES,*INTEGER_ENV_OVERRIDES):os.environ.pop(name,None)
    print(json.dumps(preflight(cfg),ensure_ascii=False,indent=2),flush=True)
    if mode=='check':return
    root.mkdir(parents=True,exist_ok=True);atexit.register(share,root)
    # Kernel advisory lock avoids two drivers updating the same shared run concurrently.
    import fcntl
    with (root/'.driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if mode=='batch-smoke':
            from scripts.batch_smoke_v6 import run
            print(json.dumps(run(cfg,root),ensure_ascii=False,indent=2));return
        from dual_isl_train.orchestrator_v6 import DualRSIOrchestrator
        orch=DualRSIOrchestrator(cfg)
        result=orch.train(resume_only=mode=='resume',prepare_only=mode=='prepare')
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if mode=='prepare':return
        if cfg['run'].get('smoke'):
            for role in ('captioner','tts'):
                checkpoint=result[role]
                orch._stage(name=f'smoke_reload_{role}',role=role,action='preflight',rows=[],directory=root/'reload',
                            checkpoint_in=checkpoint,distributed=False)
        from scripts.verify_v6 import verify
        report=verify(root);atomic_json(root/'verification.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2));raise SystemExit(0 if report['ok'] else 1)


if __name__=='__main__':main()
