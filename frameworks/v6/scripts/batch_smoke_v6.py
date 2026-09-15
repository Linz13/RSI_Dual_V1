"""Compare generation batch=4/8 on identical sources/seeds, with no API or updates."""
import json
import time
from copy import deepcopy
from pathlib import Path
from dual_isl_train.orchestrator_v6 import DualRSIOrchestrator
from dual_isl_train.schema_v6 import VERSION, admit, prompt, render
from dual_isl_train.io import read_jsonl, atomic_json


def run(config,root):
    reports=[]
    for batch in (4,8):
        cfg=deepcopy(config);run_dir=root/f'batch_{batch}'
        cfg['training']['group_size']=8
        cfg['run']['output_dir']=str(run_dir);cfg['data']['max_records']=8
        cfg['labeling']['cache_dir']=str(run_dir/'label_cache');cfg['tts']['codec_cache_dir']=str(run_dir/'codec_cache')
        cfg['captioner']['generation']['rollout_batch_size']=batch
        cfg['tts']['generation']['synthesis_batch_size']=batch
        orch=DualRSIOrchestrator(cfg);rows=orch.prepare_data();k=cfg['training']['group_size']
        jobs=[{'id':r['id'],'audio_path':r['audio_path'],'prompt':prompt(),'caption_schema':VERSION,'group_size':k,
               'candidate_seeds':[orch._seed(0,'batch_smoke',r['id'],i) for i in range(k)]} for r in rows]
        started=time.perf_counter()
        out=orch._stage(name='batch_caption_rollout',role='captioner',action='rollout',rows=jobs,directory=run_dir/'probe')
        elapsed=time.perf_counter()-started;groups=list(read_jsonl(out.output_path))
        cs=[c for g in groups for c in g['candidates']];requests=[]
        # Fixed synthetic conditions make TTS batch=4/8 comparable even if Captioner sampling differs.
        # Transcripts are fixture generation inputs only; these jobs are never used for training.
        for r in rows:
            for i in range(k):
                cap={'transcript':r['reference_transcript'] or 'This is a batch generation check.',
                     'gender':'female' if i%2 else 'male','pitch_level':('low','medium','high')[i%3],
                     'emotion':'happy','emotion_intensity':'medium'}
                requests.append({'id':f'{r["id"]}::{i}','request':render(cap),'generation_seed':orch._seed(0,'batch_tts',r['id'],i)})
        started=time.perf_counter()
        generated=orch._stage(name='batch_tts_generation',role='tts',action='generate-audio',rows=requests,directory=run_dir/'probe/synthesis')
        audio=list(read_jsonl(generated.output_path))
        def stage_metrics(stage):
            path=Path(str(stage.output_path)+'.metrics.json')
            return json.loads(path.read_text()) if path.is_file() else {}
        reports.append({'configured_batch':batch,'caption_seconds':elapsed,'caption_candidates':len(cs),
            'caption_worker_metrics':stage_metrics(out),'tts_worker_metrics':stage_metrics(generated),
            'caption_trajectory_valid':sum(bool(c.get('trajectory_valid')) for c in cs),
            'caption_actual_batch_counts':{str(b):sum(c.get('generation_batch_size',1)==b for c in cs) for b in (1,4,8)},
            'caption_fallback_count':sum(bool(c.get('generation_batch_fallback')) for c in cs),
            'tts_seconds':time.perf_counter()-started,'tts_audios':len(audio),
            'tts_actual_batch_counts':{str(b):sum(r.get('synthesis_batch_size',1)==b for r in audio) for b in (1,4,8)}})
    report={'api_requests':0,'training_updates':0,'results':reports,
            'note':'Inspect per-stage worker metrics for per-GPU peak memory; compare seconds and actual batch sizes. No automatic training launch.'}
    atomic_json(root/'batch_comparison.json',report);return report
