"""One local model per GPU process, incremental outputs for ASR/emotion resume."""
import argparse
import json
import math
import re
from pathlib import Path
from .io import atomic_json, read_jsonl


def emotion_prediction(model,path):
    result=model.generate(input=path,granularity='utterance',extract_embedding=False)
    item=result[0] if isinstance(result,list) else result
    labels=item.get('labels',[]); scores=item.get('scores',[])
    if not labels or len(labels)!=len(scores) or not all(math.isfinite(float(v)) for v in scores):
        raise ValueError('Invalid emotion evaluator response')
    index=max(range(len(scores)),key=lambda i:float(scores[i]))
    raw=str(labels[index]).casefold()
    words=set(re.findall('[a-z]+',raw))
    aliases={'angry':'angry','anger':'angry','happy':'happy','sad':'sad','neutral':'neutral','fearful':'fearful',
             'fear':'fearful','surprised':'surprised','surprise':'surprised','disgusted':'disgusted','disgust':'disgusted',
             'other':'other','others':'other','unknown':'unknown'}
    label=next((aliases[w] for w in aliases if w in words),'unknown')
    return {'emotion':label,'evidence':{'raw_label':labels[index],'labels':labels,'scores':[float(v) for v in scores]}}


def main():
    p=argparse.ArgumentParser()
    for name in ('phase','config','input'):p.add_argument('--'+name,required=True)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text()); rows=list(read_jsonl(args.input))
    rows=[r for r in rows if not Path(r['result_path']).is_file()]
    if not rows:return
    import torch
    from .config_v6 import set_gpu_budget
    set_gpu_budget(torch,0,cfg.get('gpu_memory_gib'))
    if args.phase=='asr':
        from qwen_asr import Qwen3ASRModel
        model=Qwen3ASRModel.from_pretrained(cfg['asr']['model_path'],dtype=torch.bfloat16,
                device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,
                max_inference_batch_size=cfg['asr']['batch_size'],max_new_tokens=512)
        def run(batch):
            failed=False
            try:
                outputs=model.transcribe(audio=[r['audio_path'] for r in batch],return_time_stamps=False)
            except torch.cuda.OutOfMemoryError:
                if len(batch)==1:raise
                failed=True
            if failed:
                import gc
                gc.collect();torch.cuda.empty_cache()
                half=max(1,len(batch)//2)
                run(batch[:half]);run(batch[half:]);return
            if len(outputs)!=len(batch):raise ValueError('ASR batch correspondence lost')
            for r,out in zip(batch,outputs):
                if not isinstance(out.text,str):raise ValueError('Invalid ASR transcript')
                atomic_json(r['result_path'],{'status':'complete','audio_sha256':r['audio_sha256'],
                            'identity':r['identity'],'transcript':out.text.strip(),'asr_language':getattr(out,'language',None)})
        size=cfg['asr']['batch_size']
        for i in range(0,len(rows),size):
            run(rows[i:i+size]); print(f'ASR {min(i+size,len(rows))}/{len(rows)}',flush=True)
    elif args.phase=='emotion':
        from funasr import AutoModel
        model=AutoModel(model=cfg['emotion']['model_path'],device='cuda:0',disable_update=True)
        for i,r in enumerate(rows):
            data=emotion_prediction(model,r['audio_path'])
            atomic_json(r['result_path'],{'status':'complete','audio_sha256':r['audio_sha256'],
                        'identity':r['identity'],**data})
            print(f'Emotion {i+1}/{len(rows)}',flush=True)
    else:raise ValueError('Only ASR and emotion are enabled')


if __name__=='__main__':main()
