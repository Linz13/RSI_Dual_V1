#!/usr/bin/env python3
"""Offline V5 audio similarity comparison. No synthesis, API calls or training."""
import argparse
import base64
from collections import defaultdict
import fcntl
import gc
import html
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata

from model_adapter_utils import sha256_file, sha256_json
from speech_similarity_env import WAVLM_LARGE, WAV2VEC_XLSR, load_official_speechbertscore_module

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01'
SOURCE = RUN / 'round_001/rewards/round_001_audio_rewards.output.jsonl'
OUTPUT = ROOT / 'reports/v5_speech_similarity_round001_run01'
METRICS = ('wavlm_sbs', 'xlsr_cosine', 'wavlm_cosine', 'xlsr_sbs')
MODELS = {'wavlm': WAVLM_LARGE, 'xlsr': WAV2VEC_XLSR}
WEIGHTS_SHA = {'wavlm': 'fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d',
               'xlsr': '314340227371a608f71adcd5f0de5933824fe77e55822aa4b24dba9c1c364dcb'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n'); tmp = Path(f.name)
    try:
        tmp.chmod(0o666); os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def text_normalize(value):
    if value is None:
        return ''
    return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', str(value)).casefold()).replace('_', '')


def transcript_distance(reference, predicted):
    """Normalized character edit distance, including English; not a fresh ASR result."""
    a, b = text_normalize(reference), text_normalize(predicted)
    if not a or not b or a == 'unknown' or b == 'unknown':
        return None
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]/len(a)


def make_plan(shards):
    import soundfile as sf
    models = {}
    for name, path in MODELS.items():
        hashes = {f: sha256_file(path/f) for f in ('config.json','preprocessor_config.json','pytorch_model.bin')}
        if hashes['pytorch_model.bin'] != WEIGHTS_SHA[name]:
            raise ValueError(f'{name} weight hash mismatch')
        models[name] = {'path': str(path), 'sha256': hashes, 'layer': 14}
    audio_cache = {}
    def audio(path):
        path = str(Path(path).resolve())
        if path not in audio_cache:
            info = sf.info(path)
            if info.frames < 1 or info.duration < .1:
                raise ValueError(f'Empty/too short audio: {path}')
            audio_cache[path] = {'path': path, 'sha256': sha256_file(Path(path)), 'seconds': info.duration,
                                 'sample_rate': info.samplerate, 'channels': info.channels}
        return audio_cache[path]
    groups = []
    with SOURCE.open() as handle:
        for line in handle:
            g = json.loads(line)
            reference_text = g['reference_attributes']['semantic_content']['transcript']
            cs = []
            for c in g['candidates']:
                if not c.get('semantic_valid'):
                    continue
                predicted_text = c.get('generated_attributes', {}).get('semantic_content', {}).get('transcript', '')
                d = transcript_distance(reference_text, predicted_text)
                cs.append({'id': c['candidate_id'], 'audio': audio(c['reconstructed_audio_path']),
                           'reconstruction': c['reconstruction_score'], 'format': c['format_score'],
                           'requested_transcript': c['caption']['semantic_content']['transcript'],
                           'generated_cached_transcript': predicted_text, 'transcript_distance': d,
                           'transcript_matched': d is not None and d <= .1,
                           'caption': c['caption'], 'generated_attributes': c.get('generated_attributes', {})})
            if cs:
                groups.append({'id': g['id'], 'reference_audio': audio(g['audio_path']),
                               'reference_transcript': reference_text,
                               'reference_attributes': g['reference_attributes'],
                               'language': g['reference_attributes']['semantic_content'].get('language','unknown'),
                               'candidates': cs})
    if len(groups) != 176 or sum(len(g['candidates']) for g in groups) != 432:
        raise ValueError('Expected the fixed round_001 set: 176 groups, 432 candidates')
    buckets, costs = [[] for _ in range(shards)], [0.] * shards
    def cost(g):
        return g['reference_audio']['seconds'] + sum(c['audio']['seconds'] for c in g['candidates'])
    for g in sorted(groups, key=lambda g: (-cost(g), g['id'])):
        i = min(range(shards), key=lambda i: costs[i]); buckets[i].append(g['id']); costs[i] += cost(g)
    official = Path(metadata.distribution('discrete-speech-metrics').locate_file('discrete_speech_metrics/speechbertscore.py'))
    return {'version': 1, 'source': str(SOURCE), 'source_sha256': sha256_file(SOURCE),
            'models': models, 'groups': groups, 'shards': buckets, 'shard_audio_seconds': costs,
            'preprocessing': 'mono mean; scipy resample_poly 16kHz; checkpoint feature extractor normalization; no padding/truncation',
            'dtype': 'float32', 'attention': 'eager', 'metrics': list(METRICS),
            'code_sha256': {str(p): sha256_file(p) for p in (Path(__file__), ROOT/'speech_similarity_env.py', ROOT/'run_speech_similarity_probe.sh', official)},
            'versions': {name: metadata.version(name) for name in ('torch','transformers','scipy','numpy','soundfile','discrete-speech-metrics')}}


def read_audio(asset):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    if sha256_file(Path(asset['path'])) != asset['sha256']:
        raise ValueError('Audio changed: '+asset['path'])
    data, rate = sf.read(asset['path'], dtype='float32', always_2d=True)
    data = data.mean(axis=1)
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite waveform')
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        data = resample_poly(data, 16000//divisor, rate//divisor)
    return np.asarray(data, dtype=np.float32)


def similarity(generated, reference, block=256):
    """Exact max-cosine definition, chunked to bound the pairwise matrix memory."""
    import torch
    from torch.nn.functional import normalize, cosine_similarity
    if generated.ndim != 2 or reference.ndim != 2 or min(len(generated),len(reference)) < 1:
        raise ValueError('Nonempty [frames, dimensions] features required')
    if not torch.isfinite(generated).all() or not torch.isfinite(reference).all():
        raise ValueError('Nonfinite embeddings')
    if (generated.norm(dim=1) <= 1e-12).any() or (reference.norm(dim=1) <= 1e-12).any():
        raise ValueError('Zero-norm frame embedding')
    a, b = normalize(generated.float(), dim=1), normalize(reference.float(), dim=1)
    recall = torch.full((len(b),), -torch.inf, device=b.device)
    precision_sum = 0.
    for i in range(0,len(a),block):
        matrix = a[i:i+block] @ b.T
        precision_sum += matrix.max(dim=1).values.sum().item()
        recall = torch.maximum(recall, matrix.max(dim=0).values)
    cosine = cosine_similarity(generated.float().mean(0), reference.float().mean(0), dim=0).item()
    result = {'sbs': precision_sum/len(a), 'recall': recall.mean().item(), 'cosine': cosine}
    if not all(math.isfinite(v) and -1.00001 <= v <= 1.00001 for v in result.values()):
        raise ValueError('Invalid similarity')
    return {k: max(-1.,min(1.,v)) for k,v in result.items()}


def load_encoder(name, spec, device):
    import torch
    from transformers import WavLMModel, Wav2Vec2Model, Wav2Vec2FeatureExtractor
    cls = WavLMModel if name == 'wavlm' else Wav2Vec2Model
    model = cls.from_pretrained(spec['path'], local_files_only=True, dtype=torch.float32, attn_implementation='eager')
    model.to(device).eval().requires_grad_(False)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(spec['path'], local_files_only=True)
    def encode(wave):
        values = processor(wave, sampling_rate=16000, return_tensors='pt', padding=False).input_values.to(device)
        with torch.inference_mode():
            result = model(values, output_hidden_states=True).hidden_states[spec['layer']][0].float()
        return result
    return model, encode


def result_path(output, name, group_id, smoke=False):
    return output / ('smoke' if smoke else 'scores') / name / (sha256_json(group_id)+'.json')


def validate_result(value, plan_hash, name, group):
    if value['plan_hash'] != plan_hash or value['encoder'] != name or value['group_id'] != group['id']:
        raise ValueError('Cached result identity mismatch')
    if (len(value['candidates']) != len(group['candidates']) or
            len({c['id'] for c in value['candidates']}) != len(value['candidates']) or
            {c['id'] for c in value['candidates']} != {c['id'] for c in group['candidates']}):
        raise ValueError('Incomplete cached group')
    for c in value['candidates']:
        if not all(math.isfinite(c[k]) and -1.00001 <= c[k] <= 1.00001 for k in ('sbs','recall','cosine')):
            raise ValueError('Corrupt cached score')


def worker(output, index, budget, smoke=False):
    import torch
    from gpu_budget_exec import configure_budget
    configure_budget(torch, budget)
    torch.set_num_threads(2); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    plan = read(output/'plan.json'); ph = sha256_json(plan)
    groups = [g for g in plan['groups'] if g['id'] in plan['shards'][index]]
    if smoke:
        groups = [max(groups, key=lambda g: max([g['reference_audio']['seconds']]+[c['audio']['seconds'] for c in g['candidates']]))]
    for name, spec in plan['models'].items():
        pending = []
        for g in groups:
            path = result_path(output,name,g['id'],smoke)
            if path.exists():validate_result(read(path),ph,name,g)
            else:pending.append(g)
        if not pending:continue
        started = time.monotonic(); model, encode = load_encoder(name,spec,'cuda:0')
        print(f'[LOADED] {name} groups={len(pending)}',flush=True)
        for n,g in enumerate(pending):
            t = time.monotonic(); wave = read_audio(g['reference_audio']); ref = encode(wave)
            audit = None
            if n == 0:
                repeated = encode(wave)
                repeat_diff = (ref-repeated).abs().max().item()
                identity = similarity(ref,ref)
                if repeat_diff > 1e-5 or min(identity.values()) < .99999:
                    raise ValueError(f'Encoder reproducibility/self-score check failed: {repeat_diff}, {identity}')
                audit = {'repeat_max_abs_diff':repeat_diff,'self_scores':identity,
                         'gpu':torch.cuda.get_device_name(0),'model_eval':not model.training}
                del repeated
            cs=[]
            for c in g['candidates']:
                gen = encode(read_audio(c['audio']))
                cs.append({'id':c['id'],**similarity(gen,ref), 'reference_frames':len(ref),'generated_frames':len(gen)})
                del gen
            result={'plan_hash':ph,'encoder':name,'group_id':g['id'],'candidates':cs,'seconds':time.monotonic()-t,'audit':audit}
            validate_result(result,ph,name,g);write(result_path(output,name,g['id'],smoke),result)
            print(f'[SCORE] {name} {n+1}/{len(pending)} {g["id"]}',flush=True)
            del ref
        write(output/('smoke' if smoke else 'scores')/name/f'worker_{index:02d}.timing.json',
              {'elapsed_seconds':time.monotonic()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
        del model,encode;gc.collect();torch.cuda.empty_cache()


def launch(output,gpus,budget,smoke):
    processes=[];logs=[];stage='smoke' if smoke else 'full'
    (output/'logs').mkdir(exist_ok=True)
    try:
        for i,gpu in enumerate(gpus):
            log=(output/'logs'/f'{stage}_{i:02d}.log').open('a');logs.append(log)
            env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            for k in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(k,None)
            command=[sys.executable,str(Path(__file__).resolve()),'worker','--output-dir',str(output),'--index',str(i),'--gpu-memory-gib',str(budget)]
            if smoke:command.append('--worker-smoke')
            processes.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        next_print=0
        while True:
            codes=[p.poll() for p in processes]
            if any(c not in (0,None) for c in codes):raise RuntimeError(f'{stage} worker failed {codes}; see {output}/logs; rerun same command to resume')
            if time.monotonic()>=next_print:
                count=sum(1 for p in (output/('smoke' if smoke else 'scores')).glob('*/*.json') if '.timing.' not in p.name)
                print(f'[PROGRESS] {stage}: {count} completed encoder/group results',flush=True);next_print=time.monotonic()+30
            if all(c==0 for c in codes):break
            time.sleep(1)
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for log in logs:log.close()


def distribution(xs):
    if not xs:return {'count':0}
    import numpy as np
    return {'count':len(xs),'mean':statistics.mean(xs),'median':statistics.median(xs),
            'p10':float(np.quantile(xs,.1)),'p90':float(np.quantile(xs,.9))}


def winners(cs,key):
    top=max(c[key] for c in cs)
    return {c['id'] for c in cs if abs(c[key]-top)<=1e-7}


def subset_report(groups,matched):
    selected=[[c for c in g['candidates'] if not matched or c['transcript_matched']] for g in groups]
    comparable=[cs for cs in selected if len(cs)>=2]
    report={'candidates':sum(map(len,selected)),'comparable_groups':len(comparable),'metrics':{}}
    for metric in METRICS:
        gaps=[max(c[metric] for c in cs)-min(c[metric] for c in cs) for cs in comparable]
        overlaps=sum(bool(winners(cs,metric)&winners(cs,'reconstruction')) for cs in comparable)
        report['metrics'][metric]={'scores':distribution([c[metric] for cs in selected for c in cs]),
            'group_spans':distribution(gaps),'reference_reward_top_overlap_groups':overlaps}
    report['main_metrics_disagree_groups']=sum(not (winners(cs,'wavlm_sbs')&winners(cs,'xlsr_cosine')) for cs in comparable)
    return report


def listening_page(output,groups,ph):
    eligible=[]
    for g in groups:
        cs=[c for c in g['candidates'] if c['transcript_matched']]
        if len(cs)>=2:eligible.append({**g,'candidates':cs})
    eligible.sort(key=lambda g:sha256_json('listen-random:'+g['id']))
    random_groups=eligible[:10];used={g['id'] for g in random_groups}
    disagreements=[g for g in eligible if g['id'] not in used and not (winners(g['candidates'],'wavlm_sbs')&winners(g['candidates'],'xlsr_cosine'))]
    choices=[(g,'random') for g in random_groups]+[(g,'disagreement') for g in disagreements[:10]]
    for g in eligible:
        if len(choices)>=20:break
        if g['id'] not in {x['id'] for x,_ in choices}:choices.append((g,'random_fill'))
    manifest=[];sections=[]
    def audio(asset):
        import io
        import soundfile as sf
        wave=read_audio(asset);buffer=io.BytesIO();sf.write(buffer,wave,16000,format='WAV',subtype='PCM_16')
        return '<audio controls preload="none" src="data:audio/wav;base64,'+base64.b64encode(buffer.getvalue()).decode()+'"></audio>'
    for i,(g,category) in enumerate(choices):
        cs=sorted(g['candidates'],key=lambda c:sha256_json('blind:'+c['id']))
        manifest.append({'group_id':g['id'],'category':category,'candidate_ids':[c['id'] for c in cs]})
        section=[f'<section><h2>第 {i+1} 组</h2><p>原音频</p>',audio(g['reference_audio'])]
        options=[]
        for j,c in enumerate(cs):
            label=chr(65+j);section.extend([f'<p>候选 {label}</p>',audio(c['audio'])])
            options.append(f'<option value="{html.escape(c["id"],quote=True)}">{label} 最接近</option>')
        section.append(f'<p>哪条最接近原音频的音色、情绪和表达方式？</p><select data-group="{html.escape(g["id"],quote=True)}"><option value="">未选择</option>'+''.join(options)+'<option value="tie">相近，无法分出最好</option><option value="content_mismatch">内容不一致，暂不比较风格</option><option value="unclear">无法判断</option></select>')
        section.append('<details><summary>选择后查看分数与转录</summary><pre>'+html.escape(json.dumps({'id':g['id'],'sampling':category,'reference_transcript':g['reference_transcript'],'candidates':[{'label':chr(65+j),'id':c['id'],**{k:c[k] for k in METRICS},'cached_transcript':c['generated_cached_transcript']} for j,c in enumerate(cs)]},ensure_ascii=False,indent=2))+'</pre></details></section>')
        sections.append(''.join(section))
    write(output/'listening_manifest.json',{'plan_hash':ph,'groups':manifest})
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>语音相似度盲听</title>
<style>body{max-width:850px;margin:32px auto;font:17px/1.6 system-ui;padding:16px;background:#f5f6f8}section{background:white;padding:22px;margin:24px 0;border-radius:12px}audio{width:100%}pre{white-space:pre-wrap;font-size:13px}select,button{font:inherit;padding:10px}details{margin-top:20px}</style>
<h1>语音相似度盲听</h1><p>先听原音频，再听候选，选择声音表现最接近的一条。请先选择，再展开分数，避免分数影响判断。样本由随机组和指标分歧组组成，不代表总体准确率。</p><button id="export">导出试听选择</button><span id="status"></span>
'''+''.join(sections)+'''<script>
const planHash=PLAN_HASH;const key='speech-sim-votes-'+planHash;
let votes={};try{votes=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function save(){try{localStorage.setItem(key,JSON.stringify(votes))}catch(e){}document.getElementById('status').textContent=' 已选 '+Object.values(votes).filter(Boolean).length+' 组';}
document.querySelectorAll('select[data-group]').forEach(s=>{s.value=votes[s.dataset.group]||'';s.addEventListener('change',()=>{votes[s.dataset.group]=s.value;save()})});save();
document.getElementById('export').onclick=()=>{const b=new Blob([JSON.stringify({plan_hash:planHash,votes:votes},null,2)],{type:'application/json'});const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download='listening_votes.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
</script></html>'''.replace('PLAN_HASH',json.dumps(ph))
    (output/'listen.html').write_text(page)
    return len(choices)


def analyze(output,votes_path=None):
    plan=read(output/'plan.json');ph=sha256_json(plan);groups=[]
    for g in plan['groups']:
        cs={c['id']:dict(c) for c in g['candidates']}
        for name in MODELS:
            value=read(result_path(output,name,g['id']));validate_result(value,ph,name,g)
            for c in value['candidates']:
                for key in ('sbs','recall','cosine'):cs[c['id']][name+'_'+key]=c[key]
        groups.append({**g,'candidates':list(cs.values())})
    summary={'complete':True,'plan_hash':ph,'all':subset_report(groups,False),
             'transcript_matched':subset_report(groups,True),
             'by_language':{lang:subset_report([g for g in groups if g['language']==lang],True) for lang in sorted({g['language'] for g in groups})},
             'notes':['Scores are raw cosine-based values in [-1,1]; reward mapping would be (1+score)/2.',
                      'Transcript filter uses cached model labels, normalized character edit distance <=0.1, not a fresh ASR evaluation.',
                      'Agreement with the existing reconstruction reward is not human correctness.',
                      'Both encoders use hidden_states[14], float32, eager attention, checkpoint waveform normalization.',
                      'SpeechBERTScore precision formula follows upstream; normalized input differs from upstream raw-wave demo.',
                      'No training, synthesis or API calls. Single-rollout comparisons do not establish training gains.']}
    summary['listening_groups']=listening_page(output,groups,ph)
    if votes_path:
        votes=read(votes_path)
        if votes['plan_hash']!=ph:raise ValueError('Votes belong to another experiment')
        manifest={g['group_id']:g for g in read(output/'listening_manifest.json')['groups']}
        human={m:{'groups':0,'selected_in_metric_top':0} for m in METRICS}
        by_category={}
        by_group={g['id']:g for g in groups}
        for gid,choice in votes['votes'].items():
            if gid not in manifest:raise ValueError('Unknown listening group')
            if choice in ('','tie','content_mismatch','unclear'):continue
            if choice not in manifest[gid]['candidate_ids']:raise ValueError('Invalid listening choice')
            cs=[c for c in by_group[gid]['candidates'] if c['id'] in manifest[gid]['candidate_ids']]
            category=manifest[gid]['category']
            bucket=by_category.setdefault(category,{m:{'groups':0,'selected_in_metric_top':0} for m in METRICS})
            for m in METRICS:
                human[m]['groups']+=1;human[m]['selected_in_metric_top']+=choice in winners(cs,m)
                bucket[m]['groups']+=1;bucket[m]['selected_in_metric_top']+=choice in winners(cs,m)
        summary['human_listening']=human
        summary['human_listening_by_sampling']=by_category
    write(output/'summary.json',summary);write(output/'candidates.json',groups)
    lines=['# 第二轮音频相似度实验','',f'完成 {sum(len(g["candidates"]) for g in groups)} 对音频、两个编码器的四种评分。','',
           'SpeechBERTScore 使用逐帧最大 cosine 的平均值（precision）；cosine 基线对同层特征先按时间平均。',
           '两种方法使用同一编码器时共享同一份特征；层固定为 14。结果为原始 [-1,1] 分数。','']
    for name in ('all','transcript_matched'):
        r=summary[name];lines += [f'## {name}',f'候选 {r["candidates"]} 条，可比较组 {r["comparable_groups"]} 个。','',
          '| 指标 | 平均分 | 组内跨度均值 | 组内跨度中位数 | 与属性重建第一名重合组数 |','|---|---:|---:|---:|---:|']
        fmt=lambda x:f'{x:.6f}' if x is not None else 'N/A'
        for metric,v in r['metrics'].items():
            lines.append(f'| {metric} | {fmt(v["scores"].get("mean"))} | {fmt(v["group_spans"].get("mean"))} | {fmt(v["group_spans"].get("median"))} | {v["reference_reward_top_overlap_groups"]} |')
        lines += ['',f'两种主指标第一名不重合：{r["main_metrics_disagree_groups"]} 组。','']
    lines += ['请用 listen.html 盲听，并导出选择。与属性重建分一致、或者组内分差更大，都不能单独证明指标更可靠。',
              '转录过滤基于缓存打标；仅作初筛。试听请排除内容不一致的组。随机组和分歧组混合抽样，不能把试听一致率当作总体准确率。']
    if 'human_listening' in summary:lines += ['',json.dumps(summary['human_listening'],ensure_ascii=False,indent=2)]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print('[REPORT]',output/'report.md',flush=True);print('[LISTEN]',output/'listen.html',flush=True)


def cpu_smoke(output):
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.manual_seed(42)
    official=load_official_speechbertscore_module()
    a=torch.randn(29,17);b=torch.randn(41,17)
    expected=official.bert_score(a,b);actual=similarity(a,b,7)
    assert abs(actual['sbs']-expected[0])<1e-6 and abs(actual['recall']-expected[1])<1e-6
    with SOURCE.open() as f:
        g=next(json.loads(line) for line in f if 'reconstructed_audio_path' in line)
    c=next(c for c in g['candidates'] if c.get('semantic_valid'))
    def asset(path):return {'path':path,'sha256':sha256_file(Path(path))}
    ref=read_audio(asset(g['audio_path']))[:32000];gen=read_audio(asset(c['reconstructed_audio_path']))[:32000]
    checks={}
    for name,path in MODELS.items():
        model,encode=load_encoder(name,{'path':str(path),'layer':14},'cpu')
        a,b=encode(ref),encode(gen);result=similarity(b,a);target=official.bert_score(b,a)
        assert abs(result['sbs']-target[0])<1e-6 and abs(result['recall']-target[1])<1e-6
        assert min(similarity(a,a).values())>.99999
        checks[name]={'reference_shape':list(a.shape),'generated_shape':list(b.shape),'scores':result,'formula_matches_upstream':True}
        del model,encode;gc.collect()
    write(output/'cpu_smoke.json',{'passed':True,'gpu_execution':'not_performed','scope':'two-second cropped inputs for implementation check only','checks':checks})
    print(json.dumps(checks,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('check','cpu-smoke','smoke','run','worker','analyze'))
    p.add_argument('--output-dir',type=Path,default=OUTPUT);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--gpu-memory-gib',type=float,default=26);p.add_argument('--index',type=int)
    p.add_argument('--worker-smoke',action='store_true');p.add_argument('--votes',type=Path)
    args=p.parse_args();os.umask(0);out=args.output_dir.resolve()
    if args.mode=='worker':worker(out,args.index,args.gpu_memory_gib,args.worker_smoke);return
    out.mkdir(parents=True,exist_ok=True)
    if args.mode=='cpu-smoke':cpu_smoke(out);return
    with (out/'probe.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='analyze':analyze(out,args.votes);return
        gpus=args.gpus.split(',')
        if not all(x.isdigit() for x in gpus) or len(set(gpus))!=len(gpus) or not 1<=len(gpus)<=8 or args.gpu_memory_gib<=0:
            raise ValueError('Use 1-8 distinct physical GPU IDs and a positive memory budget')
        plan=make_plan(len(gpus))
        print(json.dumps({'cpu_preflight':'passed','gpu_execution':'not_performed','groups':176,'candidates':432,
                         'transcript_matched_candidates':sum(c['transcript_matched'] for g in plan['groups'] for c in g['candidates']),
                         'workers':len(gpus),'api_calls':0,'output_dir':str(out)},indent=2),flush=True)
        if args.mode=='check':return
        if (out/'plan.json').exists() and read(out/'plan.json')!=plan:
            raise ValueError('Experiment identity changed. Use a new output directory; old results preserved.')
        write(out/'plan.json',plan)
        from v5_tts_dsd_eval import check_gpu_memory
        check_gpu_memory(gpus,args.gpu_memory_gib)
        launch(out,gpus,args.gpu_memory_gib,True)
        if args.mode=='run':launch(out,gpus,args.gpu_memory_gib,False);analyze(out,args.votes)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(130))
    main()
