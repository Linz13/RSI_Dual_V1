#!/usr/bin/env python3
"""Frozen TTS likelihood probe and offline reward-weight comparison; never trains."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics as st
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parent
FRAMEWORK = ROOT.parent / 'DualISL_Train_RewardV5_LabelRobust'
RUN = FRAMEWORK / 'runs/midasheng_v5_label_robust_10rounds_8gpu_run01'
OUT = ROOT / 'reports/v5_loss_reward_probe_round001_run01'
PYTHON = ROOT.parents[1] / 'miniconda3/envs/qwen3-tts/bin/python'
sys.path.insert(0, str(FRAMEWORK))
from dual_isl_train.render import render_qwen_request
from model_adapter_utils import sha256_file, sha256_json
from midasheng_v4_v5_eval import write_json
from v5_tts_dsd_eval import checkpoint_hash, check_gpu_memory, parse_gpus, stop_processes


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def normalized_text(text):
    return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', text).casefold()).replace('_', '')


def fixed_request(caption, reference):
    copy = deepcopy(caption)
    copy['semantic_content'] = deepcopy(reference['semantic_content'])
    return render_qwen_request(copy)


def reference_controls(reference):
    """Skip contradictory cached labels; change one field without inventing repairs."""
    gender = reference['speaker_profile']['gender']
    emotion = reference['paralinguistic']['emotion']
    if gender not in ('male', 'female') or emotion not in (
            'neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted'):
        return None
    try:
        original = render_qwen_request(reference)
        wrong_gender = deepcopy(reference)
        wrong_gender['speaker_profile']['gender'] = 'female' if gender == 'male' else 'male'
        gender_request = render_qwen_request(wrong_gender)
        wrong_emotion = deepcopy(reference)
        # Avoid introducing neutral + high intensity, which is schema-invalid.
        wrong_emotion['paralinguistic']['emotion'] = 'sad' if emotion == 'angry' else 'angry'
        emotion_request = render_qwen_request(wrong_emotion)
    except ValueError:
        return None
    return {'reference': original, 'wrong_gender': gender_request, 'wrong_emotion': emotion_request}


def make_plan():
    import yaml
    source = RUN / 'round_001/rewards/round_001_audio_rewards.output.jsonl'
    commit = read(RUN / 'round_001/commit.json')
    checkpoint = Path(commit['input_tts']['path'])
    if checkpoint_hash(checkpoint) != commit['input_tts']['sha256']:
        raise ValueError('Round-start TTS checkpoint hash mismatch')
    config = yaml.safe_load((RUN / 'resolved_config.yaml').read_text())
    config = {'run': {'seed': 42}, 'tts': deepcopy(config['tts'])}
    config['tts'].update(device='cuda:0', gradient_checkpointing=False)
    weight = float(config['tts']['training'].get('sub_talker_weight', .3))
    codecs = {r['id']: r for r in rows(RUN / 'prepared/audio_pool_codecs.output.jsonl')}
    groups = []
    # Strip rollout tensors immediately to avoid retaining GRPO trajectories in RAM.
    with source.open() as handle:
        for line in handle:
            group = json.loads(line)
            cs = [{k: c[k] for k in ('candidate_id', 'caption', 'reconstruction_score', 'format_score')}
                  for c in group['candidates'] if c.get('semantic_valid')]
            if cs:
                groups.append({'id': group['id'], 'audio_path': group['audio_path'],
                               'reference': group['reference_attributes'], 'candidates': cs})
    ordered = sorted(groups, key=lambda g: sha256_json(g['id']))
    calibration = {g['id'] for g in ordered[:max(1, len(ordered)//4)]}
    controls = {}
    for g in ordered:
        control = reference_controls(g['reference']) if g['id'] not in calibration else None
        if control:
            controls[g['id']] = control
            if len(controls) == 32:
                break
    jobs = []
    for g in groups:
        ref = g['reference']
        text = str(ref['semantic_content'].get('transcript', '')).strip()
        if not text or text.casefold() == 'unknown':
            raise ValueError('No reference transcript for group ' + g['id'])
        entry = codecs[g['id']]
        codec_path = Path(entry['codec_path'])
        codec = read(codec_path)
        codes = codec['codec_codes']
        if (Path(codec['audio_path']).resolve() != Path(g['audio_path']).resolve()
                or sha256_file(Path(g['audio_path'])) != codec['audio_sha256']
                or not codes or any(len(frame) != 16 for frame in codes)):
            raise ValueError('Original audio / cached codec mismatch: ' + g['id'])
        common = {'group_id': g['id'], 'codec_path': str(codec_path), 'codec_sha256': sha256_file(codec_path),
                  'frames': len(codes), 'codebooks': 16, 'split': 'calibration' if g['id'] in calibration else 'analysis'}
        def add(job_id, kind, request, **extra):
            jobs.append({**common, 'id': job_id, 'kind': kind, 'request': request, **extra})
        for c in g['candidates']:
            add(c['candidate_id'], 'candidate', fixed_request(c['caption'], ref),
                reconstruction=c['reconstruction_score'], format=c['format_score'],
                original_transcript=c['caption']['semantic_content']['transcript'],
                transcript_exact_normalized=normalized_text(c['caption']['semantic_content']['transcript']) == normalized_text(text))
        # Only text/language are needed here. Cached attribute labels need not
        # satisfy the cross-field constraints required of a synthesis instruction.
        add(g['id'] + '::text_only', 'text_only', {**jobs[-1]['request'], 'instruct': ''})
        if g['id'] in controls:
            for kind, request in controls[g['id']].items():
                add(g['id'] + '::' + kind, kind, request)
    for group_jobs in group_by(jobs).values():
        assert len({(j['request']['text'], j['request']['language'], j['codec_sha256'], j['frames']) for j in group_jobs}) == 1
    # Greedy length balancing; single-sample forward passes on each card.
    shards, costs = [[] for _ in range(8)], [0] * 8
    for j in sorted(jobs, key=lambda j: (-j['frames'], j['id'])):
        i = min(range(8), key=lambda x: costs[x])
        shards[i].append(j['id'])
        costs[i] += j['frames']
    code = list((FRAMEWORK / 'dual_isl_train').rglob('*.py')) + [Path(__file__), ROOT / 'run_v5_loss_reward_probe.sh',
           ROOT / 'model_adapter_utils.py', ROOT / 'gpu_budget_exec.py']
    return {'version': 'v5-loss-probe-v1', 'round_index': 1, 'checkpoint': str(checkpoint),
            'checkpoint_sha256': commit['input_tts']['sha256'], 'source_path': str(source),
            'source_sha256': sha256_file(source), 'config': config, 'sub_weight': weight,
            'jobs': jobs, 'shards': shards, 'shard_frames': costs,
            'controls': sorted(controls), 'calibration_groups': sorted(calibration),
            'code_hashes': {str(p): sha256_file(p) for p in sorted(code)},
            'notes': ['Reference transcript/attributes are cached model labels, not human-verified GT.',
                      'Reconstruction rewards reuse historical audio generated from original candidate transcripts.',
                      'This is a diagnostic weight simulation, not evaluation of the proposed complete new loop.',
                      'Calibration and analysis groups are disjoint; neither uses benchmark labels.',
                      'Attribute-replacement controls use only schema-valid cached references; candidates are not excluded on this basis.',
                      'Uses the round-start TTS checkpoint, not the model trained on these second-round samples.']}


def group_by(jobs):
    out = defaultdict(list)
    for j in jobs:
        out[j['group_id']].append(j)
    return out


def load_results(path, plan_hash):
    if not path.exists():
        return {}
    raw = path.read_bytes(); lines = raw.splitlines(keepends=True); result = {}
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            if n == len(lines)-1 and not line.endswith(b'\n'):
                raise ValueError('Incomplete last result; rerun worker to repair')
            raise
        if value['plan_hash'] != plan_hash or value['id'] in result:
            raise ValueError('Result identity mismatch or duplicate')
        if not all(math.isfinite(value[k]) for k in ('loss', 'main_nll', 'sub_nll')):
            raise ValueError('Nonfinite loss')
        result[value['id']] = value
    return result


def worker(output, plan, plan_hash, index, budget, smoke):
    import torch
    from gpu_budget_exec import configure_budget
    from dual_isl_train.workers.qwen_voice_design import QwenVoiceDesignWorker
    destination = output / ('smoke' if smoke else 'scores')
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f'{index:02d}.jsonl'
    if path.exists():
        raw = path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            last = raw.rsplit(b'\n', 1)[-1]
            try:
                json.loads(last)
            except ValueError:
                backup = path.with_suffix('.interrupted.bin')
                backup.write_bytes(raw)
                path.write_bytes(raw[:len(raw)-len(last)])
            else:
                with path.open('ab') as f:
                    f.write(b'\n')
    ids = plan['shards'][index]
    if smoke:
        ids = ids[:1]  # Longest assigned example: a useful memory check.
    all_jobs = {j['id']: j for j in plan['jobs']}
    old = load_results(path, plan_hash)
    if not set(old) <= set(ids):
        raise ValueError('Unexpected shard job IDs')
    pending = [all_jobs[i] for i in ids if i not in old]
    if not pending:
        print('[RESUME] all assigned scores already complete', flush=True)
        return
    configure_budget(torch, budget)
    started = time.monotonic()
    # Both adapters are explicitly loaded from the same checkpoint: score_reconstruction
    # uses reference=True and would otherwise silently score the base model.
    model = QwenVoiceDesignWorker(plan['config'], plan['checkpoint'], plan['checkpoint'])
    model.policy.eval()
    model.policy.requires_grad_(False)
    if not model.has_reference_adapter or not model.adapter_audits['policy_reference']['exact']:
        raise RuntimeError('Scoring checkpoint was not loaded exactly')
    print(f'[LOADED] seconds={time.monotonic()-started:.1f}; jobs={len(pending)}', flush=True)
    with path.open('a', buffering=1) as f:
        for n, j in enumerate(pending):
            if sha256_file(Path(j['codec_path'])) != j['codec_sha256']:
                raise ValueError('Codec file changed')
            job = {'candidate_id': j['id'], 'request': j['request'], 'codec_path': j['codec_path']}
            before = time.monotonic()
            with torch.inference_mode():
                result = model.score_reconstruction([job])[0]
                repeat = model.score_reconstruction([job])[0] if n == 0 else None
            if result['codec_frames'] != j['frames'] or result['codebooks'] != 16:
                raise ValueError('Scored token mask/shape mismatch')
            value = {'id': j['id'], 'plan_hash': plan_hash, 'loss': -result['tts_target_logprob'],
                     'main_nll': -result['tts_main_logprob'], 'sub_nll': -result['tts_sub_logprob'],
                     'frames': j['frames'], 'seconds': time.monotonic()-before}
            if not all(math.isfinite(value[k]) for k in ('loss', 'main_nll', 'sub_nll')):
                raise ValueError('Nonfinite loss')
            if repeat is not None:
                delta = abs(repeat['tts_target_logprob']-result['tts_target_logprob'])
                if delta > 1e-5:
                    raise RuntimeError(f'Repeated deterministic score changed by {delta}')
                write_json(destination / f'{index:02d}.audit.json', {
                    'plan_hash': plan_hash, 'checkpoint': plan['checkpoint'], 'adapter_audits': model.adapter_audits,
                    'repeat_abs_diff': delta, 'eval_mode': not model.policy.training, 'grad_enabled': False,
                    'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})
            f.write(json.dumps(value) + '\n')
            print(f'[SCORE] {len(old)+n+1}/{len(ids)} {j["id"]} loss={value["loss"]:.6f}', flush=True)
    write_json(destination / f'{index:02d}.timing.json', {'elapsed_seconds': time.monotonic()-started,
               'new_scores': len(pending), 'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3})


def environment(gpu):
    env = dict(os.environ)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
    return env


def launch(output, plan, gpus, budget, smoke):
    processes, logs = [], []
    folder = output / 'logs'; folder.mkdir(exist_ok=True)
    stage = 'smoke' if smoke else 'full'
    try:
        for i, gpu in enumerate(gpus):
            path = folder / f'{stage}_{i:02d}_gpu{gpu}.log'
            log = path.open('a', buffering=1); logs.append(log)
            cmd = [str(PYTHON), str(Path(__file__).resolve()), 'worker', '--output-dir', str(output),
                   '--index', str(i), '--gpu-memory-gib', str(budget)]
            if smoke:
                cmd.append('--worker-smoke')
            processes.append(subprocess.Popen(cmd, env=environment(gpu), stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True))
            print(f'[START] {stage} GPU={gpu} log={path}', flush=True)
        next_print = 0
        while True:
            codes = [p.poll() for p in processes]
            if any(c not in (None, 0) for c in codes):
                raise RuntimeError(f'{stage} worker failed: {codes}; inspect logs/. Rerun to resume.')
            if time.monotonic() >= next_print:
                paths = (output / ('smoke' if smoke else 'scores')).glob('*.jsonl')
                n = sum(sum(1 for line in p.open() if line.endswith('\n') and line.strip()) for p in paths)
                print(f'[PROGRESS] {stage} {n}/{8 if smoke else len(plan["jobs"])}', flush=True)
                next_print = time.monotonic()+30
            if all(c == 0 for c in codes):
                break
            time.sleep(1)
    finally:
        stop_processes(processes)
        for log in logs:
            log.close()


def quantile(values, p):
    values = sorted(values); pos = (len(values)-1)*p; a = int(pos); b = min(a+1,len(values)-1)
    return values[a] + (values[b]-values[a])*(pos-a)


def distribution(values):
    return {'count': len(values), 'mean': st.fmean(values), 'median': st.median(values),
            'p10': quantile(values,.1), 'p90': quantile(values,.9)} if values else {'count': 0}


def sigmoid_reward(loss, m, s):
    if s <= 0:
        raise ValueError('Positive sigmoid scale required')
    x = (loss-m)/s
    return math.exp(-x)/(1+math.exp(-x)) if x >= 0 else 1/(1+math.exp(x))


def rank_winners(cs, key):
    maximum = max(c[key] for c in cs)
    return {c['id'] for c in cs if abs(c[key]-maximum) <= 1e-12}


def analyze(output, plan):
    plan_hash = sha256_json(plan)
    scores = {}
    for i, ids in enumerate(plan['shards']):
        found = load_results(output / 'scores' / f'{i:02d}.jsonl', plan_hash)
        if set(found) != set(ids):
            raise ValueError(f'Shard {i} incomplete: {len(found)}/{len(ids)}')
        scores.update(found)
    jobs = [{**j, **scores[j['id']]} for j in plan['jobs']]
    cs = [j for j in jobs if j['kind']=='candidate']
    calibration = [j['loss'] for j in cs if j['split']=='calibration']
    p10, p90 = quantile(calibration,.1), quantile(calibration,.9)
    m = (p10+p90)/2; s = (p90-p10)/(2*math.log(9))
    if s <= 1e-8:
        raise ValueError('Calibration losses are effectively constant; do not amplify into a reward')
    for c in cs:
        c['loss_reward'] = sigmoid_reward(c['loss'],m,s)
        c['baseline'] = .9*c['reconstruction']+.1*c['format']
        for w in (.1,.2,.3):
            c[f'weight_{w}'] = (.9-w)*c['reconstruction']+w*c['loss_reward']+.1*c['format']
    groups = group_by([c for c in cs if c['split']=='analysis'])
    comparable = [v for v in groups.values() if len(v)>=2]
    span = lambda c, key: max(x[key] for x in c)-min(x[key] for x in c)
    weight_reports = {}
    changed = []
    for w in (.1,.2,.3):
        key = f'weight_{w}'
        rec_gaps = [(.9-w)*span(c,'reconstruction') for c in comparable]
        loss_gaps = [w*span(c,'loss_reward') for c in comparable]
        flips = sum(rank_winners(c,'baseline').isdisjoint(rank_winners(c,key)) for c in comparable)
        for group in comparable:
            if rank_winners(group,'baseline').isdisjoint(rank_winners(group,key)):
                changed.append({'group_id': group[0]['group_id'], 'loss_weight': w,
                                'baseline_winners': sorted(rank_winners(group,'baseline')),
                                'new_winners': sorted(rank_winners(group,key))})
        weight_reports[str(w)] = {'groups': len(comparable), 'winner_changes': flips,
                'winner_change_fraction': flips/len(comparable) if comparable else None,
                'weighted_reconstruction_span': distribution(rec_gaps),
                'weighted_loss_span': distribution(loss_gaps),
                'ratio_of_mean_spans': st.fmean(loss_gaps)/st.fmean(rec_gaps) if rec_gaps and st.fmean(rec_gaps)>0 else None}
    by_group = group_by(jobs)
    controls = {}
    for kind in ('wrong_gender','wrong_emotion'):
        differences = []
        for group in by_group.values():
            by_kind = {j['kind']:j for j in group}
            if kind in by_kind:
                differences.append(by_kind[kind]['loss']-by_kind['reference']['loss'])
        controls[kind] = {'wrong_minus_reference_loss': distribution(differences),
                          'reference_lower_loss': sum(d>1e-5 for d in differences),
                          'tie_within_1e-5': sum(abs(d)<=1e-5 for d in differences),
                          'wrong_lower_loss': sum(d < -1e-5 for d in differences)}
    useful = []
    for group in by_group.values():
        base = next(j['loss'] for j in group if j['kind']=='text_only')
        useful.extend(base-j['loss'] for j in group if j['kind']=='candidate' and j['split']=='analysis')
    report = {'complete': True, 'scored_jobs': len(jobs), 'candidate_count': len(cs),
              'plan_hash': plan_hash, 'checkpoint': plan['checkpoint'],
              'calibration': {'groups': len(plan['calibration_groups']), 'candidate_losses': distribution(calibration),
                              'm': m, 's': s, 'method':'P10->0.9, P90->0.1, fixed across disjoint analysis groups'},
              'analysis_candidate_loss': distribution([c['loss'] for c in cs if c['split']=='analysis']),
              'analysis_group_loss_span': distribution([span(c,'loss') for c in comparable]),
              'analysis_group_loss_reward_span': distribution([span(c,'loss_reward') for c in comparable]),
              'text_only_minus_candidate_loss': distribution(useful), 'controls': controls,
              'weights': weight_reports, 'notes': plan['notes'] + [
                  'Changed winners are not automatically improvements; inspect the saved candidate instructions.',
                  'All historical valid candidates are analyzed; a new transcript gate is not imposed in this probe.',
                  'Existing attribute rewards were not recomputed with the corrected transcript; no extra synthesis/API calls.',
                  'Group span ratios are descriptive, not exact per-pair contributions or GRPO gradient ratios.']}
    write_json(output / 'summary.json', report)
    write_json(output / 'changed_winners.json', changed)
    write_json(output / 'candidates.json', cs)
    with (output / 'candidates.csv').open('w') as f:
        fields = ['id','group_id','split','loss','main_nll','sub_nll','loss_reward','reconstruction','format',
                  'baseline','weight_0.1','weight_0.2','weight_0.3','transcript_exact_normalized',
                  'original_transcript','fixed_transcript','language','instruct']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for c in cs:
            request = c.get('request', {})
            writer.writerow({**c, 'fixed_transcript': request.get('text'),
                             'language': request.get('language'), 'instruct': request.get('instruct')})
    lines = ['# V5 loss 奖励补算', '', f'完成 {len(jobs)} 次评分，其中候选 {len(cs)} 条。', '',
             f'固定评分 checkpoint：`{plan["checkpoint"]}`', '',
             f'校准参数：m={m:.6f}，s={s:.6f}。只在预先划分的校准组确定；下表使用其余组。', '',
             '| loss 权重 | 可比较组数 | 第一名改变组数 | 重建项平均加权跨度 | loss 项平均加权跨度 | 跨度均值比 |',
             '|---|---:|---:|---:|---:|---:|']
    for w,v in weight_reports.items():
        rec_mean = v['weighted_reconstruction_span'].get('mean')
        loss_mean = v['weighted_loss_span'].get('mean')
        ratio = v['ratio_of_mean_spans']
        display = lambda x: f'{x:.6f}' if x is not None else 'N/A'
        lines.append(f'| {w} | {v["groups"]} | {v["winner_changes"]} | {display(rec_mean)} | {display(loss_mean)} | {display(ratio)} |')
    lines += ['', '重建/格式基线为 0.9/0.1。加入 loss 后分别为 0.8/0.1/0.1、0.7/0.2/0.1、0.6/0.3/0.1。', '',
              '## 属性替换对照', '', '对照以现有模型打标为参考，不能视为人工正确率。只替换一个属性，其余条件保持一致。', '',
              '| 替换属性 | 参考指令 loss 更低 | 近似相同 | 替换后 loss 更低 |', '|---|---:|---:|---:|']
    for k,v in controls.items():
        lines.append(f'| {k} | {v["reference_lower_loss"]} | {v["tie_within_1e-5"]} | {v["wrong_lower_loss"]} |')
    lines += ['', '## 如何解释', '',
              'loss 加权跨度很小时，它对排序影响通常有限；改变第一名不等于选得更好。若参考指令不能稳定优于明显替换的指令，不宜仅通过提高权重放大信号。', '',
              '这次只模拟 loss 辅助项：历史重建分仍来自原 caption 生成的音频，并未按固定转录重新生成。因此不能把这个报告当作完整新奖励方案的训练效果。', '',
              '本脚本不训练、不写回 checkpoint、不调用 API。转录来源是缓存的原音频打标；并未人工校验。', '',
              '候选详细输入与分数：candidates.json / candidates.csv；第一名改变的组：changed_winners.json。']
    (output / 'report.md').write_text('\n'.join(lines)+'\n')
    print(f'[REPORT] {output / "report.md"}', flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','run','smoke','analyze','worker'))
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--gpu-memory-gib',type=float,default=26)
    parser.add_argument('--index',type=int)
    parser.add_argument('--worker-smoke',action='store_true')
    args=parser.parse_args();os.umask(0);output=args.output_dir.resolve()
    if args.mode=='worker':
        plan=read(output/'plan.json');worker(output,plan,sha256_json(plan),args.index,args.gpu_memory_gib,args.worker_smoke);return
    if args.mode=='analyze':
        analyze(output,read(output/'plan.json'));return
    gpus=parse_gpus(args.gpus)
    if len(gpus)!=8 or args.gpu_memory_gib<=0:
        raise ValueError('Eight GPUs and a positive memory budget are required')
    plan=make_plan()
    counts=dict(Counter(j['kind'] for j in plan['jobs']))
    print(json.dumps({'cpu_preflight':'passed','gpu_execution':'not_performed', 'jobs':len(plan['jobs']),
          'job_kinds':counts,'shard_counts':list(map(len,plan['shards'])),'checkpoint':plan['checkpoint'],
          'api_requests':0,'training_updates':0},indent=2),flush=True)
    if args.mode=='check':
        return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'probe.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=output/'plan.json'
        if path.exists() and read(path)!=plan:
            raise ValueError('Probe identity changed; use a new output directory')
        write_json(path,plan)
        check_gpu_memory(gpus,args.gpu_memory_gib)
        launch(output,plan,gpus,args.gpu_memory_gib,True)
        if args.mode=='run':
            launch(output,plan,gpus,args.gpu_memory_gib,False)
            analyze(output,plan)
        else:
            print('[SMOKE DONE] Eight long examples scored successfully; run mode now performs the full probe.')


if __name__=='__main__':
    def stop(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        main()
    except KeyboardInterrupt:
        print('[STOPPED] Saved scores preserved. Rerun the same command to resume.',flush=True)
        raise SystemExit(130)
