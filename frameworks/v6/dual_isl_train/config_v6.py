"""Reject old protocol options rather than silently ignoring them."""
import math


def validate(c):
    if c.get('version') != 6:
        raise ValueError('This entrypoint requires version=6')
    t = c['training']
    if t.get('loops') != ['audio_only'] or t.get('warmstart') is not False or t.get('paired_anchor_enabled') is not False:
        raise ValueError('V6 only supports audio_only, no warmstart or paired anchors')
    if t['rounds'] < 1 or t['group_size'] < 2:
        raise ValueError('rounds >=1 and group_size >=2 required')
    r = c['reward']
    if r != {'reconstruction_weight': 0.9, 'format_weight': 0.1,
             'max_content_error': 0.1, 'sft_attribute_threshold': None}:
        raise ValueError('V6 requires agreed 0.9/0.1 reward, 10% content gate and no SFT attribute threshold')
    for role, phase in (('captioner', 'grpo'), ('tts', 'cycle_sft')):
        s = c[role]
        if not s.get('python') or not s.get('worker_module') or s.get('adapter_path'):
            raise ValueError('V6 needs worker/python and starts from base (adapter_path empty)')
        phases = s['training']['phases']
        if set(phases) != {phase} or phases[phase]['epochs'] != 1:
            raise ValueError(f'{role} must only run one epoch of {phase}')
    g = c['captioner']['generation']
    if (g['temperature'], g['top_p'], g['top_k']) != (1, 1, 0):
        raise ValueError('Batched Captioner replay requires temperature/top_p/top_k=1/1/0')
    if not 1 <= g['rollout_batch_size'] <= t['group_size'] or c['captioner']['training']['replay_batch_size'] != 1:
        raise ValueError('Captioner batch must fit group; replay microbatch remains 1')
    if c['tts']['generation']['synthesis_batch_size'] < 1 or c['tts']['num_codebooks'] != 16:
        raise ValueError('Invalid TTS batch/codebooks')
    d = c['distributed']
    if d['world_size'] < 1 or d['enabled'] != (d['world_size'] > 1):
        raise ValueError('Inconsistent distributed configuration')
    memory = c.get('gpu_memory_gib')
    if memory is not None and (not math.isfinite(memory) or memory <= 0):
        raise ValueError('GPU memory budget must be positive or null')
    if c['data']['max_records'] < 0 or c['labeling']['api_workers'] < 1 or c['labeling']['attempts'] < 1:
        raise ValueError('Invalid counts')


def set_gpu_budget(torch, device, gib):
    if gib is not None:
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 1024**3 / total), device)
