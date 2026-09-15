#!/usr/bin/env python3
"""Targeted one-H100 check: real V4 rollout off/on on one short four-candidate group.
Use the verified V3 loader for its original checkpoint; invoke the actual V4
rollout method without checkpoint conversion or a training run.
"""
from __future__ import annotations
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from probe_reference_reuse import DEFAULT_RUN, ROOT, digest, fresh_output, save, select_groups, verify_source


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def production_rollout(project):
    relative = 'dual_isl_train/workers/qwen_voice_design.py'
    # The original checkpoint loader stays V3. All methods called by rollout and
    # its module-level generation helpers must otherwise match the V4 worker.
    def definitions(path):
        tree = ast.parse(path.read_text())
        values = {}
        for item in tree.body:
            if isinstance(item, ast.ClassDef) and item.name == 'QwenVoiceDesignWorker':
                for method in item.body:
                    if isinstance(method, ast.FunctionDef) and method.name != 'rollout':
                        values['method/' + method.name] = ast.dump(method, include_attributes=False)
            elif isinstance(item, ast.FunctionDef) and item.name != 'main':
                values[item.name] = ast.dump(item, include_attributes=False)
        return values
    if definitions(project/relative) != definitions(ROOT/relative):
        raise ValueError('Probe loader no longer matches the production worker outside rollout; review harness')
    for file in ('adapters.py', 'render.py', 'trajectory.py'):
        if digest(project/'dual_isl_train'/file) != digest(ROOT/'dual_isl_train'/file):
            raise ValueError(f'Probe dependency changed: {file}')
    # The only distributed change allowed here is the inference executor. The
    # loader's context and all training schedules must still match exactly.
    def distributed_definitions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name != 'run_sharded_inference'}
    if distributed_definitions(project/'dual_isl_train/distributed.py') != distributed_definitions(ROOT/'dual_isl_train/distributed.py'):
        raise ValueError('Distributed context/training implementation changed; review probe')
    load_module('dual_isl_train.reference_replay', ROOT/'dual_isl_train/reference_replay.py')
    module = load_module('dual_isl_train.workers._v4_rollout_integration', ROOT/relative)
    return module.QwenVoiceDesignWorker.rollout


def compare(before, after):
    if len(before) != 1 or len(after) != 1:
        raise ValueError('Expected one group')
    differences = []
    for left, right in zip(before[0]['candidates'], after[0]['candidates']):
        if not left['trajectory_valid'] or not right['trajectory_valid']:
            raise ValueError('Behavior probability / trajectory validation failed')
        # These are the values consumed by training, including the sampled actions.
        for key in ('candidate_id', 'generation_seed', 'codec_codes', 'old_main_logprobs',
                    'old_sub_logprobs', 'ref_main_logprobs', 'ref_sub_logprobs'):
            if left[key] != right[key]:
                differences.append({'candidate': left['candidate_id'], 'field': key})
        # Audio must also remain identical, not just the sampled codec tokens.
        if digest(left['audio_path']) != digest(right['audio_path']):
            differences.append({'candidate': left['candidate_id'], 'field': 'audio_bytes'})
    if differences:
        raise ValueError(f'Off/on rollout changed: {differences}')
    if len(before[0]['candidates']) != 4 or len(after[0]['candidates']) != 4:
        raise ValueError('Expected four candidates')
    return {'exact': True, 'candidates': 4, 'audio_bytes_equal': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    source = args.source_run.resolve()
    identity, checkpoint, project = verify_source(source)
    rollout = production_rollout(project)
    group = select_groups(source/'round_009/collections/round_009_caption_tts_rollout.output.jsonl')[0]['group']
    output = fresh_output(args.output_dir, source)
    os.umask(0o000)
    output.chmod(0o777)
    results = {'status': 'running', 'checkpoint': checkpoint, 'source_run': str(source),
               'group_id': group['id'], 'scope': 'single GPU real generation+rollout; no optimizer, no DDP',
               'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [
                   Path(__file__), ROOT/'dual_isl_train/workers/qwen_voice_design.py', ROOT/'dual_isl_train/reference_replay.py']},
               'stages': {}}
    save(output/'results.json', results)
    try:
        import torch
        from dual_isl_train.workers.qwen_voice_design import QwenVoiceDesignWorker
        if int(os.environ.get('WORLD_SIZE', '1')) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Expose exactly one allocated H100; do not use torchrun')
        if 'H100' not in torch.cuda.get_device_name(0):
            raise ValueError('Expected H100')
        results['gpu'] = torch.cuda.get_device_name(0)
        config = copy.deepcopy(identity['config'])
        config['distributed'].update(enabled=False, world_size=1)
        config['tts']['device'] = 'cuda:0'
        config['tts']['codec_cache_dir'] = str(output/'unused_codec_cache')
        worker = QwenVoiceDesignWorker(config, checkpoint['path'], checkpoint['path'])
        outputs = []
        for enabled in (False, True):
            label = 'reuse_on' if enabled else 'reuse_off'
            worker.cfg['generation']['reference_replay_reuse'] = enabled
            torch.cuda.synchronize()
            started = time.perf_counter()
            rows = rollout(worker, [group], str(output/label/'rollout.jsonl'))
            torch.cuda.synchronize()
            meta = copy.deepcopy(worker.output_metrics['reference_replay'])
            results['stages'][label] = {'wall_seconds': time.perf_counter()-started, 'audit': meta}
            save(output/label/'rollout.json', rows)
            save(output/'results.json', results)
            outputs.append(rows)
            print(label, json.dumps(results['stages'][label]), flush=True)
        results['comparison'] = compare(*outputs)
        audit = results['stages']['reuse_on']['audit']
        if audit['reused'] != 3 or audit['validation_candidates'] != 1 or audit['validation_max_abs_error'] != 0:
            raise ValueError('Production reuse was not exercised after independent validation')
        results['status'] = 'passed'
    except Exception:
        results.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        save(output/'results.json', results)
    print(f'passed: {output / "results.json"}', flush=True)

if __name__ == '__main__':
    main()
