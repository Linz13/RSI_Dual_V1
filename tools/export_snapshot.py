#!/usr/bin/env python3
"""Export a small research snapshot; never runs training, APIs, or git push.

Source paths refer to the original shared server. Refuses to overwrite changed
exported files. Text files only; no symlink traversal, audio, weights or datasets.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone

DEST = Path(__file__).resolve().parents[1]
SUFFIXES = {'.py', '.sh', '.json', '.yaml', '.yml', '.toml', '.md', '.txt', '.csv', '.tex'}
SKIP = {'__pycache__', 'runs', 'models', 'data', 'audio', 'cache', 'logs', 'outputs',
        'runtime', 'source', 'sources', 'third_party', 'checkpoints', 'reports'}
MAX_BYTES = 2 * 1024 * 1024
KEY_RE = re.compile(r'(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{30,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,})')
SECRET_NAME = re.compile(r'^(?:api_?key|access_?token|auth_?token|secret_?key|password)$', re.I)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'

class Exporter:
    def __init__(self, root):
        self.root = root
        self.entries = []
        self.omitted = []
        self.secrets = set()
        old = DEST/'provenance/files.json'
        self.previous = {e['path']: e['export_sha256'] for e in json.loads(old.read_text())['files']} if old.exists() else {}
        # Read only credential literals, never import or execute API demos.
        for rel in ['api/gemini_audio.py', 'api/gpt_text.py', 'api/api_demo/api_demo/qwen_omni_wav_to_text.py']:
            p = root.parent/rel
            if not p.is_file(): continue
            tree = ast.parse(p.read_text())
            for n in ast.walk(tree):
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and SECRET_NAME.match(t.id) for t in n.targets):
                    if isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) and len(n.value.value) >= 12:
                        self.secrets.add(n.value.value)

    def clean(self, text):
        count = 0
        for key in sorted(self.secrets, key=len, reverse=True):
            count += text.count(key)
            text = text.replace(key, 'REDACTED_SET_VIA_ENV')
        text, n = KEY_RE.subn('REDACTED_SET_VIA_ENV', text)
        return text, count+n

    def write(self, target, text, source=None, transform=None):
        text, redactions = self.clean(text)
        p = DEST/target
        data = text.encode()
        if p.exists() and p.read_bytes() != data:
            if self.previous.get(target) != digest(p.read_bytes()):
                raise RuntimeError('Refusing to overwrite locally edited file: '+target)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        p.chmod(0o644)
        entry = {'path': target, 'bytes': len(data), 'export_sha256': digest(data), 'redactions': redactions}
        if source:
            raw = source.read_bytes()
            entry.update(source=str(source), source_sha256=digest(raw), transformation=transform or ('credential_redaction' if redactions else 'none'))
        else:
            entry.update(source=None, transformation=transform or 'generated_summary')
        self.entries.append(entry)

    def copy(self, source, target):
        if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_BYTES:
            self.omitted.append({'source':str(source),'reason':'missing_symlink_or_size_limit'})
            return
        raw = source.read_bytes()
        if b'\0' in raw: raise ValueError('Unexpected binary: '+str(source))
        text = raw.decode('utf-8-sig')
        if source.suffix == '.json':
            # Collect non-prefixed keys from JSON config dictionaries too.
            def walk(v):
                if isinstance(v, dict):
                    for k, x in v.items():
                        if SECRET_NAME.match(k) and isinstance(x,str) and len(x)>=12:
                            self.secrets.add(x)
                        walk(x)
                elif isinstance(v,list):
                    for x in v: walk(x)
            walk(json.loads(text))
        self.write(target, text, source, 'utf8_bom_removed' if raw.startswith(b'\xef\xbb\xbf') else None)

    def tree(self, source, target, skip=SKIP):
        for d, ds, fs in os.walk(source):
            ds[:] = sorted(x for x in ds if x not in skip and not x.startswith('.') and not Path(d,x).is_symlink())
            for f in sorted(fs):
                p = Path(d,f)
                if p.suffix in SUFFIXES or p.name in {'LICENSE','NOTICE'}:
                    self.copy(p, str(Path(target)/p.relative_to(source)))

    def selected(self, source, target, names):
        for name in names:
            p = source/name
            if p.is_file(): self.copy(p, str(Path(target)/name))

    def run_record(self, source, target):
        self.selected(source,target,['resolved_config.yaml','data_contract.json','status.json','latest.json'])
        rows=[]
        for folder in sorted(source.glob('round_[0-9][0-9][0-9]')):
            if not (folder/'commit.json').exists(): continue
            name=folder.name
            self.selected(folder, target+'/'+name, ['summary.json','commit.json'])
            s=json.loads((folder/'summary.json').read_text())
            row={'round_index':int(name[-3:]),'training_round':int(name[-3:])+1}
            row.update({k:v for k,v in s.items() if isinstance(v,(float,int,str)) or v is None})
            for role in ['caption_final','tts_final']:
                cp=folder/'checkpoints'/role
                self.selected(cp,target+'/'+name+'/adapter_metadata/'+role,['adapter_config.json','checkpoint.json','metadata.json'])
            rows.append(row)
        if rows:
            fields=list(dict.fromkeys(k for r in rows for k in r))
            buf=io.StringIO(); w=csv.DictWriter(buf,fields);w.writeheader();w.writerows(rows)
            self.write(target+'/rounds.csv',buf.getvalue())
        return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,default=Path('/data/L202500147/Caption'));a=ap.parse_args()
    ex=Exporter(a.source_root); root=a.source_root
    v5=root/'DualISL_Train_RewardV5_FastResume';v6=root/'DualRSI_Train_V6_AudioOnly4Attr'
    for source,target,folders in [(v5,'frameworks/v5',['dual_isl_train','scripts','configs','tests','tests_v5']),
                                  (v6,'frameworks/v6',['dual_isl_train','scripts','configs','tests_v6'])]:
        for f in folders: ex.tree(source/f,target+'/'+f)
        ex.selected(source,target,['pyproject.toml'])
    ex.selected(v6,'frameworks/v6',['run_v6_api_workers.sh','runtime_api_workers_v6.py'])
    # Preserve historic documentation as dated evidence, not as the main run guide.
    ex.copy(v5/'README.md','docs/history/v5_fastresume_original.md')
    ex.copy(v6/'README.md','docs/history/v6_original.md')
    ex.copy(root/'FOUR_ATTRIBUTE_AUDIO_ONLY_TRAINING_PLAN.md','docs/history/four_attribute_plan.md')
    ex.copy(v6/'data/data_report.json','experiments/v6/data_report.json')
    bench=root/'benchmark'
    ex.selected(bench,'evaluation', [p.name for p in bench.iterdir() if p.is_file() and p.suffix in {'.py','.sh','.md'}])
    ex.tree(bench/'tests','evaluation/tests')
    for sub in ['emotiontalk_speech_captioning','paraspeechcaps','stylecap_promptspeech_mcq','InstructTTSEval-public']:
        ex.tree(bench/sub,'evaluation/'+sub)
    # Only the project's labeling adapter code, not downloaded upstream models.
    label=root/'labeling2_transfer_20260909'
    for sub in ['Experiment/labeling2','Experiment/acc_model_pool/Caption_Bench','Experiment/acc_model_pool/AIR-Bench']:
        ex.tree(label/'source'/sub,'labeling/source/'+sub)
    ex.selected(label,'labeling',['SCHEME.md'])
    ex.selected(label/'configs','labeling/configs',['config.target.example.json'])
    ex.tree(label/'reference','labeling/reference')
    ex.tree(label/'metadata/environments','labeling/environment/original_environment_records')
    ex.selected(label/'metadata','provenance/labeling',['source_repositories.json','source_inventory.json'])
    ex.selected(label/'deployment','labeling/environment', [p.name for p in (label/'deployment').glob('constraints-*.txt')]+['requirements-stepaudio-hf.txt'])
    # Reports: bounded immediate text artifacts; leave full audio, HTML/base64 and API caches on server.
    for report in sorted((bench/'reports').iterdir()):
        if not report.is_dir():continue
        for p in sorted(report.iterdir()):
            if p.is_file() and p.suffix in {'.md','.json','.csv','.py','.tex'} and p.name not in {'plan.json'}:
                ex.copy(p,'experiments/diagnostics/'+report.name+'/'+p.name)
    eval_names=['v6_caption_all_rounds_run01','v6_round000_002_benchmarks_run01','v6_round003_benchmarks_run01',
                'midasheng_v4_all_v5_round1_caption_4gpu_20260910_run01','midasheng_v5_round001_caption_3gpu_run01']
    for name in eval_names:
        ex.selected(bench/name,'results/benchmarks/'+name,['summary.json','summary.csv','summary.md','inventory.json','full_execution.json'])
    dsd=bench/'InstructTTSEval-public/qwen3_voice_design/runs'
    ex.copy(dsd/'base/full_bilingual_seed42/evaluation_gemini_2_5_pro_audio_api/summary.json','results/tts_dsd/base_current_api.json')
    for i in (0,1):
        ex.copy(dsd/f'v5_midasheng_round{i:03d}_dsd_seed42/evaluation_gemini_2_5_pro_audio_api/dsd_summary.json',f'results/tts_dsd/v5_round{i:03d}.json')
    ex.run_record(v6/'runs/midasheng_v6_8gpu_run01','experiments/v6/training')
    ex.run_record(root/'DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01','experiments/v5/training')
    ex.selected(v5/'reports','provenance/v5_migration',['EIGHT_GPU_MIGRATION.json','POST_MIGRATION_CHECK.json','ROUND0_VERIFICATION_BEFORE_MIGRATION.json'])
    # Retain a bounded text example per round: first 2 groups in file order, no waveform/codec data.
    for i in range(10):
        folder=v6/f'runs/midasheng_v6_8gpu_run01/round_{i:03d}/collection'
        paths=sorted(folder.glob('*caption_rollout.output.jsonl'))
        if paths:
            with paths[0].open() as f:
                lines=[next(f,'') for _ in range(2)]
            if any(lines):ex.write(f'experiments/v6/examples/round_{i:03d}.jsonl',''.join(lines),paths[0],'first_two_groups_in_file_order_not_representative')
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'source_root':str(root),
                'scope':'code, aggregate results and bounded text examples; no weights/audio/API cache',
                'files':ex.entries,'omitted':ex.omitted}
    p=DEST/'provenance/files.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json_text(provenance))
    print(json.dumps({'files':len(ex.entries),'bytes':sum(x['bytes'] for x in ex.entries),'redacted_files':sum(x['redactions']>0 for x in ex.entries),'omitted':len(ex.omitted)}))

if __name__=='__main__':main()
