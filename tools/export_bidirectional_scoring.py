#!/usr/bin/env python3
"""Export the independent scoring pilot without loading models or changing source runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path('experiments/bidirectional_scoring')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/data/L202500147/Caption/BidirectionalScoringPilot'))
    args = parser.parse_args()
    source = args.source.resolve()
    provenance = ROOT / 'provenance/bidirectional_scoring.json'
    prior = json.loads(provenance.read_text()) if provenance.exists() else {'files': []}
    previous = {e['path']: e['export_sha256'] for e in prior['files']}
    entries = []

    def copy(relative, target, transform=None):
        src = source / relative
        if not src.is_file() or src.is_symlink():
            raise ValueError('Missing or symlink source: ' + str(src))
        raw = src.read_bytes()
        if len(raw) > 2 * 1024 * 1024 or b'\0' in raw:
            raise ValueError('Unexpected large/binary file: ' + str(src))
        text = raw.decode('utf-8-sig')
        if transform:
            text = transform(text)
        data = text.encode('utf-8')
        path = PREFIX / target
        dst = ROOT / path
        if dst.exists() and dst.read_bytes() != data and sha(dst.read_bytes()) != previous.get(str(path)):
            raise ValueError('Refusing to overwrite edited export: ' + str(path))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        entries.append({'path': str(path), 'bytes': len(data), 'export_sha256': sha(data),
                        'source': str(src), 'source_sha256': sha(raw), 'redactions': 0,
                        'transformation': 'archive_navigation_note' if transform else
                            ('utf8_bom_removed' if raw.startswith(b'\xef\xbb\xbf') else 'none')})

    for src in sorted(source.iterdir()):
        if src.suffix in {'.py', '.sh'}:
            copy(src.name, Path('src') / src.name)
    for folder in ['web', 'tests']:
        for src in sorted((source / folder).iterdir()):
            if src.is_file() and src.suffix in {'.py', '.js', '.html', '.css'}:
                copy(src.relative_to(source), Path('src') / src.relative_to(source))
    copy('PROTOCOL.md', 'PROTOCOL.md')
    for name in ['processor_contract.json', 'validation.json', 'pilot30_validation.json',
                 'numeric_retry_caption_cpu.json', 'numeric_retry_tts_cpu.json', 'v3_precision_cpu.json']:
        copy(Path('checks') / name, Path('checks') / name)
    copy('checks/NUMERIC_RETRY.md', 'checks/NUMERIC_RETRY.md')

    for run, reports in [('smoke01', ['v2', 'v2_review', 'v3_combined', 'v3_review']),
                         ('pilot30', ['v1', 'v1_review'])]:
        for name in ['manifest.json', 'annotations.json']:
            copy(Path('runs') / run / name, Path('results') / run / name)
        for src in sorted((source / 'runs' / run / 'scores').rglob('*.json')):
            copy(src.relative_to(source), Path('results') / run / src.relative_to(source / 'runs' / run))
        for report in reports:
            folder = source / 'runs' / run / 'reports' / report
            for src in sorted(folder.iterdir()):
                if not src.is_file() or src.suffix not in {'.json', '.csv', '.md'}:
                    continue
                def navigation(text):
                    text = text.replace('[verify_saved_results.py](verify_saved_results.py)',
                                        '`verify_saved_results.py`（原服务器核验脚本，含音频校验；未纳入此导出）')
                    text = re.sub(r'\[([^\]]+)\]\([^)]*\.zip\)',
                                  r'`\1`（试听 ZIP 仅保存在原服务器）', text)
                    return ('> GitHub 归档：此文保留运行时结论。音频及试听 ZIP 未上传；'
                            '服务器命令和原路径仅用于追溯。仓库内无模型复算入口见 '
                            '[实验首页](../../../../README.md)。\n\n' + text)
                copy(src.relative_to(source), Path('results') / run / 'reports' / report / src.name,
                     navigation if src.suffix == '.md' else None)
    output = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_root': str(source),
              'scope': 'Independent emotion-scoring pilot; source, human references, all saved scores and audits; no audio/weights/cache/logs',
              'files': entries,
              'omitted': ['audio and ZIP/HTML listening bundles', 'codec and model caches',
                          'full logs and annotation edit history', 'server-only audit helper requiring audio']}
    for e in entries:
        assert sha(Path(e['source']).read_bytes()) == e['source_sha256'], e['source']
    provenance.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'files': len(entries), 'bytes': sum(e['bytes'] for e in entries),
                      'source_unchanged': True, 'models_loaded': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
