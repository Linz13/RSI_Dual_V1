#!/usr/bin/env python3
"""Read-only CPU checks for the export. Does not import training/API modules."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SECRET_RE=re.compile(r'(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{30,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
SECRET_NAME=re.compile(r'^(?:api_?key|access_?token|auth_?token|secret_?key|password)$',re.I)
DENIED={'.wav','.mp3','.flac','.ogg','.mp4','.safetensors','.pt','.pth','.ckpt','.bin','.onnx','.zip','.gz'}
def looks_like_secret(value):
    if not isinstance(value,str) or len(value)<12:return False
    if any(x in value.lower() for x in ('example','dummy','fake','placeholder','redacted','your_','your-','test-','test_')):return False
    if re.fullmatch(r'[A-Z_]+',value):return False
    return bool(re.search('[a-zA-Z]',value) and re.search('[0-9]',value))

def main():
    if sys.version_info < (3,10):
        raise SystemExit('Use Python 3.10+ to validate this snapshot (some archived tests use newer syntax).')
    errors=[];counts={'files':0,'bytes':0,'python':0,'json':0,'jsonl':0,'shell':0,'provenance_verified':0,'generated_doc_links':0}
    files=[]
    for p in ROOT.rglob('*'):
        if any(part in {'.git','__pycache__','.pytest_cache'} for part in p.relative_to(ROOT).parts):continue
        if p.is_symlink():errors.append([str(p.relative_to(ROOT)),'symlink']);continue
        if not p.is_file():continue
        rel=str(p.relative_to(ROOT));data=p.read_bytes();counts['files']+=1;counts['bytes']+=len(data);files.append((rel,data))
        if p.suffix in DENIED or len(data)>2*1024*1024:errors.append([rel,'binary_or_large_file'])
        try:text=data.decode('utf-8-sig')
        except UnicodeDecodeError:errors.append([rel,'non_utf8']);continue
        if SECRET_RE.search(text):errors.append([rel,'credential_pattern'])
        urls=re.findall(r'''https?://[^\s/@:'"]+:[^\s/@'"]+@''',text)
        # Exact synthetic credential used to test that exception logs redact URLs.
        if rel=='frameworks/v5/tests_v5/test_judge_resume_check.py':
            urls=[url for url in urls if url!='https://'+'user:fixture-secret-123@']
        if urls:errors.append([rel,'URL_credentials'])
        try:
            if p.suffix=='.py':
                tree=ast.parse(text,filename=rel);counts['python']+=1
                for n in ast.walk(tree):
                    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and SECRET_NAME.match(t.id) for t in n.targets):
                        if isinstance(n.value,ast.Constant) and looks_like_secret(n.value.value):errors.append([rel,f'credential_literal_line_{n.lineno}'])
            elif p.suffix=='.json':
                obj=json.loads(text);counts['json']+=1
                def walk(o):
                    if isinstance(o,dict):
                        for k,v in o.items():
                            if SECRET_NAME.match(k) and looks_like_secret(v):errors.append([rel,'credential_json_value'])
                            walk(v)
                    elif isinstance(o,list):
                        for v in o:walk(v)
                walk(obj)
            elif p.suffix=='.jsonl':
                for line in text.splitlines():
                    if line.strip():json.loads(line)
                counts['jsonl']+=1
            elif p.suffix=='.sh':
                result=subprocess.run(['bash','-n',str(p)],capture_output=True)
                if result.returncode:errors.append([rel,'shell_syntax'])
                counts['shell']+=1
        except (SyntaxError,ValueError) as exc:errors.append([rel,type(exc).__name__])
    for manifest_path in [ROOT/'provenance/files.json', ROOT/'provenance/bidirectional_scoring.json']:
        if not manifest_path.exists():
            errors.append([str(manifest_path.relative_to(ROOT)),'missing_provenance'])
            continue
        manifest=json.loads(manifest_path.read_text())
        for e in manifest['files']:
            p=ROOT/e['path']
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=e['export_sha256']:
                errors.append([e['path'],'export_hash_mismatch'])
            else:counts['provenance_verified']+=1
    # Check authored navigation, not immutable old reports with server-local paths.
    doc_paths=['README.md','docs/TRAINING.md','docs/CODE_MAP.md','docs/DEPENDENCIES.md',
               'experiments/INDEX.md','experiments/v5/README.md','experiments/v6/README.md',
               'frameworks/v5/README.md','frameworks/v6/README.md','labeling/README.md',
               'results/CAPTIONER.md','results/TTS_DSD.md','experiments/bidirectional_scoring/README.md',
               'provenance/README.md']
    for rel in doc_paths:
        p=ROOT/rel
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in link or link.startswith('#'):continue
            target=(p.parent/link.split('#')[0]).resolve()
            if not target.exists():errors.append([rel,'missing_link:'+link])
            counts['generated_doc_links']+=1
    result={'status':'passed' if not errors else 'failed',**counts,'errors':errors,
            'scope':'UTF-8, Python AST, JSON/JSONL, bash -n, secret patterns, size, provenance hashes and authored navigation; no GPU/API/training'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return bool(errors)

if __name__=='__main__':sys.exit(main())
