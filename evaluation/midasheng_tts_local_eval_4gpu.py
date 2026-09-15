#!/usr/bin/env python3
"""Four-GPU local evaluation, with audited copying from the stopped eight-GPU run.

Existing eight-GPU scripts and their run identities remain unchanged. This entry
uses a new output root. Each checkpoint gets two GPUs: one for generation, then
two for local WER/WVMOS. It never calls a judge API.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

import midasheng_tts_local_eval as base

DEFAULT_OUTPUT = base.BENCH / "midasheng_original_tts_local_r5_r8_4gpu_20260909_run01"
ROOT_IDENTITY = {"launcher": "midasheng_original_local_r5_r8_4gpu_v1", "source_run": str(base.SOURCE_RUN)}


def source_records(source, identity):
    """Audit original protocol/adapter/code before accepting any old audio."""
    prior = json.loads((source / "local_run_identity.json").read_text())
    for key, value in identity.items():
        if prior.get(key) != value:
            raise ValueError(f"Source checkpoint mismatch ({key}): {source}")
    expected = base.expected_generation(identity, "full")
    if prior.get("generation") != expected:
        raise ValueError(f"Source generation settings mismatch: {source}")
    for name, digest in prior["source_sha256"].items():
        if base.sha256_file(Path(name)) != digest:
            raise ValueError(f"Source code changed since original evaluation: {name}")
    manifest = json.loads((source / "generation_manifest.json").read_text())
    if manifest.get("generation_identity") != expected or manifest.get("generation_identity_sha256") != base.sha256_json(expected):
        raise ValueError(f"Source manifest identity mismatch: {source}")
    if manifest.get("expected_samples") != 1645:
        raise ValueError("Source is not the full 1645-sample evaluation")
    records, seen = [], set()
    for record in manifest["samples"]:
        i = record["unique_id_eval"]
        if type(i) is not int or i < 0 or i >= 1645 or i in seen:
            raise ValueError("Source manifest has duplicate/invalid IDs")
        seen.add(i)
        if record["status"] not in {"generated", "existing"}:
            continue
        path = source / "audios" / f"{i}.wav"
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getnframes() <= 0:
                raise ValueError(f"Invalid source WAV: {path}")
        records.append((record, path, base.sha256_file(path)))
    return manifest, records


def import_audio(source, dest, identity, gpus, hashes):
    # After the first import, new progress belongs solely to this new directory.
    expected = {**identity, "mode": "full", "generation": base.expected_generation(identity, "full"),
                "source_sha256": hashes, "wvmos_sha256": base.sha256_file(base.MOS),
                "gpu_group": gpus, "qwen_python": str(base.QWEN_PYTHON), "eval_python": str(base.EVAL_PYTHON)}
    marker = dest / "local_run_identity.json"
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError(f"Four-GPU run identity changed: {dest}")
        if (dest / "generation_manifest.json").exists():
            return
    elif dest.exists() and any(dest.iterdir()):
        raise ValueError(f"Refusing nonempty unowned output: {dest}")
    manifest, records = source_records(source, identity)
    base.write_json(marker, expected)
    (dest / "audios").mkdir(exist_ok=True)
    copied, audit = [], []
    for record, path, digest in records:
        target = dest / "audios" / path.name
        if target.exists():
            if base.sha256_file(target) != digest:
                raise ValueError(f"Different existing audio: {target}")
        else:
            temporary = target.with_suffix(".wav.copying")
            shutil.copyfile(path, temporary)
            if base.sha256_file(temporary) != digest:
                raise ValueError(f"Audio changed during copying: {path}")
            temporary.replace(target)
        copied.append({**record, "output_path": str(target), "status": "existing"})
        audit.append({"id": record["unique_id_eval"], "source": str(path), "destination": str(target), "sha256": digest})
    manifest["samples"] = copied
    base.write_json(dest / "audio_import.json", {"source": str(source), "copied_samples": len(copied), "files": audit})
    base.write_json(dest / "generation_manifest.json", manifest)
    print(f"[REUSE] round {identity['completed_rounds']}: copied {len(copied)} verified WAVs; original run unchanged", flush=True)


class FourGPURunner(base.Runner):
    def execute(self, cmd, env, log):
        cmd = list(cmd)
        if cmd[1] == str(base.TOOLS / "run_staged_local_metrics.sh"):
            cmd[cmd.index("--workers") + 1] = "2"
            cmd[cmd.index("--devices") + 1] = "cuda:0,cuda:1"
        return super().execute(cmd, env, log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "smoke", "full", "summarize"))
    parser.add_argument("--gpus", default=os.environ.get("EVAL_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES", "4,5,6,7")))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", type=Path, default=base.DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    root, source = args.output_root.resolve(), args.source_root.resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("New evaluation root must be separate from the old run")
    if not args.output_root.is_absolute() or root == base.BENCH or base.BENCH not in root.parents:
        raise ValueError("Use an absolute dedicated directory under Caption/benchmark")
    gpus = [g.strip() for g in args.gpus.split(",")]
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(g.isdigit() or g.startswith("GPU-") for g in gpus):
        raise ValueError("Exactly four distinct GPU indices/UUIDs required")
    if args.action != "summarize":
        identities = base.check_resources()
        base.check_imports()
        if args.action == "check":
            for rounds in (6, 9):
                _, records = source_records(source / "full" / f"rounds_{rounds:02d}", identities[rounds])
                print(f"[CHECK] round {rounds}: {len(records)} reusable WAVs")
            print("[CHECK] Four-GPU CPU preflight passed; no GPU jobs or output directories created")
            return 0
        subprocess.run([str(base.QWEN_PYTHON), "-c", "import torch; assert torch.cuda.device_count()==4, 'Four visible GPUs required'"],
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}, check=True)
    marker = root / "local_eval_root.json"
    if args.action == "summarize" and not marker.exists():
        raise ValueError("No four-GPU evaluation root to summarize")
    if root.exists() and not marker.exists() and any(root.iterdir()):
        raise ValueError("Refusing nonempty directory without four-GPU root identity")
    os.umask(0o002)
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((root / ".launcher.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists() and json.loads(marker.read_text()) != ROOT_IDENTITY:
            raise ValueError("Root identity mismatch")
        base.write_json(marker, ROOT_IDENTITY)
        if args.action == "summarize":
            return base.write_summary(root, args.mode)
        hashes = base.source_hashes()
        for p in (Path(__file__), base.BENCH / "run_midasheng_tts_local_4gpu.sh"):
            hashes[str(p)] = base.sha256_file(p)
        if args.action == "full" and any(not (root / "full" / f"rounds_{r:02d}" / "generation_manifest.json").exists() for r in (6, 9)):
            # Advisory lock checks the old launcher; the user must also stop orphan workers.
            with (source / ".launcher.lock").open("r") as old_lock:
                fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for i, rounds in enumerate((6, 9)):
                    import_audio(source / "full" / f"rounds_{rounds:02d}", root / "full" / f"rounds_{rounds:02d}",
                                 identities[rounds], gpus[2*i:2*i+2], hashes)
        runner = FourGPURunner()
        signal.signal(signal.SIGINT, runner.stop)
        signal.signal(signal.SIGTERM, runner.stop)
        failed = False
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(runner.case, root, args.action, identities[r], gpus[2*i:2*i+2], hashes)
                    for i, r in enumerate((6, 9))]
            for job in as_completed(jobs):
                try:
                    job.result()
                except Exception as exc:
                    failed = True
                    print(f"[FAILED] {exc}", file=sys.stderr, flush=True)
        status = base.write_summary(root, args.action)
        return 1 if failed or runner.stopping or status else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, wave.Error, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
