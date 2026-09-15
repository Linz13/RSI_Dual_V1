#!/usr/bin/env python3
"""Original MiDasheng r5/r8: existing generation + local WER/WVMOS, never a judge.

Two independent branches receive four GPUs each. Generation uses one GPU per
branch (the established batch-8 generator); scoring uses all four branch GPUs.
No modifications to training code or the established generation/scoring workers.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH = Path(__file__).resolve().parent
CAPTION = BENCH.parent
TOOLS = BENCH / "EmergentTTS-Eval-public/qwen3_voice_design"
EVAL_ROOT = TOOLS.parent
SOURCE_RUN = CAPTION / "DualISL_Train/runs/dual_recursive_8gpu_h100_midasheng_from_r2_20260831_run01"
MODEL = CAPTION / "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_OUTPUT = BENCH / "midasheng_original_tts_local_r5_r8_20260909_run01"
DATA = EVAL_ROOT / "data/emergent_tts_eval_data.jsonl"
MOS = EVAL_ROOT / "data/wv_mos.ckpt"
QWEN_PYTHON = Path(os.environ.get("QWEN_PYTHON", "/data/L202500147/miniconda3/envs/qwen3-tts/bin/python"))
EVAL_PYTHON = Path(os.environ.get("EVAL_PYTHON", "/data/L202500147/miniconda3/envs/emergent-tts-eval/bin/python"))
sys.path.insert(0, str(TOOLS))
from staged_common import load_eval_samples, load_latest_records, sha256_json
from model_adapter_utils import describe_adapter, sha256_file


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def checkpoint_hash(path):
    # Exact original DualISL io.hash_path convention, without importing training.
    material = [{"path": str(p.relative_to(path)), "sha256": sha256_file(p)}
                for p in sorted(path.rglob("*"))
                if p.is_file() and "training_metrics" not in p.parts]
    return sha256_json(material)


def checkpoint_identity(rounds):
    code_round = rounds - 1
    folder = SOURCE_RUN / f"round_{code_round:03d}"
    commit_path = folder / "commit.json"
    commit = json.loads(commit_path.read_text())
    adapter_path = folder / "checkpoints/tts_final"
    if commit.get("round") != code_round or not commit.get("same_round_start"):
        raise ValueError(f"Invalid committed round: {commit_path}")
    record = commit.get("tts", {})
    if Path(record.get("path", "")).resolve() != adapter_path.resolve():
        raise ValueError(f"Commit TTS path mismatch: {commit_path}")
    if record.get("sha256") != checkpoint_hash(adapter_path):
        raise ValueError(f"Commit TTS directory hash mismatch: {adapter_path}")
    adapter = describe_adapter(adapter_path, MODEL)
    return {"completed_rounds": rounds, "code_round": code_round,
            "commit_path": str(commit_path), "commit_sha256": sha256_file(commit_path),
            "checkpoint_sha256": record["sha256"], "adapter": adapter}


def source_hashes():
    paths = [Path(__file__), BENCH / "run_midasheng_tts_local_8gpu.sh",
             BENCH / "model_adapter_utils.py", TOOLS / "generate.py",
             TOOLS / "staged_local_metrics.py", TOOLS / "staged_common.py",
             TOOLS / "run_inference.sh", TOOLS / "run_staged_local_metrics.sh",
             EVAL_ROOT / "prompts.py", EVAL_ROOT / "utils_eval.py"]
    for folder in ("quality_assessment", "wer_utils", "whisper_normalizer"):
        paths.extend((EVAL_ROOT / folder).rglob("*.py"))
    return {str(p): sha256_file(p) for p in sorted(set(paths))}


def check_resources():
    errors, identities = [], {}
    for rounds in (6, 9):
        try:
            identities[rounds] = checkpoint_identity(rounds)
            print(f"[CHECK] round {rounds}: commit/base/adapter hashes OK", flush=True)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"round {rounds}: {exc}")
    try:
        if len(load_eval_samples(DATA)) != 1645:
            raise ValueError("Expected exactly 1645 benchmark samples")
        for p in (MODEL / "config.json", MOS, QWEN_PYTHON, EVAL_PYTHON):
            with p.open("rb") as stream:
                stream.read(1)
        # Offline HF snapshots used by the existing local scorer.
        cache = EVAL_ROOT / "model"
        for repo in ("models--openai--whisper-large-v3", "models--facebook--wav2vec2-base"):
            revision = (cache / repo / "refs/main").read_text().strip()
            snapshot = cache / repo / "snapshots" / revision
            if not (snapshot / "config.json").is_file():
                raise FileNotFoundError(f"Missing cached model config: {snapshot}")
            weights = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("pytorch_model*.bin"))
            if not weights or not all(p.is_file() for p in weights):
                raise FileNotFoundError(f"Missing cached model weights: {snapshot}")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    if errors:
        raise RuntimeError("Precheck failed:\n" + "\n".join(errors) +
                           "\nFor permission errors use the training-owner commands in README_MIDASHENG_TTS_LOCAL.md.")
    return identities


def check_imports():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    for python, code in (
        (QWEN_PYTHON, "import torch, qwen_tts, peft, soundfile; print('TTS imports OK')"),
        (EVAL_PYTHON, "import sys; sys.path.insert(0,sys.argv[1]); import staged_local_metrics; print('Local scorer imports OK')"),
    ):
        subprocess.run([str(python), "-c", code, str(TOOLS)], env=env, check=True)


def expected_generation(identity, mode):
    return {"benchmark": "EmergentTTS-Eval",
            "benchmark_commit": "fe02c5d27f2d87751a66e32631aaa73a92d9cbbb",
            "model_path": str(MODEL), "adapter": identity["adapter"],
            "data_path": str(DATA), "data_sha256": sha256_file(DATA),
            "protocol": "strong_prompting", "seed": 42, "temperature": 1.0,
            "top_p": 0.9, "max_new_tokens": 8192, "batch_size": 1 if mode == "smoke" else 8,
            "attn": "sdpa", "num_samples": 1 if mode == "smoke" else None}


def validate_generation(folder, generation):
    if sha256_file(DATA) != generation["data_sha256"]:
        raise ValueError("Benchmark dataset changed since generation")
    manifest = json.loads((folder / "generation_manifest.json").read_text())
    if manifest.get("generation_identity_sha256") != sha256_json(generation):
        raise ValueError(f"Generation identity mismatch: {folder}")
    if manifest.get("generation_identity") != generation:
        raise ValueError(f"Generation manifest contents mismatch: {folder}")
    samples = load_eval_samples(DATA, num_samples=generation["num_samples"], seed=42)
    ids = [s["unique_id_eval"] for s in samples]
    if manifest.get("expected_samples") != len(ids):
        raise ValueError("Generation sample count mismatch")
    for i in ids:
        with wave.open(str(folder / "audios" / f"{i}.wav"), "rb") as wav:
            if wav.getnframes() <= 0 or wav.getnchannels() != 1:
                raise ValueError(f"Invalid WAV for sample {i}")
    return ids


def summarize_case(folder):
    identity = json.loads((folder / "local_run_identity.json").read_text())
    generation = identity["generation"]
    if identity["completed_rounds"] not in (6, 9):
        raise ValueError("Unexpected round")
    ids = validate_generation(folder, generation)
    stage = folder / "staged_evaluation"
    metadata = json.loads((stage / "local_metadata.json").read_text())
    expected = {"stage": "local_metrics", "format_version": 1,
                "dataset_sha256": generation["data_sha256"],
                "selected_ids_sha256": sha256_json(ids), "selected_samples": len(ids),
                "seed": 42, "num_samples": generation["num_samples"],
                "audio_dir": str(folder / "audios"),
                "whisper_model": "openai/whisper-large-v3",
                "wvmos_checkpoint": str(MOS), "wvmos_checkpoint_sha256": identity["wvmos_sha256"]}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Local metadata mismatch: {key} in {stage}")
    records_path = stage / "local_metrics.jsonl"
    records = load_latest_records(records_path, "local_metrics")
    if set(records) != set(ids):
        raise ValueError(f"Missing/extra local sample IDs: {stage}")
    for i in ids:
        row = records[i]
        if row.get("status") != "success":
            raise ValueError(f"Failed local sample {i}")
        if Path(row["audio_out_path"]).resolve() != folder / "audios" / f"{i}.wav":
            raise ValueError(f"Wrong audio source for sample {i}")
        if not all(math.isfinite(float(row[k])) for k in ("wer", "mos_score")):
            raise ValueError(f"Nonfinite local score for sample {i}")
    result = {"status": "local_complete", "completed_rounds": identity["completed_rounds"],
              "code_round": identity["code_round"], "samples": len(ids),
              "wer_percent": sum(float(records[i]["wer"]) for i in ids) / len(ids),
              "wvmos": sum(float(records[i]["mos_score"]) for i in ids) / len(ids),
              "win_rate": None, "judge_api_called_by_this_launcher": False,
              "wer_aggregation": "mean_utterance_percentage", "records_path": str(records_path),
              "records_sha256": sha256_file(records_path), "identity": identity}
    write_json(folder / "local_summary.json", result)
    return result


def write_summary(root, mode):
    results, errors = [], []
    for rounds in (6, 9):
        folder = root / mode / f"rounds_{rounds:02d}"
        try:
            results.append(summarize_case(folder))
        except (OSError, ValueError, KeyError, EOFError, wave.Error) as exc:
            # Invalidate any previous completion summary if its inputs no longer validate.
            if (folder / "local_run_identity.json").is_file():
                write_json(folder / "local_summary.json", {"status": "incomplete", "error": str(exc)})
            errors.append({"completed_rounds": rounds, "error": str(exc)})
    payload = {"mode": mode, "status": "local_complete" if not errors else "incomplete",
               "results": results, "errors": errors, "win_rate": None}
    write_json(root / mode / "summary.json", payload)
    with (root / mode / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["completed_rounds", "code_round", "samples", "wer_percent", "wvmos", "win_rate"])
        for r in results:
            writer.writerow([r[k] for k in ("completed_rounds", "code_round", "samples", "wer_percent", "wvmos", "win_rate")])
    lines = ["# MiDasheng original: local TTS evaluation", "",
             "| Completed rounds | Code checkpoint | Samples | WER (%) ↓ | WVMOS ↑ | Win Rate |",
             "|---|---|---:|---:|---:|---|"]
    for r in results:
        lines.append(f"| {r['completed_rounds']} | r{r['code_round']} | {r['samples']} | {r['wer_percent']:.2f} | {r['wvmos']:.3f} | — |")
    lines += ["", "Local metrics only; no API judge. Smoke results are not full benchmark results."]
    lines += [f"\nIncomplete round {e['completed_rounds']}: {e['error']}" for e in errors]
    (root / mode / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[SUMMARY] {root / mode / 'summary.md'}", flush=True)
    return 1 if errors else 0


def commands(folder, identity, mode):
    extra = ["--num-samples", "1"] if mode == "smoke" else []
    generate = ["bash", str(TOOLS / "run_inference.sh"), "--model-path", str(MODEL),
                "--adapter-dir", identity["adapter"]["path"], "--output-dir", str(folder / "audios"),
                "--batch-size", "1" if mode == "smoke" else "8", "--seed", "42",
                "--attn", "sdpa", "--temperature", "1.0", "--top-p", "0.9",
                "--max-new-tokens", "8192", *extra]
    local = ["bash", str(TOOLS / "run_staged_local_metrics.sh"), "--audio-dir", str(folder / "audios"),
             "--output-dir", str(folder / "staged_evaluation"), "--workers", "4",
             "--devices", "cuda:0,cuda:1,cuda:2,cuda:3", "--seed", "42", *extra]
    return generate, local


class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.processes = set()
        self.stopping = False

    def stop(self, *_):
        with self.lock:
            self.stopping = True
            for p in self.processes:
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def execute(self, cmd, env, log):
        with log.open("a") as stream:
            stream.write("\nCOMMAND " + json.dumps(cmd) + "\n")
            stream.flush()
            with self.lock:
                if self.stopping:
                    raise RuntimeError("Interrupted")
                p = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.processes.add(p)
            try:
                code = p.wait()
            finally:
                with self.lock:
                    self.processes.discard(p)
            if code:
                raise RuntimeError(f"Subprocess exit {code}; see {log}")

    def case(self, root, mode, identity, gpu_group, hashes):
        folder = root / mode / f"rounds_{identity['completed_rounds']:02d}"
        expected = {**identity, "mode": mode, "generation": expected_generation(identity, mode),
                    "source_sha256": hashes, "wvmos_sha256": sha256_file(MOS),
                    "gpu_group": gpu_group, "qwen_python": str(QWEN_PYTHON), "eval_python": str(EVAL_PYTHON)}
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / "local_run_identity.json"
        if marker.exists():
            if json.loads(marker.read_text()) != expected:
                raise ValueError(f"Resume identity changed: {folder}; use a new output root")
        elif any(folder.iterdir()):
            raise ValueError(f"Refusing nonempty unowned output: {folder}")
        else:
            write_json(marker, expected)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpu_group),
               "QWEN_PYTHON": str(QWEN_PYTHON), "EVAL_PYTHON": str(EVAL_PYTHON),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        generate, local = commands(folder, identity, mode)
        try:
            validate_generation(folder, expected["generation"])
            print(f"[SKIP] round {identity['completed_rounds']} audio already complete", flush=True)
        except (OSError, ValueError, KeyError, EOFError, wave.Error):
            print(f"[GENERATE] round {identity['completed_rounds']} on GPU {gpu_group[0]}; {folder / 'generation.log'}", flush=True)
            self.execute(generate, env, folder / "generation.log")
            validate_generation(folder, expected["generation"])
        print(f"[LOCAL] round {identity['completed_rounds']} GPUs {gpu_group}; {folder / 'local_metrics.log'}", flush=True)
        self.execute(local, env, folder / "local_metrics.log")
        result = summarize_case(folder)
        print(f"[DONE] round {identity['completed_rounds']} WER={result['wer_percent']:.4f}% WVMOS={result['wvmos']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check", "smoke", "full", "summarize"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default=os.environ.get("EVAL_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")))
    parser.add_argument("--mode", choices=("smoke", "full"), default="full", help="For summarize only")
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    if args.action == "plan":
        for rounds in (6, 9):
            print(f"Completed round {rounds}: {SOURCE_RUN / f'round_{rounds-1:03d}/checkpoints/tts_final'}")
        print(f"Output: {root}; full = 1645 samples per checkpoint; local metrics only; no round 12")
        return 0
    if args.action == "check":
        check_resources()
        check_imports()
        print("[CHECK] CPU precheck passed; no models loaded, no GPU jobs started")
        return 0
    if not args.output_root.is_absolute():
        raise ValueError("--output-root must be absolute")
    # Keep all new writes under a dedicated benchmark directory, away from training.
    if root == BENCH or BENCH not in root.parents:
        raise ValueError("Output must be a new dedicated subdirectory of Caption/benchmark")
    if args.action != "summarize":
        identities = check_resources()
        check_imports()
        gpus = [x.strip() for x in args.gpus.split(",")]
        if len(gpus) != 8 or len(set(gpus)) != 8 or not all(x.isdigit() or x.startswith("GPU-") for x in gpus):
            raise ValueError("Exactly eight unique GPU indices or GPU UUIDs are required")
        subprocess.run([str(QWEN_PYTHON), "-c", "import torch; assert torch.cuda.device_count()==8, 'Eight visible GPUs required'"],
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}, check=True)
    marker = root / "local_eval_root.json"
    if args.action == "summarize" and not marker.is_file():
        raise ValueError("No initialized evaluation root to summarize")
    if root.exists() and not marker.exists() and any(root.iterdir()):
        raise ValueError("Refusing nonempty directory without this launcher's root marker")
    os.umask(0o002)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected_root = {"launcher": "midasheng_original_local_r5_r8_v1", "source_run": str(SOURCE_RUN)}
        if marker.exists() and json.loads(marker.read_text()) != expected_root:
            raise ValueError("Output root belongs to a different experiment")
        write_json(marker, expected_root)
        if args.action == "summarize":
            return write_summary(root, args.mode)
        runner = Runner()
        signal.signal(signal.SIGINT, runner.stop)
        signal.signal(signal.SIGTERM, runner.stop)
        failed = False
        with ThreadPoolExecutor(max_workers=2) as pool:
            hashes = source_hashes()
            jobs = [pool.submit(runner.case, root, args.action, identities[r], gpus[4*i:4*i+4], hashes)
                    for i, r in enumerate((6, 9))]
            for job in as_completed(jobs):
                try:
                    job.result()
                except Exception as exc:
                    failed = True
                    print(f"[FAILED] {exc}", file=sys.stderr, flush=True)
        summary_code = write_summary(root, args.action)
        return 1 if failed or runner.stopping or summary_code else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
