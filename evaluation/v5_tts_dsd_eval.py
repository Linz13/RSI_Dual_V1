#!/usr/bin/env python3
"""V5 committed-round TTS: resumable, sharded DSD generation and Gemini evaluation."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CAPTION = ROOT.parent
CLUSTER = CAPTION.parent
PIPELINE = ROOT / "InstructTTSEval-public/qwen3_voice_design"
sys.path.insert(0, str(PIPELINE))
from common import (atomic_write_json, load_samples, selected_samples, record_key,
                    sha256_file, sha256_json, load_latest_records, DATASET_REVISION, OFFICIAL_COMMIT)
sys.path.insert(0, str(ROOT))
from model_adapter_utils import describe_adapter

ROUND = CAPTION / "DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01/round_000"
BASE = CAPTION / "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DATA = PIPELINE.parent / "data/manifests"
TTS_PY = CLUSTER / "miniconda3/envs/qwen3-tts/bin/python"
EVAL_PY = CLUSTER / "miniconda3/envs/emergent-tts-eval/bin/python"
OUTPUT = PIPELINE / "runs/v5_midasheng_round000_dsd_seed42"
BASE_RUN = PIPELINE / "runs/base/full_bilingual_seed42"
JUDGE_DIR = "evaluation_gemini_2_5_pro"
JUDGE_MODEL = "models/gemini-2.5-pro"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_hash(path):
    # Same directory hash as the training commit; logs under training_metrics
    # are deliberately excluded by that protocol.
    return sha256_json([
        {"path": str(p.relative_to(path)), "sha256": sha256_file(p)}
        for p in sorted(path.rglob("*"))
        if p.is_file() and "training_metrics" not in p.parts
    ])


def parse_gpus(value):
    gpus = value.split(",")
    if not gpus or any(not p.isdigit() for p in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("DSD_GPUS must be comma-separated, distinct physical GPU indices")
    return gpus


def make_plan(args):
    if args.batch_size < 1 or args.workers < 1 or args.gpu_memory_gib <= 0:
        raise ValueError("Batch size, workers and GPU memory budget must be positive")
    gpus = parse_gpus(args.gpus)
    round_index = getattr(args, "round_index", 0)
    if round_index < 0:
        raise ValueError("round-index must be nonnegative (0 means the first round)")
    round_dir = ROUND.parent / f"round_{round_index:03d}"
    commit = read_json(round_dir / "commit.json")
    adapter_path = round_dir / "checkpoints/tts_final"
    if commit["round"] != round_index or Path(commit["tts"]["path"]).resolve() != adapter_path.resolve():
        raise RuntimeError("Unexpected round/checkpoint in commit.json")
    if checkpoint_hash(adapter_path) != commit["tts"]["sha256"]:
        raise RuntimeError("TTS checkpoint hash differs from the selected committed round")
    adapter = describe_adapter(adapter_path, BASE)
    metadata = read_json(DATA.parent / "metadata.json")
    if metadata["dataset_revision"] != DATASET_REVISION:
        raise RuntimeError("Dataset revision changed")
    hashes = {lang: sha256_file(DATA / f"{lang}.jsonl") for lang in ("en", "zh")}
    for lang in hashes:
        if hashes[lang] != metadata["manifests"][lang]["sha256"]:
            raise RuntimeError(f"Dataset manifest hash mismatch: {lang}")
    samples = load_samples(DATA, ("en", "zh"))
    counts = {lang: sum(r["language"] == lang for r in samples) for lang in ("en", "zh")}
    if counts != {"en": 1000, "zh": 1000}:
        raise RuntimeError(f"Unexpected DSD dataset size: {counts}")
    limit = math.ceil(len(gpus) * args.batch_size / 2) if args.mode == "smoke" else None
    samples = selected_samples(samples, limit)
    if len(gpus) > len(samples):
        raise ValueError("More GPU shards than samples")
    base_identity = read_json(BASE_RUN / "generation_manifest.json")["generation_identity"]
    base_judge = read_json(BASE_RUN / JUDGE_DIR / "judge_metadata.json")
    prompt_hash = sha256_file(PIPELINE.parent / "eval/eval_prompt.txt")
    if (base_judge["judge_model"] != JUDGE_MODEL or base_judge["backend"] != "inline"
            or base_judge["prompt_sha256"] != prompt_hash):
        raise RuntimeError("Base judge protocol differs from this DSD launcher")
    decoding = {key: base_identity[key] for key in
                ("seed", "temperature", "top_p", "max_new_tokens", "attention")}
    plan = {
        "version": f"v5-round{round_index:03d}-dsd-v1", "round": round_index, "tasks": ["DSD"],
        "checkpoint_commit_sha256": commit["tts"]["sha256"], "adapter": adapter,
        "dataset_revision": DATASET_REVISION, "manifest_hashes": hashes,
        "selected_ids": [r["id"] for r in samples], "expected_audios": len(samples),
        "num_shards": len(gpus), "batch_size": args.batch_size, "decoding": decoding,
        "seed_strategy": "sha256_seed_and_batch_keys_v1",
        "judge_model": JUDGE_MODEL, "judge_backend": "inline", "prompt_sha256": prompt_hash,
        "code_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                        (PIPELINE / "generate.py", PIPELINE / "judge.py",
                         PIPELINE / "common.py", ROOT / "model_adapter_utils.py")},
    }
    return plan, samples, gpus, limit


def environment(gpu=""):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    return env


def preflight():
    # Shared deployment hosts need not have the git executable installed.
    git_dir = PIPELINE.parent / ".git"
    head = (git_dir / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:]
        loose = git_dir / ref
        if loose.is_file():
            head = loose.read_text().strip()
        else:
            head = next((line.split()[0] for line in (git_dir / "packed-refs").read_text().splitlines()
                         if line.endswith(" " + ref)), "")
    if head != OFFICIAL_COMMIT:
        raise RuntimeError("InstructTTSEval repository revision mismatch")
    for name in ("config.json", "model.safetensors"):
        if not (BASE / name).is_file() or not (BASE / name).stat().st_size:
            raise FileNotFoundError(BASE / name)
    code = ("import torch, soundfile, peft, numpy; from qwen_tts import Qwen3TTSModel; "
            "import importlib.metadata as m; "
            "print({p:m.version(p) for p in ('qwen-tts','torch','peft','soundfile')})")
    subprocess.run([str(TTS_PY), "-c", code], env=environment(), check=True)
    subprocess.run([str(EVAL_PY), str(PIPELINE / "judge.py"), "--check-config",
                    "--backend", "inline"], env=environment(), check=True)


def check_gpu_memory(gpus, budget):
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,memory.total,name",
        "--format=csv,noheader,nounits"], text=True)
    devices = {}
    for line in raw.strip().splitlines():
        index, free, total, name = [s.strip() for s in line.split(",", 3)]
        devices[index] = (float(free) / 1024, float(total) / 1024, name)
    for gpu in gpus:
        if gpu not in devices:
            raise RuntimeError(f"Physical GPU {gpu} is not visible; check DSD_GPUS")
        free, total, name = devices[gpu]
        print(f"[GPU {gpu}] {name}: free={free:.1f}/{total:.1f} GiB; TTS budget={budget:g} GiB", flush=True)
        if free < budget + 2:
            raise RuntimeError(f"GPU {gpu} needs at least {budget + 2:g} GiB free at startup; "
                               "choose fewer/other GPU IDs or wait for the other job")


def lock_plan(output, plan):
    path = output / "dsd_plan.json"
    if path.exists() and read_json(path) != plan:
        raise RuntimeError("Output directory has a different DSD plan. Restore its shard/batch settings "
                           "or use a new output directory; existing results were preserved.")
    atomic_write_json(path, plan)


def generation_command(output, plan, index, limit, budget):
    decoding = plan["decoding"]
    command = [str(TTS_PY), str(PIPELINE / "generate.py"),
               "--model-path", str(BASE), "--adapter-dir", plan["adapter"]["path"],
               "--manifest-dir", str(DATA), "--output-dir", str(output / "shards" / f"{index:02d}"),
               "--languages", "en", "zh", "--tasks", "DSD", "--device", "cuda:0",
               "--num-shards", str(plan["num_shards"]), "--shard-index", str(index),
               "--batch-size", str(plan["batch_size"]), "--seed-per-batch",
               "--gpu-memory-gib", str(budget), "--seed", str(decoding["seed"]),
               "--temperature", str(decoding["temperature"]), "--top-p", str(decoding["top_p"]),
               "--max-new-tokens", str(decoding["max_new_tokens"]), "--attn", decoding["attention"]]
    if limit is not None:
        command += ["--num-samples-per-language", str(limit)]
    return command


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def generate(output, plan, gpus, limit, budget):
    check_gpu_memory(gpus, budget)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    processes, logs = [], []
    started = time.monotonic()
    try:
        for index, gpu in enumerate(gpus):
            logpath = output / "logs" / f"generate_{index:02d}_gpu{gpu}.log"
            log = logpath.open("a", buffering=1)
            logs.append(log)
            command = generation_command(output, plan, index, limit, budget)
            log.write("COMMAND " + json.dumps(command) + "\n")
            processes.append(subprocess.Popen(command, env=environment(gpu), stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True))
            print(f"[START] shard={index} GPU={gpu} log={logpath}", flush=True)
        next_update = 0
        while True:
            statuses = [p.poll() for p in processes]
            if any(code not in (None, 0) for code in statuses):
                raise RuntimeError(f"Generation worker failed: {statuses}; inspect logs/. "
                                   "Successful WAVs are resumable; other jobs were not stopped.")
            now = time.monotonic()
            if now >= next_update:
                print_status(output, plan["expected_audios"])
                next_update = now + 30
            if all(code == 0 for code in statuses):
                break
            time.sleep(1)
    finally:
        stop_processes(processes)
        for log in logs:
            log.close()
    print(f"[GENERATION DONE] elapsed={(time.monotonic() - started) / 60:.1f} minutes", flush=True)


def merge(output, plan, samples):
    items, children, template = [], [], None
    for index in range(plan["num_shards"]):
        path = output / "shards" / f"{index:02d}" / "generation_manifest.json"
        child = read_json(path)
        identity = dict(child["generation_identity"])
        if sha256_json(identity) != child["generation_identity_sha256"]:
            raise RuntimeError(f"Corrupt generation identity: {path}")
        expected_ids = [r["id"] for r in samples[index::plan["num_shards"]]]
        if (not child.get("complete") or identity["selected_ids"] != expected_ids
                or identity["tasks"] != ["DSD"] or identity["adapter"] != plan["adapter"]
                or identity["batch_size"] != plan["batch_size"]
                or identity.get("seed_strategy") != plan["seed_strategy"]
                or any(identity[k] != v for k, v in plan["decoding"].items())):
            raise RuntimeError(f"Incomplete or mismatched DSD shard: {path}")
        for lang, digest in plan["manifest_hashes"].items():
            if identity["data_files"][lang]["sha256"] != digest:
                raise RuntimeError(f"DSD shard dataset mismatch: {path}")
        identity.pop("selected_ids")
        shard = identity.pop("shard", {"count": 1, "index": 0})
        if shard != {"count": plan["num_shards"], "index": index}:
            raise RuntimeError(f"Shard index/count mismatch: {path}")
        if template is None:
            template = identity
        elif template != identity:
            raise RuntimeError("Generation shards use different models/software/settings")
        children.append(child["generation_identity_sha256"])
        items.extend(child["items"])
    expected = {(r["language"], r["id"], "DSD"): r for r in samples}
    if len(items) != len(expected) or {record_key(r) for r in items} != set(expected):
        raise RuntimeError("DSD merge has missing, extra or duplicate items")
    for item in items:
        ref = expected[record_key(item)]
        path = Path(item["audio_path"])
        if (item["status"] not in ("generated", "existing") or item["text"] != ref["text"]
                or item["instruction"] != ref["DSD"] or sha256_file(path) != item["audio_sha256"]):
            raise RuntimeError(f"DSD content/audio mismatch: {record_key(item)}")
    identity = {**template, "selected_ids": plan["selected_ids"], "shard_identities": children}
    manifest = {"benchmark": "InstructTTSEval", "subtask": "DSD", "complete": True,
                "generation_identity": identity, "generation_identity_sha256": sha256_json(identity),
                "expected_audios": len(items), "items": sorted(items, key=record_key)}
    atomic_write_json(output / "generation_manifest.json", manifest)
    return manifest


def judge(output, workers, dry_run=False):
    destination = output / ("dry_run" if dry_run else JUDGE_DIR)
    command = [str(EVAL_PY), str(PIPELINE / "judge.py"),
               "--generation-manifest", str(output / "generation_manifest.json"),
               "--output-dir", str(destination), "--model", JUDGE_MODEL,
               "--backend", "inline", "--workers", str(workers), "--allow-incomplete"]
    command += ["--dry-run"] if dry_run else ["--confirm-paid", "--retry-failed"]
    # Default five bounded attempts/request. Reruns only retry missing/failed judgments.
    process = subprocess.Popen(command, env=environment(), start_new_session=True)
    try:
        if process.wait() != 0:
            raise RuntimeError("DSD judge failed; completed judgments remain resumable")
    finally:
        stop_processes([process])


def score(output, dry_run=False):
    destination = output / ("dry_run" if dry_run else JUDGE_DIR)
    command = [str(EVAL_PY), str(PIPELINE / "score.py"),
               "--generation-manifest", str(output / "generation_manifest.json"),
               "--judge-results", str(destination / "judge_results.jsonl"),
               "--output", str(destination / "summary.json"), "--allow-incomplete"]
    if dry_run:
        command.append("--allow-dry-run")
    subprocess.run(command, env=environment(), check=True, stdout=subprocess.DEVNULL)
    summary = read_json(destination / "summary.json")
    base_summary = read_json(BASE_RUN / JUDGE_DIR / "summary.json")
    metrics = {lang: summary["metrics"][lang]["DSD"] for lang in ("en", "zh")}
    base_scores = {lang: base_summary["metrics"][lang]["DSD"]["percentage"] for lang in ("en", "zh")}
    values = [metrics[lang]["percentage"] for lang in metrics]
    bilingual = round(sum(values) / 2, 4) if all(v is not None for v in values) else None
    base_bilingual = round(sum(base_scores.values()) / 2, 4)
    usage = summary["usage"]
    report = {
        "benchmark": "InstructTTSEval DSD",
        "round": read_json(output / "dsd_plan.json")["round"] if (output / "dsd_plan.json").exists() else 0,
        "dry_run": dry_run,
        "complete": summary["complete"], "expected": summary["expected"],
        "scored": summary["scored"], "coverage_percentage": summary["coverage_percentage"],
        "metrics": metrics, "bilingual_percentage": bilingual,
        "base_percentage": {**base_scores, "bilingual": base_bilingual},
        "delta_percentage_points": round(bilingual - base_bilingual, 4)
            if summary["complete"] and not dry_run and bilingual is not None else None,
        "judge_model": JUDGE_MODEL, "official_exact_reproduction": False,
        "usage": {**usage, "budget_usd_to_cny": 7.0,
                  "estimated_cost_cny": round(usage["estimated_cost_usd"] * 7, 2)},
        "notes": ["Only DSD; not the full three-task benchmark.",
                  "Same base judge model, prompt and inline audio backend.",
                  "Base used batch=8 and a global seed; this run uses fixed batches and batch-key seeds. "
                  "Sampling trajectories differ; small score deltas can include sampling/judge variation.",
                  "Cost is a list-price estimate for recorded successful responses, not an API invoice."],
    }
    atomic_write_json(destination / "dsd_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[REPORT] {destination / 'dsd_summary.json'}", flush=True)
    return 0 if summary["complete"] else 1


def print_status(output, expected=2000):
    valid = failed = 0
    for path in sorted((output / "shards").glob("*/generation_manifest.json")):
        child = read_json(path)
        valid += sum(r["status"] in ("generated", "existing") for r in child["items"])
        failed += sum(r["status"] == "failed" for r in child["items"])
    latest = load_latest_records(output / JUDGE_DIR / "judge_results.jsonl")
    judged = sum(r.get("status") == "success" and not r.get("dry_run") for r in latest.values())
    print(f"[STATUS] generated={valid}/{expected} generation_failed={failed} judged={judged}/{expected}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "smoke", "run", "generate", "judge", "score", "status"),
                        help="run: generate + paid Gemini judge; smoke: local GPU + offline simulated judge")
    parser.add_argument("output_dir", nargs="?", type=Path)
    parser.add_argument("--round-index", type=int, default=0, help="Zero based: 1 selects the second round")
    parser.add_argument("--gpus", default=os.getenv("DSD_GPUS", "4,5,6,7"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("DSD_BATCH_SIZE", "8")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("DSD_JUDGE_WORKERS", "32")))
    parser.add_argument("--gpu-memory-gib", type=float, default=float(os.getenv("DSD_GPU_MEMORY_GIB", "32")))
    args = parser.parse_args()
    os.umask(0o000)  # Artifacts are shared across GPU hosts with different numeric UIDs.
    output = (args.output_dir or PIPELINE / f"runs/v5_midasheng_round{args.round_index:03d}_dsd_seed42").resolve()
    if args.mode == "smoke":
        output /= "smoke"
    if args.mode == "status":
        print_status(output)
        return 0
    plan, samples, gpus, limit = make_plan(args)
    if args.mode == "check":
        preflight()
        print(json.dumps({"cpu_preflight": "passed", "round": plan["round"], "expected_dsd": len(samples),
                          "checkpoint_commit_sha256": plan["checkpoint_commit_sha256"],
                          "gpus_requested": gpus, "batch_size": args.batch_size,
                          "gpu_execution": "not_performed", "api_requests": 0}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dsd.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another DSD launcher is using this output directory") from exc
        lock_plan(output, plan)
        if args.mode in ("run", "generate", "smoke"):
            preflight()
            manifests = [output / "shards" / f"{i:02d}" / "generation_manifest.json"
                         for i in range(plan["num_shards"])]
            if all(p.is_file() and read_json(p).get("complete") for p in manifests):
                print("[RESUME] All generation shards are complete; verifying audio before judging.", flush=True)
            else:
                generate(output, plan, gpus, limit, args.gpu_memory_gib)
        merge(output, plan, samples)
        if args.mode in ("run", "judge", "smoke"):
            judge(output, args.workers, dry_run=args.mode == "smoke")
        if args.mode in ("run", "judge", "score", "smoke"):
            return score(output, dry_run=args.mode == "smoke")
    return 0


if __name__ == "__main__":
    def stopped(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stopped)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[STOPPED] Completed WAVs/judgments preserved. Rerun the same command to resume.", flush=True)
        raise SystemExit(130)
