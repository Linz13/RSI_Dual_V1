"""Frozen V4 committed rounds + V5 round-one Captioner benchmarks on four GPUs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time

import later_caption_eval as existing

ROOT = Path(__file__).resolve().parent
CAPTION = ROOT.parent
ENVS = CAPTION.parent / "miniconda3/envs"
MIDAPY = ENVS / "midasheng-captioner/bin/python"
METRICPY = ENVS / "emotiontalk-metrics/bin/python"
ET = ROOT / "emotiontalk_speech_captioning"
PSC = ROOT / "paraspeechcaps/evaluations/qwen3_captioner_attr6"
SC = ROOT / "stylecap_promptspeech_mcq"
V4 = CAPTION / "DualISL_Train_RewardV4/runs/midasheng_7b_reward_v4_10rounds_optimized_run01"
V5 = CAPTION / "DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01"
OUTPUT = ROOT / "midasheng_v4_all_v5_round1_caption_4gpu_20260910_run01"
VERSION = "midasheng-v4-v5-caption-four-gpu-v1"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        temporary = Path(f.name)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    try:
        temporary.chmod(0o666)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def discover():
    numbers = sorted(int(p.parent.name.split("_")[1]) for p in V4.glob("round_[0-9][0-9][0-9]/commit.json"))
    if not numbers:
        raise RuntimeError("No committed V4 rounds")
    candidates = existing.discover_reward_candidates(CAPTION,
        [(V4, "midasheng_rewardv4", numbers), (V5, "midasheng_rewardv5", [0])], True, True)
    for c in candidates:
        if c.status != "ready":
            raise RuntimeError(c.candidate_id + ": " + c.detail)
        summary = existing.read_json(Path(c.adapter).parents[1] / "summary.json")
        if summary["output_captioner"] != {"path": c.adapter, "sha256": c.detail}:
            raise RuntimeError("Summary/commit mismatch: " + c.candidate_id)
    return candidates


def validate_candidates(candidates):
    for c in candidates:
        ok, detail = existing.adapter_check(Path(c.adapter), Path(c.model))
        if not ok:
            raise RuntimeError(detail)
        commit = existing.read_json(Path(c.adapter).parents[1] / "commit.json")
        if (commit.get("round") != c.round or commit["captioner"] != {"path": c.adapter, "sha256": c.detail}
                or existing.hash_path(Path(c.adapter)) != c.detail):
            raise RuntimeError("Frozen checkpoint changed: " + c.candidate_id)


def resource_identity():
    data = [ET / "data/test_inference.jsonl", ET / "data/test_references.jsonl",
            PSC / "runs/new_cluster_default/manifest.jsonl", SC / "data/benchmark.jsonl"]
    code = [Path(__file__), ROOT / "later_caption_eval.py", ROOT / "model_adapter_utils.py",
            ROOT / "run_midasheng_v4_v5_benchmarks_4gpu.sh"]
    for base in (ET, PSC, SC):
        code.extend(base.glob("*.py"))
        for sub in ("backends", "prompts"):
            if (base / sub).is_dir():
                code.extend(p for p in (base / sub).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p): existing.sha256_file(p) for p in sorted(set(data + code))}


def inventory(output):
    path = output / "inventory.json"
    if path.exists():
        record = existing.read_json(path)
        if record["version"] != VERSION or record["resources"] != resource_identity():
            raise RuntimeError("Benchmark code/data changed; use a new output directory")
        candidates = [existing.Candidate(**c) for c in record["candidates"]]
        validate_candidates(candidates)
        return record, candidates
    candidates = discover()
    return {"version": VERSION, "created_utc": existing.utc_now(),
            "candidates": [asdict(c) for c in candidates], "resources": resource_identity(),
            "protocol": {"emotiontalk": "standard_public", "paraspeechcaps": "Scheme A attr6",
                         "stylecap": "speaker-open MCQ", "midasheng_attn": "sdpa",
                         "emotiontalk_batch": 4, "emotiontalk_max_new_tokens": 128, "stylecap_batch": 4}}, candidates


def cpu_check(candidates):
    validate_candidates(candidates)
    paths = [(ET / "data/test_inference.jsonl", ET, 7716),
             (PSC / "runs/new_cluster_default/manifest.jsonl", PSC, 140),
             (SC / "data/benchmark.jsonl", SC, 3112)]
    audio = set()
    for manifest, base, expected in paths:
        records = rows(manifest)
        if len(records) != expected:
            raise RuntimeError(f"Unexpected benchmark count: {manifest}: {len(records)}")
        for row in records:
            p = Path(row["audio_path"])
            audio.add(p if p.is_absolute() else base / p)
    missing = [str(p) for p in audio if not p.is_file() or not os.access(p, os.R_OK)]
    if missing:
        raise RuntimeError("Unreadable benchmark audio: " + str(missing[:5]))
    for py, imports in ((MIDAPY, "import torch, transformers, peft, soundfile"),
                        (METRICPY, "import torch, aac_metrics, transformers, sentence_transformers")):
        subprocess.run([str(py), "-c", imports], env=environment("", OUTPUT, "check"), check=True,
                       stdout=subprocess.DEVNULL, timeout=120)
    for p in (ENVS / "emotiontalk-metrics/bin/java", ET / "cache/aac_metrics", ET / "cache/huggingface"):
        if not p.exists():
            raise RuntimeError("Missing local scoring resource: " + str(p))
    return {"cpu_preflight": "passed", "gpu_execution": "not_performed", "checkpoints": len(candidates),
            "full_benchmark_tasks": 3 * len(candidates), "benchmark_audio_files": len(audio)}


def parse_gpus(spec):
    parts = spec.split(",")
    if len(parts) != 4 or any(not p.isdigit() for p in parts) or len(set(parts)) != 4:
        raise ValueError("EVAL_GPUS must contain four distinct physical GPU indices")
    return parts


def gpu_check(gpus):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], text=True)
    visible = {s.strip() for s in output.splitlines()}
    if not set(gpus) <= visible:
        raise RuntimeError("Selected GPU indices are not visible: " + str(gpus))
    env = environment(",".join(gpus), OUTPUT, "check")
    code = ("import torch,json; assert torch.cuda.device_count()==4; "
            "print(json.dumps({'gpu_check':'passed','devices':[torch.cuda.get_device_name(i) for i in range(4)]}))")
    subprocess.run([str(MIDAPY), "-c", code], env=env, check=True)


def environment(gpu, output, name):
    env = dict(os.environ)
    cache = ET / "cache"
    env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
        AAC_METRICS_CACHE_PATH=str(cache / "aac_metrics"), HF_HOME=str(cache / "huggingface"),
        HF_DATASETS_CACHE=str(cache / "huggingface/datasets"),
        TRANSFORMERS_CACHE=str(cache / "huggingface/transformers"),
        SENTENCE_TRANSFORMERS_HOME=str(cache / "huggingface/sentence_transformers"),
        XDG_CACHE_HOME=str(cache / "xdg"), AAC_METRICS_DEVICE="cuda_if_available",
        AAC_METRICS_TMP_PATH=str(output / "metrics_tmp" / name), JAVA_HOME=str(ENVS / "emotiontalk-metrics"))
    env["PATH"] = str(METRICPY.parent) + os.pathsep + env.get("PATH", "")
    return env


def commands(c, suite, size, out):
    py, model, adapter = c.python, c.model, c.adapter
    if suite == "emotiontalk":
        infer = [py, str(ET / "run_inference.py"), "--backend", "midasheng", "--tasks", "all",
                 "--gpu", "GPU", "--model-path", model, "--adapter-dir", adapter, "--attn-backend", "sdpa",
                 "--batch-size", "4", "--max-new-tokens", "128", "--output-dir", str(out), "--resume"]
        score = [str(METRICPY), str(ET / "evaluate.py"), "--predictions", str(out / "predictions.jsonl"),
                 "--references", str(ET / "data/test_references.jsonl"), "--profile", "standard_public",
                 "--device", "cuda_if_available", "--output-dir", str(out / "metrics_standard_public")]
        if size == "smoke":
            infer += ["--max-samples", "1"]
            score += ["--allow-partial"]
        return [infer, score]
    if suite == "paraspeechcaps":
        infer = [py, str(PSC / "run_content_scheme_a.py"), "--manifest", str(PSC / "runs/new_cluster_default/manifest.jsonl"),
                 "--output-dir", str(out / "outputs"), "--candidate-name", c.candidate_id,
                 "--backend", "midasheng", "--model-dir", model, "--adapter-dir", adapter,
                 "--attn-backend", "sdpa", "--resume"]
        if size == "smoke":
            infer += ["--max-samples", "1"]
        return [infer, [py, str(PSC / "score_content_scheme_a.py"), "--manifest", str(out / "outputs/selected_manifest.jsonl"),
                        "--predictions", str(out / "outputs/predictions.jsonl"), "--output-dir", str(out / "reports")]]
    infer = [py, str(SC / "run_midasheng.py"), "--backend", "midasheng", "--benchmark", str(SC / "data/benchmark.jsonl"),
             "--model-dir", model, "--adapter-dir", adapter, "--output-dir", str(out), "--attn-backend", "sdpa",
             "--batch-size", "4", "--resume"]
    if size == "smoke":
        return [infer + ["--max-questions", "4"]]
    return [infer, [py, str(SC / "evaluate.py"), "--benchmark", str(SC / "data/benchmark.jsonl"),
                    "--predictions", str(out / "predictions.jsonl"), "--output", str(out / "evaluation_summary.json")]]


def task_status(output, suite, size, c):
    ok, detail = existing.completion_status(output, suite, size, c.candidate_id, c)
    if not ok:
        return ok, detail
    out = existing.output_dir(output, suite, size, c.candidate_id)
    pred = rows(out / ("outputs/predictions.jsonl" if suite == "paraspeechcaps" else "predictions.jsonl"))
    key = (lambda r: (r.get("id"), r.get("task"))) if suite == "emotiontalk" else (
          (lambda r: r.get("sample_id")) if suite == "paraspeechcaps" else (lambda r: r.get("question_id")))
    keys = [key(r) for r in pred]
    if len(keys) != len(set(keys)):
        return False, "Duplicate prediction IDs"
    if suite == "emotiontalk":
        scores = existing.load_suite_result(suite, out)
        for name, task in scores["tasks"].items():
            for metric in ("spider", "fense"):
                data = task["metrics"].get(metric, {})
                if data.get("status") != "ok" or data.get("value") is None:
                    return False, f"Required EmotionTalk metric unavailable: {name}/{metric}"
        if any(not isinstance(r.get("prediction"), str) or not r["prediction"].strip() for r in pred):
            return False, "Empty EmotionTalk prediction"
    return True, detail


def summarize(output, candidates):
    entries = []
    for c in candidates:
        for suite in existing.BENCHMARKS:
            ok, detail = task_status(output, suite, "full", c)
            out = existing.output_dir(output, suite, "full", c.candidate_id)
            entries.append({"candidate": c.candidate_id, "round_index": c.round, "training_round": c.round + 1,
                            "suite": suite, "state": "complete" if ok else "incomplete", "detail": detail,
                            "result_path": str(out), "result": existing.load_suite_result(suite, out) if ok else {}})
    write_json(output / "summary.json", {"entries": entries, "created_utc": existing.utc_now()})
    lines = ["# MiDasheng V4 / V5 Captioner 评测", "", "轮次为从 1 开始的训练轮次；r0 即第一轮。缺失指标保持缺失。", "",
             "| 模型 | 训练轮次 | ET SPIDEr | ET FENSE | ParaSpeechCaps | StyleCap |", "|---|---:|---:|---:|---:|---:|"]
    table = []
    for c in candidates:
        selected = {e["suite"]: e for e in entries if e["candidate"] == c.candidate_id}
        et = selected["emotiontalk"]["result"].get("tasks", {}).get("overall", {}).get("metrics", {})
        values = [et.get("spider", {}).get("value"), et.get("fense", {}).get("value"),
                  selected["paraspeechcaps"]["result"].get("final_score"),
                  selected["stylecap"]["result"].get("macro_average_accuracy")]
        table.append([c.candidate_id, c.round + 1, *values])
        lines.append(f"| {c.candidate_id} | {c.round + 1} | " + " | ".join("—" if v is None else f"{v:.6f}" for v in values) + " |")
    lines += ["", "EmotionTalk 沿用 standard_public 口径；完整指标及不可用原因见各任务报告，不把缺失值记为零。",
              "不同训练轮数不属于等预算对照；V4 第一轮与 V5 第一轮可作同轮次比较。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate", "training_round", "et_spider", "et_fense", "paraspeechcaps", "stylecap"])
        writer.writerows(table)
    return entries


def ordered_tasks(candidates):
    # Start all three V5 benchmarks immediately; keep the fourth GPU busy with V4.
    priority = [c for c in candidates if c.trajectory == "midasheng_rewardv5"]
    remaining = [c for c in candidates if c.trajectory != "midasheng_rewardv5"]
    return ([(c, suite) for c in priority for suite in existing.BENCHMARKS]
            + [(c, suite) for suite in existing.BENCHMARKS for c in remaining])


class Runner:
    def __init__(self, output):
        self.output = output
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.children = set()

    def cancel(self):
        self.stop.set()
        with self.lock:
            children = list(self.children)
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        while any(p.poll() is None for p in children) and time.monotonic() < deadline:
            time.sleep(.2)
        for p in children:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def run_command(self, command, env, log):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError("Evaluation cancelled")
            p = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.children.add(p)
        try:
            code = p.wait()
            if code:
                raise RuntimeError(f"Worker exited with code {code}; see log")
        finally:
            with self.lock:
                self.children.discard(p)

    def stage(self, candidates, size, gpus):
        tasks = queue.Queue()
        for c, suite in ordered_tasks(candidates):
            if not task_status(self.output, suite, size, c)[0]:
                tasks.put((c, suite))
        failures = []
        def worker(gpu):
            while not self.stop.is_set():
                try:
                    c, suite = tasks.get_nowait()
                except queue.Empty:
                    return
                name = f"{size}_{suite}_{c.candidate_id}"
                out = existing.output_dir(self.output, suite, size, c.candidate_id)
                out.mkdir(parents=True, exist_ok=True)
                logpath = self.output / "logs" / (name + ".log")
                logpath.parent.mkdir(exist_ok=True)
                (self.output / "metrics_tmp" / name).mkdir(parents=True, exist_ok=True)
                print(f"START GPU {gpu}: {name}; log={logpath}", flush=True)
                started = time.monotonic()
                try:
                    with logpath.open("a") as log:
                        sizes = ["smoke", "full"] if size == "full" else ["smoke"]
                        for phase in sizes:
                            if task_status(self.output, suite, phase, c)[0]:
                                continue
                            phase_out = existing.output_dir(self.output, suite, phase, c.candidate_id)
                            phase_out.mkdir(parents=True, exist_ok=True)
                            for command in commands(c, suite, phase, phase_out):
                                command = [gpu if v == "GPU" else v for v in command]
                                log.write("COMMAND " + json.dumps(command) + "\n")
                                log.flush()
                                self.run_command(command, environment(gpu, self.output, name), log)
                            ok, detail = task_status(self.output, suite, phase, c)
                            if not ok:
                                raise RuntimeError(phase + ": " + detail)
                    ok, detail = task_status(self.output, suite, size, c)
                    if not ok:
                        raise RuntimeError(detail)
                    elapsed = round(time.monotonic() - started, 2)
                    write_json(out / "launcher_timing.json", {"seconds_this_invocation": elapsed,
                               "completed_utc": existing.utc_now(), "gpu": gpu})
                    print(f"DONE GPU {gpu}: {name}; elapsed={elapsed / 60:.1f} min", flush=True)
                    with self.lock:
                        summarize(self.output, candidates)
                except Exception as exc:
                    with self.lock:
                        failures.append({"task": name, "error": str(exc), "log": str(logpath)})
                    print(f"FAILED GPU {gpu}: {name}: {exc}", flush=True)
        pool = ThreadPoolExecutor(max_workers=4)
        try:
            futures = [pool.submit(worker, gpu) for gpu in gpus]
            for f in futures:
                f.result()
        except BaseException:
            self.cancel()
            raise
        finally:
            pool.shutdown(wait=True)
        write_json(self.output / (size + "_execution.json"), {"failures": failures, "gpus": gpus})
        if failures:
            raise RuntimeError(f"{len(failures)} {size} tasks failed; rerun the same command to retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "run", "smoke", "status", "summarize"])
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    os.umask(0)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".launcher.lock").open("a") as lock:
        if args.mode in ("check", "run", "smoke"):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record, candidates = inventory(output)
        if args.mode in ("check", "run", "smoke"):
            report = cpu_check(candidates)
            write_json(output / "inventory.json", record)
            write_json(output / "cpu_preflight.json", report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            print("Frozen candidates: " + ", ".join(c.candidate_id for c in candidates), flush=True)
        if args.mode == "check":
            return 0
        if args.mode in ("run", "smoke"):
            gpus = parse_gpus(os.getenv("EVAL_GPUS", "4,5,6,7"))
            gpu_check(gpus)
            runner = Runner(output)
            def terminate(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, terminate)
            try:
                runner.stage(candidates, "full" if args.mode == "run" else "smoke", gpus)
            finally:
                runner.cancel()
                summarize(output, candidates)
        else:
            entries = summarize(output, candidates)
            print(json.dumps({"full_complete": sum(e["state"] == "complete" for e in entries),
                              "full_total": len(entries), "summary": str(output / "summary.md")}, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Evaluation stopped; completed results preserved. Rerun the same command to resume.", flush=True)
        raise SystemExit(130)
