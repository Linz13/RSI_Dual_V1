"""CPU-only audit of saved stage timings and replay workloads; no model loading."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import statistics

HERE = Path(__file__).resolve().parent
CAPTION = HERE.parents[2]
RUNS = {
    "midasheng_v4_8gpu": CAPTION / "DualISL_Train_RewardV4/runs/midasheng_7b_reward_v4_10rounds_optimized_run01",
    "qwen25_v4_4gpu": CAPTION / "DualISL_Train_RewardV4/runs/qwen25_omni_3b_reward_v4_10rounds_4gpu_run01",
    "qwen25_v2_8gpu": CAPTION / "DualISL_Train_RewardV2/runs/dual_recursive_8gpu_h100_qwen2_5_omni_3b_reward_v2_20260903_run02",
}
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def jsonlines(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                yield json.loads(line)
    SOURCES[str(path)] = digest.hexdigest()


def audit_run(directory):
    result = {"path": str(directory), "rounds": [], "unreadable": []}
    for identity_path in directory.glob("*identity.json"):
        identity = read(identity_path)
        if "code_sha256" not in identity or "config" not in identity:
            continue
        config = identity["config"]
        result["config"] = {
            "world_size": config["distributed"]["world_size"],
            "captioner_model": config["captioner"]["model_path"],
            "tts_model": config["tts"]["model_path"],
            "captioner_generation": config["captioner"].get("generation", {}),
            "captioner_training": config["captioner"].get("training", {}),
        }
        mismatches = []
        for relative, expected in identity["code_sha256"].items():
            path = directory.parent.parent / relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                mismatches.append(relative)
        result["code_identity"] = {"files": len(identity["code_sha256"]), "mismatches": mismatches}
        break
    for commit in sorted(directory.glob("round_*/commit.json")):
        round_dir = commit.parent
        stages = {}
        for path in sorted(round_dir.glob("**/*.stage.json")):
            try:
                stage = read(path)
            except PermissionError:
                result["unreadable"].append(str(path))
                continue
            seconds = stage.get("elapsed_seconds")
            if stage.get("status") == "complete" and isinstance(seconds, (int, float)):
                prefix = round_dir.name + "_"
                assert stage["name"].startswith(prefix)
                name = stage["name"][len(prefix):]
                assert name not in stages, (round_dir, name)
                stages[name] = seconds / 60
        result["rounds"].append({
            "code_round": round_dir.name,
            "completed_round": int(round_dir.name.split("_")[-1]) + 1,
            "stage_minutes": stages,
            "timed_stage_count": len(stages),
            "total_minutes": sum(stages.values()),
        })
    result["round_001_rollout"] = {}
    for tag, count_key in [("audio_caption_rollout", "sampled_token_ids"), ("caption_tts_rollout", "codec_codes")]:
        path = directory / "round_001/collections" / f"round_001_{tag}.output.jsonl"
        sizes, reference_modes, groups, eos, error = [], Counter(), 0, 0, 0.0
        for row in jsonlines(path):
            groups += 1
            for candidate in row["candidates"]:
                sizes.append(len(candidate[count_key]))
                reference_modes[candidate.get("reference_replay_mode", "not_recorded")] += 1
                eos += bool(candidate.get("terminated_by_eos"))
                error = max(error, candidate.get("policy_reference_max_abs_error", 0))
        metrics = read(Path(str(path) + ".metrics.json"))
        ranks = metrics.get("per_rank", [])
        result["round_001_rollout"][tag] = {
            "groups": groups, "candidates": len(sizes), "length_sum": sum(sizes),
            "length_mean": statistics.mean(sizes), "length_max": max(sizes),
            "caption_eos_count": eos if count_key == "sampled_token_ids" else None,
            "reference_modes": dict(reference_modes),
            "caption_policy_reference_max_error": error if count_key == "sampled_token_ids" else None,
            "ranks": [{k: r[k] for k in ("rank", "processed_rows", "inference_seconds", "gpu_peak_memory_bytes") if k in r} for r in ranks],
        }
    if "v4" in directory.name:
        forward, backward, candidate_count, step_seconds = 0.0, 0.0, 0, []
        for path in sorted((directory / "round_001/checkpoints/tts_after_grpo/training_progress").glob("rank_*.jsonl")):
            started, ready = {}, {}
            for event in jsonlines(path):
                key = (event.get("epoch"), event.get("step"), event.get("candidate_id"))
                if event["event"] == "candidate_start":
                    started[key] = event["timestamp"]
                elif event["event"] == "candidate_loss_ready":
                    ready[key] = event["timestamp"]
                elif event["event"] == "candidate_complete" and key in started and key in ready:
                    forward += ready[key] - started.pop(key)
                    backward += event["timestamp"] - ready.pop(key)
                    candidate_count += 1
                elif event["event"] == "step_complete":
                    step_seconds.append(event["elapsed_seconds"])
        result["round_001_tts_grpo_host_intervals"] = {
            "candidates_including_padding": candidate_count,
            "forward_loss_seconds_sum_all_ranks": forward,
            "backward_sync_seconds_sum_all_ranks": backward,
            "forward_fraction": forward / (forward + backward),
            "backward_fraction": backward / (forward + backward),
            "step_seconds_sum_all_ranks": sum(step_seconds),
            "warning": "Host timestamps across all ranks: includes logging and waiting; not pure CUDA kernels, communication, or global wall time.",
        }
    return result


report = {name: audit_run(path) for name, path in RUNS.items()}
report["source_sha256"] = SOURCES
(HERE / "evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
lines = ["# Saved stage timings", "", "Minutes; successful timed subprocess stages only. Excludes untimed bookkeeping, initial preparation and failed work.", "",
         "| Run | Completed round | Timed stages | Total | TTS rollout | TTS GRPO | Captioner rollout | Captioner GRPO | Both SFT |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for name in RUNS:
    entry = report[name]
    for row in entry["rounds"]:
        s = row["stage_minutes"]
        values = [row["total_minutes"], s.get("caption_tts_rollout", 0), s.get("tts_grpo", 0), s.get("audio_caption_rollout", 0), s.get("caption_grpo", 0), s.get("caption_sft", 0) + s.get("tts_sft", 0)]
        lines.append(f"| {name} | {row['completed_round']} | {row['timed_stage_count']} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
(HERE / "stage_times.md").write_text("\n".join(lines) + "\n")
for name in RUNS:
    entry = report[name]
    print(name, "round_minutes", [round(x["total_minutes"], 2) for x in entry["rounds"]], "code_identity", entry.get("code_identity"))
    if "round_001_tts_grpo_host_intervals" in entry:
        print(entry["round_001_tts_grpo_host_intervals"])
print("Wrote evidence.json and stage_times.md; no GPU or training changes.")
