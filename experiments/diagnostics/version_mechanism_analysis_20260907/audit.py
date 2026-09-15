"""Read-only training/benchmark audit; writes derived evidence beside this file."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


OUTPUT = Path(__file__).resolve().parent
CAPTION = OUTPUT.parents[2]
BENCHMARK = CAPTION / "benchmark"
EVALUATIONS = BENCHMARK / "later_caption_eval_runs_20260902_run01"
RUNS = {
    "v1_initial": "DualISL_Train/runs/dual_recursive_4gpu_h100_midasheng_20260830_run01",
    "v1_continuation": "DualISL_Train/runs/dual_recursive_8gpu_h100_midasheng_from_r2_20260831_run01",
    "v2": "DualISL_Train_RewardV2/runs/dual_recursive_8gpu_h100_midasheng_reward_v2_20260901_run01",
    "v3": "DualISL_Train_RewardV3/runs/midasheng_7b_reward_v3_10rounds_20260905_run01",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def auc(positive, negative):
    return sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def rank_metrics(directory, checkpoint):
    files = sorted((directory / "checkpoints" / checkpoint / "training_metrics").glob("rank_*/metrics.jsonl"))
    assert files, directory
    rows, steps = [], []
    for path in files:
        current = read_jsonl(path)
        assert len({(r["epoch"], r["step"]) for r in current}) == len(current), path
        rows.extend(r for r in current if not r.get("padded"))
        steps.append(max(r["step"] for r in current) + 1)
    assert len(set(steps)) == 1, steps
    return {"rows": len(rows), "anchors": sum(r["sample_id"].startswith("anchor::") for r in rows),
            "optimizer_steps": steps[0], "ranks": len(files), "sources": [str(p) for p in files]}


def calibration_diagnostics(rows):
    positive = [r for r in rows if r["label"] == 1]
    negative = [r for r in rows if r["label"] == 0]
    assert len(positive) == len(negative) == 39
    result = {}
    for name in ("score", "reconstruction", "counterfactual"):
        def value(row):
            return row["score"] if name == "score" else row["reward_components_raw"][name]
        result[name + "_auc"] = auc([value(r) for r in positive], [value(r) for r in negative])
    points = []
    for threshold in sorted({r["score"] for r in rows}):
        tp = sum(r["score"] >= threshold for r in positive)
        fp = sum(r["score"] >= threshold for r in negative)
        if tp / 39 >= .25:
            points.append((tp / (tp + fp), tp, fp))
    result["max_precision_at_recall_ge_025"] = max(points)[0]
    neg_by_id = {r["id"].rsplit("::", 1)[0]: r for r in negative}
    result["paired_reconstruction_positive_wins"] = sum(
        r["reward_components_raw"]["reconstruction"] > neg_by_id[r["id"].rsplit("::", 1)[0]]["reward_components_raw"]["reconstruction"]
        for r in positive
    )
    return result


def bootstrap_difference(a, b, clusters=None):
    ids = sorted(a)
    assert set(ids) == set(b)
    groups = defaultdict(list)
    for key in ids:
        groups[clusters[key] if clusters else key].append(a[key] - b[key])
    totals = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(20260907)
    sampled = rng.integers(0, len(groups), (20000, len(groups)))
    differences = totals[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    return {"difference": float(totals.sum() / sizes.sum()),
            "ci95": np.quantile(differences, [.025, .975]).tolist(),
            "units": len(groups), "observations": len(ids), "replicates": 20000, "seed": 20260907}


def main():
    evidence = {"created_utc": datetime.now(timezone.utc).isoformat(), "rounds": [], "permission_limits": []}
    csv_rows = []
    for run_name, relative in RUNS.items():
        run = CAPTION / relative
        version = "v1" if run_name.startswith("v1") else run_name
        for directory in sorted(run.glob("round_*")):
            number = int(directory.name.split("_")[1])
            if number > 9 or not (directory / "commit.json").exists():
                continue
            row = {"version": version, "round": number}
            detail = {**row, "run": str(run)}
            metrics = {stage: rank_metrics(directory, stage) for stage in ("caption_final", "caption_after_grpo")}
            detail["training_metrics"] = metrics
            sft = metrics["caption_final"]
            assert sft["anchors"] == 78
            row.update(caption_sft_total=sft["rows"], caption_sft_anchors=sft["anchors"],
                       caption_sft_cycle=sft["rows"] - sft["anchors"], caption_sft_steps=sft["optimizer_steps"],
                       world_size=sft["ranks"], caption_grpo_groups=metrics["caption_after_grpo"]["rows"])
            try:
                summary = read_json(directory / "summary.json")
                detail["summary"] = {k: summary[k] for k in ("audio_only", "caption_only", "sft_thresholds")}
            except PermissionError:
                summary = None
                evidence["permission_limits"].append(str(directory / "summary.json"))
            if version != "v1":
                assert summary is not None
                a, c = summary["audio_only"], summary["caption_only"]
                assert c["sft_selected"] == row["caption_sft_cycle"]
                row.update(semantic_groups=a["semantic_grpo_groups"], schema_groups=a["schema_curriculum_groups"],
                           semantic_candidates=a["structurally_valid_candidates"], tts_cycle_pairs=a["sft_selected"])
                if version == "v3":
                    row.update(raw_valid=a["raw_schema_valid_candidates"], input_valid=a["semantic_input_valid_candidates"],
                               nonzero_semantic_groups=a["semantic_nonzero_advantage_groups"])
                detail["calibration_diagnostics"] = {}
                for loop in ("audio_only", "caption_only"):
                    path = directory / f"rewards/round_{number:03d}_{loop}_sft_confidence_calibration.output.jsonl"
                    diagnosis = calibration_diagnostics(read_jsonl(path))
                    detail["calibration_diagnostics"][loop] = {**diagnosis, "source": str(path)}
                    row[loop + "_gate_auc"] = diagnosis["score_auc"]
                    row[loop + "_gate_status"] = summary["sft_thresholds"]["loops"][loop]["status"]
            csv_rows.append(row)
            evidence["rounds"].append(detail)
    evidence["totals"] = {}
    for version in ("v1", "v2", "v3"):
        rows = [r for r in csv_rows if r["version"] == version]
        assert len(rows) == 10
        evidence["totals"][version] = {k: sum(r[k] for r in rows) for k in
            ("caption_sft_total", "caption_sft_anchors", "caption_sft_cycle", "caption_sft_steps", "caption_grpo_groups")}
        if version != "v1":
            evidence["totals"][version].update({k: sum(r[k] for r in rows) for k in ("semantic_groups", "schema_groups", "semantic_candidates", "tts_cycle_pairs")})
            evidence["totals"][version]["round0_calibration"] = read_json(CAPTION / RUNS[version] / "reward_calibration.json")
    para, style = {}, {}
    evidence["benchmark_breakdown_r9"] = {}
    style_rows = read_jsonl(BENCHMARK / "stylecap_promptspeech_mcq/data/benchmark.jsonl")
    clusters = {r["question_id"]: r["speaker_id"] for r in style_rows}
    for version, candidate in (("base", None), ("v1", "midasheng_v1_r9"), ("v2", "midasheng_rewardv2_r9"), ("v3", "midasheng_rewardv3_r9")):
        para_root = EVALUATIONS / f"paraspeechcaps/full/{candidate}" if candidate else BENCHMARK / "round_completion_eval_runs_20260831/paraspeechcaps_content_scheme_a_v1/eval_midasheng_base_full_20260831_run01"
        style_root = EVALUATIONS / f"stylecap/full/{candidate}" if candidate else BENCHMARK / "stylecap_promptspeech_mcq/runs/midasheng_base_full_20260901_run01"
        with (para_root / "reports/per_sample.csv").open(encoding="utf-8-sig") as handle:
            sample_rows = list(csv.DictReader(handle))
        para[version] = {r["sample_id"]: float(r["sample_score"]) for r in sample_rows}
        predictions = {r["question_id"]: r["prediction"] for r in read_jsonl(style_root / "predictions.jsonl")}
        style[version] = {r["question_id"]: float(predictions[r["question_id"]] == r["answer"]) for r in style_rows}
        assert len(para[version]) == 140 and len(style[version]) == 3112
        assert abs(np.mean(list(para[version].values())) - read_json(para_root / "reports/summary.json")["final_score"]) < 1e-10
        assert abs(np.mean(list(style[version].values())) - read_json(style_root / "evaluation_summary.json")["macro_average_accuracy"]) < 1e-10
        evidence["benchmark_breakdown_r9"][version] = {
            "para_source": str(para_root), "style_source": str(style_root),
            "para_single_fields": {
                field: {"prediction_counts": dict(Counter(r["pred_" + field] for r in sample_rows)),
                        "label_counts": dict(Counter(r["gt_" + field] for r in sample_rows)),
                        "accuracy": float(np.mean([float(r["score_" + field]) for r in sample_rows]))}
                for field in ("gender", "pitch", "speaking_rate", "accent")
            },
            "style_tasks": {
                task: {"prediction_counts": dict(Counter(predictions[r["question_id"]] for r in style_rows if r["task"] == task)),
                       "correct": sum(style[version][r["question_id"]] for r in style_rows if r["task"] == task)}
                for task in ("gender", "pitch", "speaking_speed", "volume")
            },
        }
    evidence["paired_bootstrap_r9"] = {}
    for a, b in (("v1", "base"), ("v2", "base"), ("v3", "base"), ("v1", "v2"), ("v1", "v3"), ("v3", "v2")):
        evidence["paired_bootstrap_r9"][f"{a}_minus_{b}"] = {
            "paraspeechcaps_sample_resampling": bootstrap_difference(para[a], para[b]),
            "stylecap_speaker_cluster_resampling": bootstrap_difference(style[a], style[b], clusters),
        }
    (OUTPUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = list(dict.fromkeys(k for r in csv_rows for k in r))
    with (OUTPUT / "training_signal.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"totals": evidence["totals"], "bootstrap": evidence["paired_bootstrap_r9"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
