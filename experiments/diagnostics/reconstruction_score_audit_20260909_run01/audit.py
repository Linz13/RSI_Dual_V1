"""CPU-only audit of saved audio-only reward records; never imports training code.

Writes derived CSV/JSON beside this script. Completed reward stages may precede
round commit. Permission-denied inputs are recorded, never bypassed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median, pstdev

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUNS = {
    "midasheng_v1_initial": "DualISL_Train/runs/dual_recursive_4gpu_h100_midasheng_20260830_run01",
    "midasheng_v1_continuation": "DualISL_Train/runs/dual_recursive_8gpu_h100_midasheng_from_r2_20260831_run01",
    "midasheng_v2": "DualISL_Train_RewardV2/runs/dual_recursive_8gpu_h100_midasheng_reward_v2_20260901_run01",
    "midasheng_v3": "DualISL_Train_RewardV3/runs/midasheng_7b_reward_v3_10rounds_20260905_run01",
    "midasheng_v4": "DualISL_Train_RewardV4/runs/midasheng_7b_reward_v4_10rounds_optimized_run01",
    "qwen25_v4_4gpu": "DualISL_Train_RewardV4/runs/qwen25_omni_3b_reward_v4_10rounds_4gpu_run01",
}
SOURCES, LIMITS, CHECKS = {}, [], []


def read(path, jsonl=False):
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
    SOURCES[str(path)] = {"sha256": hashlib.sha256(data).hexdigest(),
                          "bytes": len(data), "mtime_ns": after.st_mtime_ns}
    if jsonl:
        assert not data or data.endswith(b"\n"), path
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    return json.loads(data)


def stage_records(path):
    meta = path.with_name(path.name.replace(".output.jsonl", ".stage.json"))
    stage = read(meta)
    if stage.get("status") not in {"complete", "reused"}:
        LIMITS.append({"path": str(path), "reason": "stage_not_complete"})
        return None
    return read(path, jsonl=True)


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def quantile(values, q):
    a = sorted(values)
    k = (len(a) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return a[lo] + (a[hi] - a[lo]) * (k - lo)


def stats(values):
    a = [float(v) for v in values if finite(v)]
    if not a:
        return {"n": 0}
    return {"n": len(a), "mean": fmean(a), "std": pstdev(a), "min": min(a),
            "p05": quantile(a, .05), "median": median(a), "p95": quantile(a, .95), "max": max(a)}


def auc(positive, negative):
    return fmean((p > n) + .5 * (p == n) for p in positive for n in negative)


def flat(prefix, values):
    return {prefix + "_" + k: v for k, v in stats(values).items()}


def dump_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = datetime.now(timezone.utc).isoformat()
    per_round, comparisons, calibrations = [], [], {}
    source_files = [ROOT / "DualISL_Train_RewardV4" / "dual_isl_train" / name for name in
                    ("rewards.py", "orchestrator.py", "workers/qwen_voice_design.py")]
    code_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    for run_id, relative in RUNS.items():
        run = ROOT / relative
        try:
            calibration = read(run / "reward_calibration.json")
            calibrations[run_id] = calibration
        except (OSError, ValueError) as exc:
            LIMITS.append({"path": str(run / "reward_calibration.json"), "reason": type(exc).__name__})
            calibration = None
        for p in sorted(run.glob("round_*/rewards/*audio_rewards.output.jsonl")):
            try:
                groups = stage_records(p)
            except (OSError, ValueError) as exc:
                LIMITS.append({"path": str(p), "reason": type(exc).__name__})
                continue
            if groups is None:
                continue
            rd = p.parent.parent
            candidates = [c for g in groups for c in g["candidates"]]
            assert len({c["candidate_id"] for c in candidates}) == len(candidates)
            assert len(groups) == 184 and len(candidates) == 736
            valid = [c for c in candidates if c.get("semantic_valid") is True]
            rec = lambda c: c.get("reward_components_raw", {}).get("reconstruction")
            assert all(finite(rec(c)) for c in valid)
            modes, gaps, agreement = Counter(), [], []
            errors_rec, errors_z, errors_reward, errors_advantage = [], [], [], []
            zstats = calibration["loops"]["audio_only"]["reconstruction"]
            for c in valid:
                if finite(c.get("tts_main_logprob")) and finite(c.get("tts_sub_logprob")):
                    errors_rec.append(abs(rec(c) - c["tts_main_logprob"] - .3*c["tts_sub_logprob"]))
                z = (rec(c)-zstats["mean"])/zstats["std"] if zstats["std"] >= 1e-12 else 0.
                n = c["reward_components_normalized"]
                errors_z.append(abs(z-n["reconstruction"]))
                errors_reward.append(abs(.5*n["reconstruction"]+.5*n["counterfactual"]
                                         -n["anchor_penalty"]-c["semantic_reward"]))
            for g in groups:
                eligible = [c for c in g["candidates"] if c.get("semantic_valid") is True]
                active = [c for c in g["candidates"] if not c.get("skip_update", True)]
                mode = active[0].get("grpo_mode") if active else "skipped"
                modes[mode] += 1
                if len(eligible) >= 2:
                    gaps.append(max(map(rec, eligible))-min(map(rec, eligible)))
                    rec_winner = max(eligible, key=lambda c:(rec(c),str(c["candidate_id"])))
                    reward_winner = max(eligible, key=lambda c:(c["semantic_reward"],str(c["candidate_id"])))
                    agreement.append(rec_winner["candidate_id"] == reward_winner["candidate_id"])
                if active:
                    values = [c["reward"] for c in active]
                    std, mean = pstdev(values), fmean(values)
                    for c in active:
                        expected = (c["reward"]-mean)/std if std else 0.
                        errors_advantage.append(abs(c["advantage"]-expected))
            row = {"run": run_id, "round": int(rd.name.split("_")[1]),
                   "round_commit_exists": (rd/"commit.json").exists(),
                   "groups": len(groups), "candidates": len(candidates), "semantic_valid": len(valid),
                   "semantic_groups": modes["dual_semantic"], "format_groups": modes["schema_curriculum"],
                   "sft_selected": sum(bool(c.get("sft_selected")) for c in candidates),
                   **flat("all_finite_reconstruction", map(rec,candidates)),
                   **flat("valid_reconstruction", map(rec,valid)),
                   **flat("valid_z_reconstruction", [c["reward_components_normalized"]["reconstruction"] for c in valid]),
                   **flat("valid_counterfactual", [c["reward_components_raw"]["counterfactual"] for c in valid]),
                   **flat("valid_semantic_reward", [c["semantic_reward"] for c in valid]),
                   **flat("within_group_rec_range", gaps),
                   "rec_top1_matches_reward_top1": fmean(agreement) if agreement else None,
                   "rec_comparison_groups": len(agreement),
                   "anchor_penalty_positive_candidates": sum(c["reward_components_normalized"]["anchor_penalty"]>0 for c in valid)}
            cp = p.parent / (rd.name+"_audio_only_sft_confidence_calibration.output.jsonl")
            try:
                examples = stage_records(cp)
                assert examples is not None
                by_id = {e["id"]:e for e in examples}
                assert len(by_id) == len(examples) == 78
                pairs = []
                for pos in examples:
                    if pos["label"] != 1:
                        continue
                    source_id = pos["id"].rsplit("::",1)[0]
                    neg = by_id[source_id+"::negative"]
                    assert neg["label"] == 0
                    pr, nr = pos["reward_components_raw"]["reconstruction"], neg["reward_components_raw"]["reconstruction"]
                    pair = {"run":run_id,"round":row["round"],"source_id":source_id,
                            "positive_rec":pr,"negative_rec":nr,"rec_difference":pr-nr,
                            "positive_reward":pos["score"],"negative_reward":neg["score"]}
                    comparisons.append(pair)
                    pairs.append(pair)
                assert len(pairs) == 39
                row.update({"paired_n":len(pairs),
                            "paired_rec_wins":sum(d["rec_difference"]>0 for d in pairs),
                            "paired_rec_ties":sum(d["rec_difference"]==0 for d in pairs),
                            "paired_reward_wins":sum(d["positive_reward"]>d["negative_reward"] for d in pairs),
                            "pooled_reconstruction_auc":auc([d["positive_rec"] for d in pairs],[d["negative_rec"] for d in pairs]),
                            "pooled_semantic_reward_auc":auc([d["positive_reward"] for d in pairs],[d["negative_reward"] for d in pairs]),
                            **flat("paired_positive_rec",[d["positive_rec"] for d in pairs]),
                            **flat("paired_negative_rec",[d["negative_rec"] for d in pairs]),
                            **flat("paired_rec_difference",[d["rec_difference"] for d in pairs])})
            except (OSError, ValueError) as exc:
                LIMITS.append({"path":str(cp),"reason":type(exc).__name__})
            for name, errors in [("reconstruction_formula",errors_rec),("zscore",errors_z),
                                 ("semantic_reward_formula",errors_reward),("advantage",errors_advantage)]:
                maximum = max(errors,default=0.)
                assert maximum < 1e-9, (run_id,rd.name,name,maximum)
                CHECKS.append({"run":run_id,"round":row["round"],"check":name,"count":len(errors),"max_abs_error":maximum})
            per_round.append(row)
    assert code_before == {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    dump_csv(OUT/"per_round.csv",per_round)
    dump_csv(OUT/"paired_comparisons.csv",comparisons)
    result = {"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),
              "scope":"audio-only TTS reconstruction for Captioner reward; saved completed stages only",
              "runs":RUNS,"calibrations":calibrations,"rounds":per_round,"sources":SOURCES,
              "limitations":LIMITS,"arithmetic_checks":CHECKS,"training_source_hashes_unchanged":code_before,
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/"evidence.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"rounds":len(per_round),"candidate_records":sum(r["candidates"] for r in per_round),
                      "paired_comparisons":len(comparisons),"sources":len(SOURCES),"limitations":len(LIMITS),
                      "arithmetic_checks":len(CHECKS)},ensure_ascii=False))


if __name__ == "__main__":
    main()
