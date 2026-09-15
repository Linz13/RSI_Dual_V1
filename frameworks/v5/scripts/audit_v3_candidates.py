"""CPU-only replay of stored V3 reward inputs through V4; no model inference."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dual_isl_train.io import load_yaml
from dual_isl_train.rewards import score_groups

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent / "DualISL_Train_RewardV3/runs/midasheng_7b_reward_v3_10rounds_20260905_run01"


def main():
    reward = load_yaml(SOURCE / "resolved_config.yaml")["reward"]
    calibration = json.loads((SOURCE / "reward_calibration.json").read_text())
    records, hashes = [], {}
    checked = 0
    unchanged = ("semantic_valid", "semantic_reward", "reward_components_raw",
                 "reward_components_normalized", "reward", "advantage", "grpo_mode", "skip_update")
    for r in range(10):
        directory = SOURCE / f"round_{r:03d}/rewards"
        thresholds = json.loads((directory / f"round_{r:03d}_sft_thresholds.json").read_text())
        for loop, suffix, model, expected in (("audio_only", "audio", "TTS", 184),
                                               ("caption_only", "caption", "Captioner", 196)):
            path = directory / f"round_{r:03d}_{suffix}_rewards.output.jsonl"
            digest = hashlib.sha256()
            groups = eligible = actual = selected = diagnostic = 0
            loop_reward = {**reward[loop], **{k: reward[k] for k in ("anchor_penalty_weight", "anchor_tolerance_z")}}
            with path.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    if not line.strip():
                        continue
                    old = json.loads(line)
                    new = score_groups([old], loop=loop, calibration=calibration,
                                       reward_config=loop_reward,
                                       sft_threshold=thresholds["loops"][loop]["threshold"],
                                       quality=reward["validity"][loop])[0]
                    valid = [c for c in old["candidates"] if c["semantic_valid"]]
                    expected_winner = max(valid, key=lambda c: (float(c["semantic_reward"]), str(c["candidate_id"]))) if valid else None
                    winners = [c["candidate_id"] for c in new["candidates"] if c["sft_selected"]]
                    assert winners == ([expected_winner["candidate_id"]] if expected_winner else [])
                    for a, b in zip(old["candidates"], new["candidates"], strict=True):
                        for key in unchanged:
                            # JSON comparison treats corresponding NaN diagnostics identically.
                            assert json.dumps(a.get(key), sort_keys=True) == json.dumps(b.get(key), sort_keys=True), (r, loop, a["candidate_id"], key)
                        for key in ("raw_text", "prompt", "sampled_token_ids", "old_token_logprobs", "ref_token_logprobs", "codec_codes"):
                            assert a.get(key) == b.get(key), (r, loop, key)
                        assert bool(a["sft_selected"]) == b["sft_gate_would_select"]
                        checked += 1
                    groups += 1
                    eligible += bool(valid)
                    actual += sum(c["sft_selected"] for c in old["candidates"])
                    selected += len(winners)
                    diagnostic += sum(c["sft_gate_would_select"] for c in new["candidates"])
            assert groups == expected and selected == eligible and diagnostic == actual
            hashes[str(path)] = digest.hexdigest()
            records.append({"round": r, "loop": loop, "target_model": model, "groups": groups,
                            "v3_actual_selected": actual, "v4_selected": selected,
                            "diagnostic_would_select": diagnostic})
            print(f"r{r} {model}: V3={actual} V4={selected}; reward/GRPO unchanged", flush=True)
    for path in (SOURCE / "resolved_config.yaml", SOURCE / "reward_calibration.json", PROJECT / "dual_isl_train/rewards.py"):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"status": "passed", "time_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "executable": sys.executable,
              "source_run": str(SOURCE), "checked_candidates": checked,
              "unchanged_fields": unchanged, "rounds": records, "source_sha256": hashes,
              "limitation": "Historical candidates only; no prediction of a new training trajectory or pseudo-pair correctness."}
    output = PROJECT / "reports/historical_v3_candidate_replay.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
