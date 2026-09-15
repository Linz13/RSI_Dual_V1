"""Read-only model/artifact verification; writes a report without executing models."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from dual_isl_train.attribute_reward import score_audio_groups
from dual_isl_train.checkpoints import checkpoint_record
from dual_isl_train.io import atomic_json, load_yaml, read_jsonl


def verify(run_dir):
    root = Path(run_dir)
    cfg = load_yaml(root / "resolved_config.yaml")
    mock = cfg["run"].get("inline_mock", False)
    errors, warnings, summaries = [], [], []
    def require(condition, message):
        if not condition:
            errors.append(message)
    require(cfg.get("version") == 5, "not a V5 run")
    calibration = json.loads((root / "reward_calibration.json").read_text())
    require(set(calibration["loops"]) == {"caption_only"}, "audio likelihood calibration still enabled")
    rounds = sorted(root.glob("round_[0-9][0-9][0-9]/summary.json"))
    require(bool(rounds), "no committed rounds")
    for path in rounds:
        rd = path.parent
        prefix = rd.name
        summary = json.loads(path.read_text())
        require(summary["collections_complete_before_update"], prefix + ": early optimizer update")
        commit = json.loads((rd / "commit.json").read_text())
        for role in ("captioner", "tts"):
            require(checkpoint_record(commit[role]["path"]) == commit[role], prefix + ": checkpoint hash mismatch " + role)
        audio = list(read_jsonl(rd / "rewards" / (prefix + "_audio_rewards.output.jsonl")))
        recomputed = score_audio_groups(audio, **{
            k: cfg["reward"]["audio_only"][k] for k in ("reconstruction_weight", "format_weight")})
        for g, expected in zip(audio, recomputed, strict=True):
            require(len(g["candidates"]) == 4, prefix + ": wrong candidate count")
            for c, e in zip(g["candidates"], expected["candidates"], strict=True):
                require(c["reward"] == e["reward"] and c["sft_selected"] == e["sft_selected"], c["candidate_id"] + ": wrong reward/selection")
        cap_rows = list(read_jsonl(rd / "training" / (prefix + "_caption_sft.input.jsonl")))
        tts_rows = list(read_jsonl(rd / "training" / (prefix + "_tts_sft.input.jsonl")))
        require(all(r["target_schema"] == "synth_v1" for r in cap_rows), prefix + ": original-schema caption anchor present")
        require(all(r["audio_path"] == r["source_audio_path"] for r in tts_rows), prefix + ": TTS target is not source audio")
        require(summary["caption_anchor_rows"] == sum(bool(r["is_anchor"]) for r in cap_rows), prefix + ": caption anchor count mismatch")
        require(summary["tts_anchor_rows"] == sum(bool(r["is_anchor"]) for r in tts_rows), prefix + ": TTS anchor count mismatch")
        require(summary["caption_anchor_rows"] == summary["tts_anchor_rows"], prefix + ": unequal anchors")
        candidates = [c for g in audio for c in g["candidates"]]
        generated = sum(bool(c.get("reconstructed_audio_path")) for c in candidates)
        if not mock:
            require(generated > 0, prefix + ": no audio-only synthesis/label reward exercised; inspect caption admission and trajectories")
            for name in ("caption_after_grpo", "caption_final", "tts_after_grpo", "tts_final"):
                meta = rd / "checkpoints" / name / "dual_isl_train_training.json"
                require(meta.is_file(), prefix + ": missing real training metadata " + name)
                if meta.is_file():
                    data = json.loads(meta.read_text())
                    loss = data.get("mean_loss")
                    if data.get("optimizer_steps", data.get("steps", 0)):
                        require(type(loss) in (int, float) and math.isfinite(loss), prefix + ": nonfinite/missing loss " + name)
                        require(data.get("parameter_before") != data.get("parameter_after"), prefix + ": optimizer did not change parameters " + name)
                    else:
                        warnings.append(prefix + ": no optimizer steps in " + name)
            batched = sum(c.get("generation_batch_size", 1) > 1 for c in candidates)
            if cfg["captioner"]["generation"]["rollout_batch_size"] > 1 and not batched:
                warnings.append(prefix + ": Captioner batching not exercised (fallback or unavailable metadata)")
            tts = list(read_jsonl(rd / "collections" / (prefix + "_caption_tts_rollout.output.jsonl")))
            fallback = sum(bool(g.get("batch_fallback")) for g in tts)
            if fallback:
                warnings.append(prefix + f": TTS rollout fell back to serial in {fallback} groups")
        summaries.append({"round": summary["round"], "generated_audio": generated,
            "caption_anchor_rows": summary["caption_anchor_rows"], "tts_anchor_rows": summary["tts_anchor_rows"],
            "audio_reward": summary["audio_only"]})
    report = {"ok": not errors, "scope": "CPU mock artifacts" if mock else "real run artifacts",
              "errors": errors, "warnings": warnings, "rounds": summaries}
    atomic_json(root / "v5_verification.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    report = verify(args.run_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)
