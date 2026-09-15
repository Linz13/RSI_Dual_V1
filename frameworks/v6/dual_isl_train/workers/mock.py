from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from dual_isl_train.constants import SOURCE_DOMAIN_TARGET, TRAJECTORY_VERSION
from dual_isl_train.dual_space import project_synth_caption, synth_caption_json
from dual_isl_train.schema import empty_caption
from dual_isl_train.trajectory import mark_trajectory
from dual_isl_train.workers.common import (
    load_job, save_checkpoint_marker, validate_checkpoint_version, worker_parser, write_output,
)


def mock_caption(index: int, transcript: str = "A deterministic mock utterance.") -> dict[str, Any]:
    value = empty_caption()
    value["semantic_content"].update(
        language="English", transcript=transcript, topic="testing", intent="statement",
    )
    value["speaker_profile"].update(
        gender="female" if index % 2 else "male", age="adult",
        timbre="breathy" if index == 3 else "neutral", accent="US English",
    )
    value["paralinguistic"].update(
        speaking_rate=("slow", "moderate", "fast", "moderate")[index % 4],
        pitch_level=("low", "medium", "high", "medium")[index % 4],
        volume_level="medium", emotion=("neutral", "happy", "sad", "neutral")[index % 4],
        emotion_intensity=("none", "low", "medium", "none")[index % 4],
        prosody="even declarative phrasing", pause="occasional short pauses",
        nonverbal_vocalization=["none"],
    )
    value["paralinguistic"]["emphasis"] = {"level": "none", "emphasized_text": []}
    value["environment"] = {
        "background_sound_events": ["none"], "recording_quality": "good", "acoustic_scene": "studio",
    }
    return value


def _candidate_index(candidate_id: str) -> int:
    for part in reversed(str(candidate_id).split("::")):
        if part.isdigit() and len(part) < 2:
            return int(part)
    return 0


def _mock_reconstruction(candidate_id: str, base: float, step: float) -> float:
    value = base + step * _candidate_index(candidate_id)
    if "sftcal::" in candidate_id:
        return base + (0.15 if candidate_id.endswith("::positive") else -0.25)
    if "::cf::" in candidate_id:
        return value - 0.25
    return value


def _checkpoint(action: str, rows: list[dict[str, Any]], checkpoint_in: str, checkpoint_out: str) -> list[dict[str, Any]]:
    for row in rows:
        if action == "sft-update" and row.get("target_origin") != SOURCE_DOMAIN_TARGET:
            raise ValueError("Mock worker rejected a non-source-domain SFT target")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    before = 1.0
    after = before + (0.1 if rows else 0.0)
    save_checkpoint_marker(checkpoint_out, "mock", checkpoint_in, {
        "action": action, "input_hash": digest, "rows": len(rows), "steps": len(rows),
        "parameter_before": before, "parameter_after": after, "mean_loss": 0.5,
        "max_grad_norm": 1.0 if rows else 0.0,
        "parameter_sync": {"ok": True, "max_abs_diff": 0.0},
    })
    return [{
        "status": "updated", "action": action, "rows": len(rows), "checkpoint": checkpoint_out,
        "parameter_before": before, "parameter_after": after, "mean_loss": 0.5,
    }]


def execute_mock(
    *, action: str, config: dict[str, Any], rows: list[dict[str, Any]], output: str,
    checkpoint_in: str = "", target_checkpoint: str = "", checkpoint_out: str = "", phase: str = "",
) -> None:
    del target_checkpoint, phase
    if checkpoint_in:
        validate_checkpoint_version(checkpoint_in)
    result: list[dict[str, Any]] = []
    if action == "prepare-codecs":
        cache = Path(config["tts"].get("codec_cache_dir", Path(output).parent / "codecs"))
        cache.mkdir(parents=True, exist_ok=True)
        for row in rows:
            path = cache / f"{row['id']}.json"
            path.write_text(
                json.dumps({"codec_codes": [[index for index in range(16)] for _ in range(4)]}),
                encoding="utf-8",
            )
            result.append({
                "id": row["id"], "audio_path": row["audio_path"],
                "codec_path": str(path), "cache_reused": False,
            })
    elif action == "generate-audio":
        for row in rows:
            audio_path = Path(output).parent / "audio" / (row["id"].replace("::", "_") + ".wav")
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            import wave
            with wave.open(str(audio_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(bytes(480))
            result.append({**row, "audio_path": str(audio_path), "purpose": "attribute_reward_only"})
    elif action == "rollout":
        for row in rows:
            candidates = []
            seeds = row.get("candidate_seeds") or [None] * int(row.get("group_size", 4))
            for index in range(int(row.get("group_size", 4))):
                candidate_id = f"{row['id']}::{index}"
                if "audio_path" in row:
                    caption = mock_caption(index)
                    if row.get("caption_schema") == "synth_v1":
                        caption = project_synth_caption(caption)
                    raw = synth_caption_json(caption) if row.get("caption_schema") == "synth_v1" else json.dumps({"Target_JSON_Schema": caption})
                    candidate = {
                        "candidate_id": candidate_id, "caption": caption, "canonical_text": raw,
                        "raw_text": raw, "parse_errors": [], "raw_schema_valid": True,
                        "normalized_schema_valid": True, "normalization_rules": [],
                        "generation_seed": seeds[index], "generation_attempts": 1,
                        "sampled_token_ids": [10 + index, 20 + index],
                        "old_token_logprobs": [-0.2 - index * 0.02, -0.1],
                        "ref_token_logprobs": [-0.22, -0.12],
                        "behavior_replay_max_abs_error": 0.0, "policy_reference_max_abs_error": 0.0,
                        "behavior_logprob_mode": "generation_scores", "trajectory_version": TRAJECTORY_VERSION,
                        "finish_reason": "eos", "terminal_token_id": 20 + index,
                        "terminated_by_eos": True, "post_eos_token_count": 0,
                        "generated_token_count": 2, "eos_token_id": 20 + index,
                    }
                    candidates.append(mark_trajectory(candidate, "caption"))
                else:
                    audio_path = Path(output).parent / "audio" / f"{candidate_id.replace('::', '_')}.wav"
                    codes = [[(frame + codebook + index) % 100 for codebook in range(16)] for frame in range(4)]
                    candidate = {
                        "candidate_id": candidate_id, "audio_path": str(audio_path), "request": row["request"],
                        "generation_seed": seeds[index], "codec_codes": codes, "codec_frames": 4, "codebooks": 16,
                        "old_main_logprobs": [-0.2 - 0.02 * index] * 4,
                        "ref_main_logprobs": [-0.22] * 4,
                        "old_sub_logprobs": [[-0.3 - 0.01 * index] * 15 for _ in range(4)],
                        "ref_sub_logprobs": [[-0.32] * 15 for _ in range(4)],
                        "behavior_logprob_mode": "processed_generation_scores", "trajectory_version": TRAJECTORY_VERSION,
                        "main_behavior_replay_max_abs_error": 0.0, "subtalker_behavior_replay_max_abs_error": 0.0,
                        "main_policy_reference_max_abs_error": 0.0, "subtalker_policy_reference_max_abs_error": 0.0,
                    }
                    candidates.append(mark_trajectory(candidate, "tts"))
            result.append({**row, "candidates": candidates})
    elif action == "score-reconstruction":
        for row in rows:
            value = _mock_reconstruction(str(row["candidate_id"]), -0.55, 0.08)
            result.append({
                "candidate_id": row["candidate_id"], "tts_target_logprob": value,
                "tts_main_logprob": value + 0.05, "tts_sub_logprob": -0.2,
                "codec_frames": 4, "codebooks": 16,
            })
    elif action == "score":
        for row in rows:
            index = _candidate_index(row["candidate_id"])
            if "target_caption" in row:
                value = _mock_reconstruction(str(row["candidate_id"]), -0.45, 0.07)
                result.append({
                    "candidate_id": row["candidate_id"], "caption_target_logprob": value,
                    "caption_target_logprob_macro": value,
                    "caption_value_field_logprobs": {"mock": value},
                    "caption_target_value_token_count": 8,
                })
            else:
                result.append({
                    "candidate_id": row["candidate_id"], "audio_health": 0.78 + 0.04 * index,
                    "asr_score": 0.76 + 0.05 * index, "asr_text": row.get("transcript", ""),
                })
    elif action in {"grpo-update", "sft-update"}:
        result = _checkpoint(action, rows, checkpoint_in, checkpoint_out)
    elif action == "gradient-audit":
        result = [{"status": "ok", "loss": 0.5, "gradient_family_nonzero": {"audio": 1, "text": 1}}]
    elif action == "preflight":
        result = [{"status": "ok", "checkpoint_reload": bool(checkpoint_in), "num_codebooks": 16}]
    else:
        raise ValueError(f"Unsupported mock action: {action}")
    write_output(output, result, {"mock": True, "gpu_validation": "not_run"})


def main() -> None:
    actions = [
        "prepare-codecs", "rollout", "score", "score-reconstruction",
        "grpo-update", "sft-update", "gradient-audit", "preflight",
    ]
    parser = worker_parser("Deterministic DualISL-Train mock", actions)
    args: Namespace = parser.parse_args()
    config, rows = load_job(args, initialize_torch=False)
    execute_mock(
        action=args.action, config=config, rows=rows, output=args.output,
        checkpoint_in=args.checkpoint_in, target_checkpoint=args.target_checkpoint,
        checkpoint_out=args.checkpoint_out, phase=args.training_phase,
    )


if __name__ == "__main__":
    main()
