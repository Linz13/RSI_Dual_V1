from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from .checkpoints import checkpoint_record, commit_round
from .config import public_config
from .constants import SOURCE_DOMAIN_TARGET
from .counterfactuals import counterfactual_captions
from .data import load_records, validate_manifest_roles
from .dual_space import (
    dual_subspace_report, project_synth_caption, reparse_synth_candidate, synth_caption_json,
)
from .io import atomic_json, dump_yaml, read_jsonl, stable_hash, write_jsonl
from .render import caption_prompt, render_qwen_request, synth_caption_prompt
from .rewards import (
    candidate_components, fit_round0_calibration, fit_sft_threshold, reward_summary,
    score_calibration_examples, score_groups, validate_calibration,
)
from .stages import StageManager, StageResult, run_worker


class DualISLOrchestrator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.project_root = Path(__file__).resolve().parent.parent
        self.run_dir = Path(config["run"]["output_dir"]).resolve()
        self.manager = StageManager(self.run_dir, config)
        self.resolved_config_path = self.run_dir / "resolved_config.yaml"
        dump_yaml(self.resolved_config_path, public_config(config))

    def _worker(self, role: str) -> tuple[str, str]:
        section = self.config[role]
        return str(section["python"]), str(section["worker_module"])

    @staticmethod
    def _validate_sft_contract(role: str, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("target_origin") != SOURCE_DOMAIN_TARGET:
                raise ValueError(f"{role} SFT row {row.get('id')} is not source-domain supervised")
            if role == "tts" and row.get("audio_path") != row.get("source_audio_path"):
                raise ValueError(f"TTS SFT row {row.get('id')} does not target the original source audio")

    def _stage(
        self, *, name: str, role: str, action: str, rows: list[dict[str, Any]], directory: Path,
        checkpoint_in: str | None = None, target_checkpoint: str | None = None,
        checkpoint_out: Path | None = None, phase: str | None = None,
        distributed: bool | None = None,
    ) -> StageResult:
        if action == "sft-update":
            self._validate_sft_contract(role, rows)
        input_path = directory / f"{name}.input.jsonl"
        output_path = directory / f"{name}.output.jsonl"
        metadata_path = directory / f"{name}.stage.json"
        write_jsonl(input_path, rows)
        invocation = stable_hash({
            "role": role,
            "action": action,
            "checkpoint_in": checkpoint_record(checkpoint_in),
            "target_checkpoint": checkpoint_record(target_checkpoint),
            "checkpoint_out": str(checkpoint_out.resolve()) if checkpoint_out else None,
            "phase": phase,
            "distributed": self.config.get("distributed", {}) if distributed is not False else {},
        })
        reusable = self.manager.reusable(name, input_path, invocation)
        if reusable:
            old_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            atomic_json(metadata_path, {
                **old_metadata,
                "name": name, "role": role, "action": action, "status": "reused",
                "invocation_hash": invocation,
            })
            return reusable
        python, module = self._worker(role)
        started = time.perf_counter()
        atomic_json(metadata_path, {
            "name": name, "role": role, "action": action, "status": "running",
            "invocation_hash": invocation,
        })
        try:
            if bool(self.config.get("run", {}).get("inline_mock", False)) and module.endswith(".workers.mock"):
                from .workers.mock import execute_mock

                execute_mock(
                    action=action, config=self.config, rows=rows, output=str(output_path),
                    checkpoint_in=checkpoint_in or "", target_checkpoint=target_checkpoint or "",
                    checkpoint_out=str(checkpoint_out) if checkpoint_out else "", phase=phase or "",
                )
            else:
                run_worker(
                    python=python, module=module, action=action,
                    config_path=str(self.resolved_config_path), input_path=input_path,
                    output_path=output_path, checkpoint_in=checkpoint_in,
                    target_checkpoint=target_checkpoint, checkpoint_out=checkpoint_out,
                    log_path=self.run_dir / "logs" / f"{name}.log",
                    project_root=self.project_root, training_phase=phase,
                    distributed=self.config.get("distributed", {}) if distributed is not False else {},
                )
            if not output_path.is_file():
                raise RuntimeError(f"Worker did not create {output_path}")
            if checkpoint_out is not None and not checkpoint_out.is_dir():
                raise RuntimeError(f"Worker did not create checkpoint {checkpoint_out}")
            self.manager.complete(name, input_path, output_path, checkpoint_out, invocation)
            metrics_path = Path(str(output_path) + ".metrics.json")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
            atomic_json(metadata_path, {
                "name": name, "role": role, "action": action, "status": "complete",
                "invocation_hash": invocation, "elapsed_seconds": time.perf_counter() - started,
                "worker_metrics": metrics,
            })
            return StageResult(name, output_path, checkpoint_out)
        except Exception as exc:
            self.manager.fail(name, input_path, repr(exc))
            atomic_json(metadata_path, {
                "name": name, "role": role, "action": action, "status": "failed",
                "invocation_hash": invocation, "error": repr(exc),
                "elapsed_seconds": time.perf_counter() - started,
            })
            raise

    def _local_stage(
        self, *, name: str, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]],
        directory: Path, invocation_material: dict[str, Any], action: str,
    ) -> StageResult:
        input_path = directory / f"{name}.input.jsonl"
        output_path = directory / f"{name}.output.jsonl"
        write_jsonl(input_path, inputs)
        invocation = stable_hash(invocation_material)
        reusable = self.manager.reusable(name, input_path, invocation)
        if reusable:
            return reusable
        write_jsonl(output_path, outputs)
        self.manager.complete(name, input_path, output_path, invocation_hash=invocation)
        atomic_json(directory / f"{name}.stage.json", {
            "name": name, "role": "orchestrator", "action": action,
            "status": "complete", "invocation_hash": invocation,
        })
        return StageResult(name, output_path)

    @staticmethod
    def _index(path: Path, key: str = "candidate_id") -> dict[str, dict[str, Any]]:
        return {str(row[key]): row for row in read_jsonl(path)}

    def _seed(self, round_index: int, branch: str, sample_id: str, candidate_index: int) -> int:
        digest = stable_hash({
            "base_seed": int(self.config["run"].get("seed", 42)),
            "round": round_index, "branch": branch,
            "sample_id": str(sample_id), "candidate_index": candidate_index,
        })
        return int(digest[:15], 16) % (2**31 - 1)

    def _counterfactuals(self, caption: dict[str, Any], key: str) -> list[dict[str, Any]]:
        return counterfactual_captions(
            caption, key=key,
            max_count=int(self.config["reward"].get("counterfactuals_per_candidate", 3)),
        )

    def _loop_reward_config(self, loop: str) -> dict[str, Any]:
        reward = self.config["reward"]
        return {
            **reward[loop],
            "anchor_penalty_weight": reward["anchor_penalty_weight"],
            "anchor_tolerance_z": reward["anchor_tolerance_z"],
        }

    def _quality_config(self, loop: str) -> dict[str, Any]:
        return dict(self.config["reward"].get("validity", {}).get(loop, {}))

    def data_contract(self) -> dict[str, Any]:
        manifests = {
            "paired": self.config["data"]["paired_path"],
            "audio_only": self.config["data"]["audio_only_path"],
            "caption_only": self.config["data"]["caption_only_path"],
        }
        report = validate_manifest_roles(
            manifests, require_disjoint=bool(self.config["data"].get("strict_disjoint", True)),
            hash_audio=bool(self.config["data"].get("hash_audio", False)),
        )
        for role in manifests:
            if int(report.get("counts", {}).get(role, 0)) < 1:
                report["errors"].append({"role": role, "error": "manifest must contain at least one row"})
        report["ok"] = not report["errors"]
        atomic_json(self.run_dir / "data_contract.json", report)
        if not report["ok"]:
            raise ValueError(f"Data contract failed: {report['errors'][:3]}")
        return report

    def _prepare_codecs(
        self, name: str, rows: list[dict[str, Any]], directory: Path,
    ) -> dict[str, dict[str, Any]]:
        stage = self._stage(
            name=name, role="tts", action="prepare-codecs", directory=directory,
            rows=[{"id": row["id"], "audio_path": row["audio_path"]} for row in rows],
            distributed=False,
        )
        return {str(row["id"]): row for row in read_jsonl(stage.output_path)}

    def _collect_audio_only(
        self, round_index: int, rows: list[dict[str, Any]], codec_index: dict[str, dict[str, Any]],
        caption_checkpoint: str | None, tts_checkpoint: str | None,
        anchor_tts_checkpoint: str | None, directory: Path,
    ) -> list[dict[str, Any]]:
        group_size = int(self.config["training"]["group_size"])
        rollout_rows = [{
            "id": row["id"], "source_id": row["id"], "audio_path": row["audio_path"],
            "prompt": synth_caption_prompt(), "caption_schema": "synth_v1", "group_size": group_size,
            "candidate_seeds": [self._seed(round_index, "audio_only", row["id"], index) for index in range(group_size)],
        } for row in rows]
        rollout = self._stage(
            name=f"round_{round_index:03d}_audio_caption_rollout", role="captioner", action="rollout",
            rows=rollout_rows, directory=directory, checkpoint_in=caption_checkpoint,
            target_checkpoint=caption_checkpoint,
        )
        groups = list(read_jsonl(rollout.output_path))
        # Admission is derived from the original sampled text before allocating
        # reverse-model scoring jobs. Keep the original tokens/logprobs intact.
        groups = [
            {**group, "candidates": [reparse_synth_candidate(item) for item in group["candidates"]]}
            for group in groups
        ]
        reconstruction_rows, critic_rows = [], []
        plans: dict[str, list[dict[str, Any]]] = {}
        for group in groups:
            for candidate in group.get("candidates", []):
                caption = candidate.get("caption")
                if caption is None or not bool(candidate.get("semantic_input_valid")):
                    continue
                try:
                    request = render_qwen_request(caption)
                except ValueError:
                    continue
                candidate_id = str(candidate["candidate_id"])
                reconstruction_rows.append({
                    "candidate_id": candidate_id, "request": request,
                    "codec_path": codec_index[str(group["id"])]["codec_path"],
                })
                plans[candidate_id] = self._counterfactuals(
                    caption, f"round:{round_index}:audio:{candidate_id}"
                )
                for index, counterfactual in enumerate(plans[candidate_id]):
                    counterfactual["score_id"] = f"{candidate_id}::cf::{index:02d}"
                    reconstruction_rows.append({
                        "candidate_id": counterfactual["score_id"],
                        "request": render_qwen_request(counterfactual["caption"]),
                        "codec_path": codec_index[str(group["id"])]["codec_path"],
                    })
                critic_rows.append({
                    "candidate_id": candidate_id, "audio_path": group["audio_path"],
                    "transcript": caption["semantic_content"]["transcript"],
                    "language": caption["semantic_content"]["language"],
                })
        reconstruction = self._stage(
            name=f"round_{round_index:03d}_audio_tts_reconstruction", role="tts",
            action="score-reconstruction", rows=reconstruction_rows, directory=directory,
            checkpoint_in=tts_checkpoint, target_checkpoint=tts_checkpoint,
        )
        anchor_reconstruction = self._stage(
            name=f"round_{round_index:03d}_audio_tts_anchor_reconstruction", role="tts",
            action="score-reconstruction", rows=reconstruction_rows, directory=directory,
            checkpoint_in=anchor_tts_checkpoint, target_checkpoint=anchor_tts_checkpoint,
        )
        critics = self._stage(
            name=f"round_{round_index:03d}_audio_caption_quality", role="critics", action="score",
            rows=critic_rows, directory=directory,
        )
        reconstruction_index = self._index(reconstruction.output_path)
        anchor_reconstruction_index = self._index(anchor_reconstruction.output_path)
        critic_index = self._index(critics.output_path)
        merged = []
        for group in groups:
            candidates = []
            for candidate in group.get("candidates", []):
                candidate_id = str(candidate["candidate_id"])
                current = reconstruction_index.get(candidate_id, {})
                anchor = anchor_reconstruction_index.get(candidate_id, {})
                counterfactual_current, counterfactual_anchor = [], []
                for item in plans.get(candidate_id, []):
                    current_item = reconstruction_index.get(str(item["score_id"]), {})
                    anchor_item = anchor_reconstruction_index.get(str(item["score_id"]), {})
                    metadata = {
                        key: item[key] for key in (
                            "score_id", "field", "original_value", "alternative_value"
                        )
                    }
                    if "tts_target_logprob" in current_item:
                        counterfactual_current.append({
                            **metadata, "reconstruction": current_item["tts_target_logprob"],
                        })
                    if "tts_target_logprob" in anchor_item:
                        counterfactual_anchor.append({
                            **metadata, "reconstruction": anchor_item["tts_target_logprob"],
                        })
                candidates.append({
                    **candidate, **current, **critic_index.get(candidate_id, {}),
                    "anchor_tts_target_logprob": anchor.get("tts_target_logprob"),
                    "counterfactual_reconstruction": counterfactual_current,
                    "anchor_counterfactual_reconstruction": counterfactual_anchor,
                })
            merged.append({**group, "candidates": candidates})
        stage = self._local_stage(
            name=f"round_{round_index:03d}_audio_collection", inputs=groups, outputs=merged,
            directory=directory, action="merge-audio-only-components",
            invocation_material={"caption_checkpoint": checkpoint_record(caption_checkpoint),
                                 "tts_checkpoint": checkpoint_record(tts_checkpoint),
                                 "anchor_tts_checkpoint": checkpoint_record(anchor_tts_checkpoint),
                                 "counterfactuals_per_candidate": self.config["reward"]["counterfactuals_per_candidate"]},
        )
        return list(read_jsonl(stage.output_path))

    def _collect_sft_calibration_examples(
        self, round_index: int, paired: list[dict[str, Any]],
        paired_codecs: dict[str, dict[str, Any]], caption_checkpoint: str | None,
        tts_checkpoint: str | None, anchor_caption_checkpoint: str | None,
        anchor_tts_checkpoint: str | None, directory: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        plans: dict[str, dict[str, Any]] = {}
        tts_rows, caption_rows = [], []
        for row in paired:
            source_id = str(row["id"])
            source_caption = project_synth_caption(row["caption"])
            counterfactuals = self._counterfactuals(
                source_caption, f"round:{round_index}:paired:{source_id}"
            )
            if not counterfactuals:
                continue
            positive_id = f"sftcal::{source_id}::positive"
            plans[source_id] = {
                "source": row, "caption": source_caption,
                "positive_id": positive_id, "counterfactuals": counterfactuals,
            }
            tts_rows.append({
                "candidate_id": positive_id, "request": render_qwen_request(source_caption),
                "codec_path": paired_codecs[source_id]["codec_path"],
            })
            caption_rows.append({
                "candidate_id": positive_id, "audio_path": row["audio_path"],
                "target_caption": source_caption, "prompt": synth_caption_prompt(),
                "target_schema": "synth_v1", "score_mode": "synth_values_macro",
            })
            for index, counterfactual in enumerate(counterfactuals):
                score_id = f"sftcal::{source_id}::negative::{index:02d}"
                counterfactual["score_id"] = score_id
                tts_rows.append({
                    "candidate_id": score_id,
                    "request": render_qwen_request(counterfactual["caption"]),
                    "codec_path": paired_codecs[source_id]["codec_path"],
                })
                caption_rows.append({
                    "candidate_id": score_id, "audio_path": row["audio_path"],
                    "target_caption": counterfactual["caption"],
                    "prompt": synth_caption_prompt(), "target_schema": "synth_v1",
                    "score_mode": "synth_values_macro",
                })

        tts_current = self._stage(
            name=f"round_{round_index:03d}_sftcal_tts_current", role="tts",
            action="score-reconstruction", rows=tts_rows, directory=directory,
            checkpoint_in=tts_checkpoint, target_checkpoint=tts_checkpoint,
        )
        tts_anchor = self._stage(
            name=f"round_{round_index:03d}_sftcal_tts_anchor", role="tts",
            action="score-reconstruction", rows=tts_rows, directory=directory,
            checkpoint_in=anchor_tts_checkpoint, target_checkpoint=anchor_tts_checkpoint,
        )
        caption_current = self._stage(
            name=f"round_{round_index:03d}_sftcal_caption_current", role="captioner",
            action="score", rows=caption_rows, directory=directory,
            checkpoint_in=caption_checkpoint, target_checkpoint=caption_checkpoint,
        )
        caption_anchor = self._stage(
            name=f"round_{round_index:03d}_sftcal_caption_anchor", role="captioner",
            action="score", rows=caption_rows, directory=directory,
            checkpoint_in=anchor_caption_checkpoint, target_checkpoint=anchor_caption_checkpoint,
        )
        tts_current_index = self._index(tts_current.output_path)
        tts_anchor_index = self._index(tts_anchor.output_path)
        caption_current_index = self._index(caption_current.output_path)
        caption_anchor_index = self._index(caption_anchor.output_path)

        audio_examples, caption_examples = [], []
        for source_id, plan in plans.items():
            positive_id = plan["positive_id"]
            source = plan["source"]
            source_caption = plan["caption"]
            counterfactuals = plan["counterfactuals"]
            tts_positive = tts_current_index[positive_id]["tts_target_logprob"]
            tts_anchor_positive = tts_anchor_index[positive_id]["tts_target_logprob"]
            caption_positive = caption_current_index[positive_id]["caption_target_logprob_macro"]
            caption_anchor_positive = caption_anchor_index[positive_id]["caption_target_logprob_macro"]
            tts_negative_scores = [{
                "reconstruction": tts_current_index[item["score_id"]]["tts_target_logprob"],
                "field": item["field"], "score_id": item["score_id"],
            } for item in counterfactuals]
            tts_anchor_negative_scores = [{
                "reconstruction": tts_anchor_index[item["score_id"]]["tts_target_logprob"],
                "field": item["field"], "score_id": item["score_id"],
            } for item in counterfactuals]
            caption_negative_scores = [{
                "reconstruction": caption_current_index[item["score_id"]]["caption_target_logprob_macro"],
                "field": item["field"], "score_id": item["score_id"],
            } for item in counterfactuals]
            caption_anchor_negative_scores = [{
                "reconstruction": caption_anchor_index[item["score_id"]]["caption_target_logprob_macro"],
                "field": item["field"], "score_id": item["score_id"],
            } for item in counterfactuals]
            shared_audio = {
                "candidate_id": positive_id, "caption": source_caption,
                "raw_text": synth_caption_json(source_caption),
                "raw_schema_valid": True, "normalized_schema_valid": True,
                "trajectory_valid": True, "asr_score": 1.0,
                "tts_target_logprob": tts_positive,
                "anchor_tts_target_logprob": tts_anchor_positive,
                "counterfactual_reconstruction": tts_negative_scores,
                "anchor_counterfactual_reconstruction": tts_anchor_negative_scores,
            }
            shared_caption = {
                "candidate_id": positive_id, "audio_path": source["audio_path"],
                "trajectory_valid": True, "audio_health": 1.0, "asr_score": 1.0,
                "caption_target_logprob_macro": caption_positive,
                "anchor_caption_target_logprob_macro": caption_anchor_positive,
                "counterfactual_reconstruction": caption_negative_scores,
                "anchor_counterfactual_reconstruction": caption_anchor_negative_scores,
            }
            audio_examples.append({
                "id": f"{source_id}::positive", "label": 1, "candidate": shared_audio,
            })
            caption_examples.append({
                "id": f"{source_id}::positive", "label": 1, "candidate": shared_caption,
            })

            negative = counterfactuals[0]
            negative_id = str(negative["score_id"])
            audio_examples.append({
                "id": f"{source_id}::negative", "label": 0,
                "candidate": {
                    **shared_audio, "candidate_id": negative_id,
                    "caption": negative["caption"],
                    "raw_text": synth_caption_json(negative["caption"]),
                    "tts_target_logprob": tts_current_index[negative_id]["tts_target_logprob"],
                    "anchor_tts_target_logprob": tts_anchor_index[negative_id]["tts_target_logprob"],
                    "counterfactual_reconstruction": [{"reconstruction": tts_positive}],
                    "anchor_counterfactual_reconstruction": [{"reconstruction": tts_anchor_positive}],
                },
            })
            caption_examples.append({
                "id": f"{source_id}::negative", "label": 0,
                "candidate": {
                    **shared_caption, "candidate_id": negative_id,
                    "caption_target_logprob_macro": caption_current_index[negative_id]["caption_target_logprob_macro"],
                    "anchor_caption_target_logprob_macro": caption_anchor_index[negative_id]["caption_target_logprob_macro"],
                    "counterfactual_reconstruction": [{"reconstruction": caption_positive}],
                    "anchor_counterfactual_reconstruction": [{"reconstruction": caption_anchor_positive}],
                },
            })
        return audio_examples, caption_examples

    def _collect_caption_only(
        self, round_index: int, rows: list[dict[str, Any]],
        caption_checkpoint: str | None, tts_checkpoint: str | None,
        anchor_caption_checkpoint: str | None, directory: Path,
    ) -> list[dict[str, Any]]:
        group_size = int(self.config["training"]["group_size"])
        rollout_rows = []
        for row in rows:
            source_caption = project_synth_caption(row["caption"])
            rollout_rows.append({
                "id": row["id"], "source_id": row["id"], "source_caption": source_caption,
                "request": render_qwen_request(source_caption), "group_size": group_size,
                "candidate_seeds": [self._seed(round_index, "caption_only", row["id"], index) for index in range(group_size)],
            })
        if self.config["tts"].get("generation", {}).get("rollout_schedule", "round_robin") == "previous_round_lpt":
            from .rollout_balance import attach_previous_costs
            rollout_rows = attach_previous_costs(rollout_rows, self.run_dir, round_index)
        rollout = self._stage(
            name=f"round_{round_index:03d}_caption_tts_rollout", role="tts", action="rollout",
            rows=rollout_rows, directory=directory, checkpoint_in=tts_checkpoint,
            target_checkpoint=tts_checkpoint,
        )
        groups = list(read_jsonl(rollout.output_path))
        critic_rows, reconstruction_rows = [], []
        plans: dict[str, list[dict[str, Any]]] = {}
        for group in groups:
            source_caption = group["source_caption"]
            for candidate in group.get("candidates", []):
                candidate_id = str(candidate["candidate_id"])
                critic_rows.append({
                    "candidate_id": candidate_id, "audio_path": candidate["audio_path"],
                    "transcript": source_caption["semantic_content"]["transcript"],
                    "language": source_caption["semantic_content"]["language"],
                })
                reconstruction_rows.append({
                    "candidate_id": candidate_id, "audio_path": candidate["audio_path"],
                    "target_caption": source_caption, "prompt": synth_caption_prompt(),
                    "target_schema": "synth_v1", "score_mode": "synth_values_macro",
                })
                plans[candidate_id] = self._counterfactuals(
                    source_caption, f"round:{round_index}:caption:{candidate_id}"
                )
                for index, counterfactual in enumerate(plans[candidate_id]):
                    counterfactual["score_id"] = f"{candidate_id}::cf::{index:02d}"
                    reconstruction_rows.append({
                        "candidate_id": counterfactual["score_id"],
                        "audio_path": candidate["audio_path"],
                        "target_caption": counterfactual["caption"],
                        "prompt": synth_caption_prompt(), "target_schema": "synth_v1",
                        "score_mode": "synth_values_macro",
                    })
        critics = self._stage(
            name=f"round_{round_index:03d}_caption_audio_quality", role="critics", action="score",
            rows=critic_rows, directory=directory,
        )
        reconstruction = self._stage(
            name=f"round_{round_index:03d}_caption_caption_reconstruction", role="captioner",
            action="score", rows=reconstruction_rows, directory=directory,
            checkpoint_in=caption_checkpoint, target_checkpoint=caption_checkpoint,
        )
        anchor_reconstruction = self._stage(
            name=f"round_{round_index:03d}_caption_anchor_reconstruction", role="captioner",
            action="score", rows=reconstruction_rows, directory=directory,
            checkpoint_in=anchor_caption_checkpoint, target_checkpoint=anchor_caption_checkpoint,
        )
        critic_index = self._index(critics.output_path)
        reconstruction_index = self._index(reconstruction.output_path)
        anchor_reconstruction_index = self._index(anchor_reconstruction.output_path)
        merged = []
        for group in groups:
            candidates = []
            for candidate in group.get("candidates", []):
                candidate_id = str(candidate["candidate_id"])
                current = reconstruction_index.get(candidate_id, {})
                anchor = anchor_reconstruction_index.get(candidate_id, {})
                counterfactual_current, counterfactual_anchor = [], []
                for item in plans.get(candidate_id, []):
                    current_item = reconstruction_index.get(str(item["score_id"]), {})
                    anchor_item = anchor_reconstruction_index.get(str(item["score_id"]), {})
                    metadata = {
                        key: item[key] for key in (
                            "score_id", "field", "original_value", "alternative_value"
                        )
                    }
                    current_score = current_item.get(
                        "caption_target_logprob_macro", current_item.get("caption_target_logprob")
                    )
                    anchor_score = anchor_item.get(
                        "caption_target_logprob_macro", anchor_item.get("caption_target_logprob")
                    )
                    if current_score is not None:
                        counterfactual_current.append({
                            **metadata, "reconstruction": current_score,
                        })
                    if anchor_score is not None:
                        counterfactual_anchor.append({
                            **metadata, "reconstruction": anchor_score,
                        })
                candidates.append({
                    **candidate, **critic_index.get(candidate_id, {}), **current,
                    "anchor_caption_target_logprob": anchor.get("caption_target_logprob"),
                    "anchor_caption_target_logprob_macro": anchor.get(
                        "caption_target_logprob_macro", anchor.get("caption_target_logprob")
                    ),
                    "counterfactual_reconstruction": counterfactual_current,
                    "anchor_counterfactual_reconstruction": counterfactual_anchor,
                })
            merged.append({**group, "candidates": candidates})
        stage = self._local_stage(
            name=f"round_{round_index:03d}_caption_collection", inputs=groups, outputs=merged,
            directory=directory, action="merge-caption-only-components",
            invocation_material={"caption_checkpoint": checkpoint_record(caption_checkpoint),
                                 "tts_checkpoint": checkpoint_record(tts_checkpoint),
                                 "anchor_caption_checkpoint": checkpoint_record(anchor_caption_checkpoint),
                                 "counterfactuals_per_candidate": self.config["reward"]["counterfactuals_per_candidate"]},
        )
        return list(read_jsonl(stage.output_path))

    def _calibration(
        self, round_index: int, audio_groups: list[dict[str, Any]], caption_groups: list[dict[str, Any]],
        audio_examples: list[dict[str, Any]], caption_examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        path = self.run_dir / "reward_calibration.json"
        initial_path_value = self.config["reward"].get("calibration", {}).get("initial_path")
        initial_path = Path(str(initial_path_value)).resolve() if initial_path_value else None
        if path.is_file():
            calibration = json.loads(path.read_text(encoding="utf-8"))
            validate_calibration(calibration)
            if initial_path is not None:
                source = json.loads(initial_path.read_text(encoding="utf-8"))
                validate_calibration(source)
                if calibration != source:
                    raise RuntimeError(
                        "Run reward_calibration.json differs from the frozen continuation source"
                    )
            return calibration
        if initial_path is not None:
            calibration = json.loads(initial_path.read_text(encoding="utf-8"))
            validate_calibration(calibration)
            if calibration.get("fitted_round") != 0 or calibration.get("frozen_across_rounds") is not True:
                raise ValueError("Continuation requires the original frozen round-0 reward calibration")
            atomic_json(path, calibration)
            return calibration
        if round_index != 0:
            raise RuntimeError("Missing frozen round-0 reward calibration")
        calibration = fit_round0_calibration(
            audio_groups, caption_groups,
            quality={
                "audio_only": self._quality_config("audio_only"),
                "caption_only": self._quality_config("caption_only"),
            },
            fallback_candidates={
                "audio_only": [item["candidate"] for item in audio_examples],
                "caption_only": [item["candidate"] for item in caption_examples],
            },
        )
        atomic_json(path, calibration)
        return calibration

    def _continuation_contract(
        self, caption_checkpoint: str | None, tts_checkpoint: str | None,
        anchor_caption_checkpoint: str | None, anchor_tts_checkpoint: str | None,
        round_offset: int,
    ) -> None:
        if round_offset == 0:
            return
        calibration_path = self.config["reward"]["calibration"]["initial_path"]
        record = {
            "kind": "committed_dual_checkpoint_continuation",
            "round_offset": round_offset,
            "round_count": int(self.config["training"]["rounds"]),
            "input_captioner": checkpoint_record(caption_checkpoint),
            "input_tts": checkpoint_record(tts_checkpoint),
            "frozen_reward_anchor": {
                "captioner_adapter": checkpoint_record(anchor_caption_checkpoint),
                "tts_adapter": checkpoint_record(anchor_tts_checkpoint),
            },
            "source_reward_calibration": checkpoint_record(calibration_path),
        }
        path = self.run_dir / "continuation.json"
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != record:
                raise RuntimeError("Existing continuation.json does not match the configured source lineage")
            return
        atomic_json(path, record)

    def _reward_anchor_contract(
        self, caption_checkpoint: str | None, tts_checkpoint: str | None,
    ) -> None:
        record = {
            "kind": "frozen_recursive_start_anchor",
            "captioner_adapter": checkpoint_record(caption_checkpoint),
            "tts_adapter": checkpoint_record(tts_checkpoint),
            "captioner_model_path": str(Path(self.config["captioner"]["model_path"]).resolve()),
            "tts_model_path": str(Path(self.config["tts"]["model_path"]).resolve()),
            "updated_during_training": False,
        }
        path = self.run_dir / "reward_anchor.json"
        if path.is_file():
            if json.loads(path.read_text(encoding="utf-8")) != record:
                raise RuntimeError("Existing reward_anchor.json does not match the configured start models")
            return
        atomic_json(path, record)

    def _fit_sft_thresholds(
        self, round_index: int, audio_examples: list[dict[str, Any]],
        caption_examples: list[dict[str, Any]], calibration: dict[str, Any], directory: Path,
    ) -> dict[str, Any]:
        gate = self.config["reward"]["sft_gate"]
        thresholds: dict[str, Any] = {
            "version": 1,
            "diagnostic_only": True,
            "cycle_sft_selection": self.config["reward"]["cycle_sft_selection"],
            "round": round_index,
            "method": "paired_vs_counterfactual_empirical_precision",
            "loops": {},
        }
        for loop, examples in (
            ("audio_only", audio_examples), ("caption_only", caption_examples)
        ):
            values = score_calibration_examples(
                examples, loop=loop, calibration=calibration,
                reward_config=self._loop_reward_config(loop),
                quality=self._quality_config(loop),
            )
            stage = self._local_stage(
                name=f"round_{round_index:03d}_{loop}_sft_confidence_calibration",
                inputs=examples, outputs=values, directory=directory,
                action="paired-counterfactual-sft-calibration",
                invocation_material={
                    "loop": loop, "calibration": calibration,
                    "reward": self._loop_reward_config(loop), "gate": gate,
                },
            )
            audited = list(read_jsonl(stage.output_path))
            positive_count = sum(int(item["label"]) == 1 and math.isfinite(item["score"]) for item in audited)
            negative_count = sum(int(item["label"]) == 0 and math.isfinite(item["score"]) for item in audited)
            if not positive_count or not negative_count:
                # Unavailable diagnostic calibration must not become an implicit SFT gate.
                thresholds["loops"][loop] = {
                    "threshold": None, "status": "insufficient_calibration_examples",
                    "target_precision": float(gate["target_precision"]),
                    "min_recall": float(gate["min_recall"]),
                    "positive_count": positive_count, "negative_count": negative_count,
                }
            else:
                thresholds["loops"][loop] = fit_sft_threshold(
                    audited, target_precision=float(gate["target_precision"]),
                    min_recall=float(gate["min_recall"]),
                )
        atomic_json(directory / f"round_{round_index:03d}_sft_thresholds.json", thresholds)
        return thresholds

    def _base_diagnostics(
        self, audio_groups: list[dict[str, Any]], caption_groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate both frozen round-0 loops before the first optimizer step."""
        branches: dict[str, Any] = {}
        for loop, groups in (("audio_only", audio_groups), ("caption_only", caption_groups)):
            evaluated_groups = []
            for group in groups:
                evaluated_groups.append([
                    (candidate, *candidate_components(loop, candidate, self._quality_config(loop)))
                    for candidate in group.get("candidates", [])
                ])
            evaluated = [item for group in evaluated_groups for item in group]
            candidates = [item[0] for item in evaluated]
            valid = sum(item[1] for item in evaluated)
            trajectory_valid = sum(bool(candidate.get("trajectory_valid")) for candidate in candidates)
            generated = sum(
                bool(candidate.get("caption")) if loop == "audio_only"
                else bool(candidate.get("audio_path")) and bool(candidate.get("codec_codes"))
                for candidate in candidates
            )
            curriculum_groups = 0
            if loop == "audio_only":
                curriculum_groups = sum(
                    sum(
                        bool(candidate.get("trajectory_valid"))
                        and isinstance(components.get("schema_progress"), (int, float))
                        and math.isfinite(float(components["schema_progress"]))
                        for candidate, _semantic_valid, components in group
                    ) >= 2
                    for group in evaluated_groups
                )
            semantic_ready = valid > 0
            curriculum_ready = loop == "audio_only" and curriculum_groups > 0
            branches[loop] = {
                "groups": len(groups),
                "candidates": len(candidates),
                "generated_candidates": generated,
                "trajectory_valid_candidates": trajectory_valid,
                "reconstruction_valid_candidates": valid,
                "schema_curriculum_usable_groups": curriculum_groups,
                "diagnostic_mode": (
                    "dual_semantic" if semantic_ready
                    else "schema_curriculum" if curriculum_ready
                    else None
                ),
                "ok": (
                    bool(groups) and trajectory_valid > 0
                    and (semantic_ready or curriculum_ready)
                    and (generated > 0 if loop == "caption_only" else True)
                ),
            }
        dual_space = dual_subspace_report()
        report = {
            "stage": "round0_frozen_base_diagnostics",
            "before_any_optimizer_step": True,
            "quality_thresholds_applied": True,
            "dual_space": dual_space,
            "branches": branches,
        }
        report["ok"] = bool(dual_space.get("ok")) and all(item["ok"] for item in branches.values())
        atomic_json(self.run_dir / "base_diagnostics.json", report)
        if not report["ok"]:
            raise RuntimeError("Frozen base-model diagnostics failed before round-0 training")
        return report

    def _score_round(
        self, round_index: int, audio_raw: list[dict[str, Any]], caption_raw: list[dict[str, Any]],
        calibration: dict[str, Any], thresholds: dict[str, Any], directory: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        audio_threshold = thresholds["loops"]["audio_only"].get("threshold")
        caption_threshold = thresholds["loops"]["caption_only"].get("threshold")
        audio_scored = score_groups(
            audio_raw, loop="audio_only", calibration=calibration,
            reward_config=self._loop_reward_config("audio_only"),
            sft_threshold=audio_threshold, quality=self._quality_config("audio_only"),
        )
        caption_scored = score_groups(
            caption_raw, loop="caption_only", calibration=calibration,
            reward_config=self._loop_reward_config("caption_only"),
            sft_threshold=caption_threshold, quality=self._quality_config("caption_only"),
        )
        audio_stage = self._local_stage(
            name=f"round_{round_index:03d}_audio_rewards", inputs=audio_raw, outputs=audio_scored,
            directory=directory, action="anchored-counterfactual-dual-reward",
            invocation_material={"loop": "audio_only", "calibration": calibration,
                                 "reward": self._loop_reward_config("audio_only"),
                                 "cycle_sft_selection": self.config["reward"]["cycle_sft_selection"],
                                 "sft_gate_diagnostic_only": True,
                                 "sft_threshold": audio_threshold},
        )
        caption_stage = self._local_stage(
            name=f"round_{round_index:03d}_caption_rewards", inputs=caption_raw, outputs=caption_scored,
            directory=directory, action="anchored-counterfactual-dual-reward",
            invocation_material={"loop": "caption_only", "calibration": calibration,
                                 "reward": self._loop_reward_config("caption_only"),
                                 "cycle_sft_selection": self.config["reward"]["cycle_sft_selection"],
                                 "sft_gate_diagnostic_only": True,
                                 "sft_threshold": caption_threshold},
        )
        return list(read_jsonl(audio_stage.output_path)), list(read_jsonl(caption_stage.output_path))

    @staticmethod
    def _require_trainable(loop: str, groups: list[dict[str, Any]]) -> dict[str, Any]:
        summary = reward_summary(groups)
        if summary["grpo_usable_groups"] < 1:
            raise RuntimeError(f"{loop} has no structurally usable GRPO group")
        return summary

    def _caption_anchors(self, paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for source in paired:
            rows.extend((
                {
                    "id": f"anchor::source::{source['id']}", "audio_path": source["audio_path"],
                    "caption": source["caption"], "prompt": caption_prompt(),
                    "target_schema": "source_no_environment",
                    "target_origin": SOURCE_DOMAIN_TARGET, "is_anchor": True,
                },
                {
                    "id": f"anchor::psyn::{source['id']}", "audio_path": source["audio_path"],
                    "caption": project_synth_caption(source["caption"]), "prompt": synth_caption_prompt(),
                    "target_schema": "synth_v1", "target_origin": SOURCE_DOMAIN_TARGET, "is_anchor": True,
                },
            ))
        return rows

    def _tts_anchors(
        self, paired: list[dict[str, Any]], codec_index: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [{
            "id": f"anchor::{row['id']}", "audio_path": row["audio_path"],
            "source_audio_path": row["audio_path"], "codec_path": codec_index[str(row["id"])]["codec_path"],
            "request": render_qwen_request(project_synth_caption(row["caption"])),
            "target_origin": SOURCE_DOMAIN_TARGET, "is_anchor": True,
        } for row in paired]

    def _update_captioner(
        self, round_index: int, audio_groups: list[dict[str, Any]], caption_groups: list[dict[str, Any]],
        paired: list[dict[str, Any]], start_checkpoint: str | None, directory: Path,
    ) -> str:
        after_grpo = directory / "checkpoints" / "caption_after_grpo"
        self._stage(
            name=f"round_{round_index:03d}_caption_grpo", role="captioner", action="grpo-update",
            rows=audio_groups, directory=directory / "training", checkpoint_in=start_checkpoint,
            target_checkpoint=start_checkpoint, checkpoint_out=after_grpo, phase="grpo",
        )
        pseudo_rows = []
        for group in caption_groups:
            for candidate in group.get("candidates", []):
                if candidate.get("sft_selected"):
                    pseudo_rows.append({
                        "id": candidate["candidate_id"], "audio_path": candidate["audio_path"],
                        "caption": group["source_caption"], "prompt": synth_caption_prompt(),
                        "target_schema": "synth_v1", "target_origin": SOURCE_DOMAIN_TARGET,
                        "is_anchor": False,
                    })
        sft_rows = [*pseudo_rows, *self._caption_anchors(paired)]
        final = directory / "checkpoints" / "caption_final"
        self._stage(
            name=f"round_{round_index:03d}_caption_sft", role="captioner", action="sft-update",
            rows=sft_rows, directory=directory / "training", checkpoint_in=str(after_grpo),
            checkpoint_out=final, phase="cycle_sft",
        )
        return str(final)

    def _update_tts(
        self, round_index: int, audio_groups: list[dict[str, Any]], caption_groups: list[dict[str, Any]],
        paired: list[dict[str, Any]], paired_codecs: dict[str, dict[str, Any]],
        audio_codecs: dict[str, dict[str, Any]], start_checkpoint: str | None, directory: Path,
    ) -> str:
        after_grpo = directory / "checkpoints" / "tts_after_grpo"
        self._stage(
            name=f"round_{round_index:03d}_tts_grpo", role="tts", action="grpo-update",
            rows=caption_groups, directory=directory / "training", checkpoint_in=start_checkpoint,
            target_checkpoint=start_checkpoint, checkpoint_out=after_grpo, phase="grpo",
        )
        pseudo_rows = []
        for group in audio_groups:
            for candidate in group.get("candidates", []):
                if candidate.get("sft_selected") and candidate.get("caption") is not None:
                    pseudo_rows.append({
                        "id": candidate["candidate_id"], "audio_path": group["audio_path"],
                        "source_audio_path": group["audio_path"],
                        "codec_path": audio_codecs[str(group["id"])]["codec_path"],
                        "request": render_qwen_request(candidate["caption"]),
                        "target_origin": SOURCE_DOMAIN_TARGET, "is_anchor": False,
                    })
        sft_rows = [*pseudo_rows, *self._tts_anchors(paired, paired_codecs)]
        final = directory / "checkpoints" / "tts_final"
        self._stage(
            name=f"round_{round_index:03d}_tts_sft", role="tts", action="sft-update",
            rows=sft_rows, directory=directory / "training", checkpoint_in=str(after_grpo),
            checkpoint_out=final, phase="cycle_sft",
        )
        return str(final)

    def _reload_check(self, round_index: int, caption_checkpoint: str, tts_checkpoint: str, directory: Path) -> None:
        self._stage(
            name=f"round_{round_index:03d}_caption_reload", role="captioner", action="preflight",
            rows=[], directory=directory / "verification", checkpoint_in=caption_checkpoint, distributed=False,
        )
        self._stage(
            name=f"round_{round_index:03d}_tts_reload", role="tts", action="preflight",
            rows=[], directory=directory / "verification", checkpoint_in=tts_checkpoint, distributed=False,
        )

    def train(self, *, resume_only: bool = False) -> dict[str, Any]:
        self.data_contract()
        current = self.manager.state.get("current", {})
        if resume_only and not self.manager.state.get("stages"):
            raise RuntimeError("No existing DualISL-Train stages to resume")
        configured_caption_checkpoint = self.config["captioner"].get("adapter_path") or None
        configured_tts_checkpoint = self.config["tts"].get("adapter_path") or None
        caption_checkpoint = current.get("caption_checkpoint") or configured_caption_checkpoint
        tts_checkpoint = current.get("tts_checkpoint") or configured_tts_checkpoint
        round_offset = int(self.config["training"].get("round_offset", 0))
        reward_anchor = self.config["reward"].get("anchor") or {}
        anchor_caption_checkpoint = (
            reward_anchor.get("captioner_adapter_path") or None
            if round_offset > 0 else configured_caption_checkpoint
        )
        anchor_tts_checkpoint = (
            reward_anchor.get("tts_adapter_path") or None
            if round_offset > 0 else configured_tts_checkpoint
        )
        round_limit = round_offset + int(self.config["training"]["rounds"])
        start_round = max(round_offset, int(current.get("round", round_offset - 1)) + 1)
        self._continuation_contract(
            configured_caption_checkpoint, configured_tts_checkpoint,
            anchor_caption_checkpoint, anchor_tts_checkpoint,
            round_offset,
        )
        self._reward_anchor_contract(anchor_caption_checkpoint, anchor_tts_checkpoint)
        if start_round >= round_limit:
            return current
        audio_rows = load_records(self.config["data"]["audio_only_path"], {"audio_only", "paired"}, "train")
        caption_rows = load_records(self.config["data"]["caption_only_path"], {"caption_only", "paired"}, "train")
        paired = load_records(self.config["data"]["paired_path"], {"paired"}, "train")
        limit = int(self.config["data"].get("max_records_per_role", 0))
        if limit:
            audio_rows, caption_rows, paired = audio_rows[:limit], caption_rows[:limit], paired[:limit]
        audio_codecs = self._prepare_codecs("audio_pool_codecs", audio_rows, self.run_dir / "prepared")
        paired_codecs = self._prepare_codecs("paired_anchor_codecs", paired, self.run_dir / "prepared")
        for round_index in range(start_round, round_limit):
            directory = self.run_dir / f"round_{round_index:03d}"
            start_caption, start_tts = caption_checkpoint, tts_checkpoint
            lineage = {
                "round": round_index, "same_round_start": True,
                "input_captioner": checkpoint_record(start_caption),
                "input_tts": checkpoint_record(start_tts),
                "frozen_anchor_captioner": checkpoint_record(anchor_caption_checkpoint),
                "frozen_anchor_tts": checkpoint_record(anchor_tts_checkpoint),
                "audio_pool_ids": [str(row["id"]) for row in audio_rows],
                "caption_pool_ids": [str(row["id"]) for row in caption_rows],
                "collections_complete_before_update": False,
            }
            atomic_json(directory / "lineage.json", lineage)
            audio_raw = self._collect_audio_only(
                round_index, audio_rows, audio_codecs, start_caption, start_tts,
                anchor_tts_checkpoint, directory / "collections",
            )
            caption_raw = self._collect_caption_only(
                round_index, caption_rows, start_caption, start_tts,
                anchor_caption_checkpoint, directory / "collections",
            )
            audio_sft_examples, caption_sft_examples = self._collect_sft_calibration_examples(
                round_index, paired, paired_codecs, start_caption, start_tts,
                anchor_caption_checkpoint, anchor_tts_checkpoint, directory / "sft_calibration",
            )
            lineage["collections_complete_before_update"] = True
            atomic_json(directory / "lineage.json", lineage)
            if round_index == 0:
                self._base_diagnostics(audio_raw, caption_raw)
            calibration = self._calibration(
                round_index, audio_raw, caption_raw, audio_sft_examples, caption_sft_examples,
            )
            thresholds = self._fit_sft_thresholds(
                round_index, audio_sft_examples, caption_sft_examples,
                calibration, directory / "rewards",
            )
            audio_groups, caption_groups = self._score_round(
                round_index, audio_raw, caption_raw, calibration, thresholds, directory / "rewards",
            )
            audio_summary = self._require_trainable("audio_only", audio_groups)
            caption_summary = self._require_trainable("caption_only", caption_groups)
            caption_checkpoint = self._update_captioner(
                round_index, audio_groups, caption_groups, paired, start_caption, directory,
            )
            tts_checkpoint = self._update_tts(
                round_index, audio_groups, caption_groups, paired, paired_codecs,
                audio_codecs, start_tts, directory,
            )
            self._reload_check(round_index, caption_checkpoint, tts_checkpoint, directory)
            commit = commit_round(
                self.run_dir, round_index, caption_checkpoint=caption_checkpoint,
                tts_checkpoint=tts_checkpoint, input_caption_checkpoint=start_caption,
                input_tts_checkpoint=start_tts,
            )
            self.manager.set_current(caption_checkpoint, tts_checkpoint, round_index)
            atomic_json(directory / "summary.json", {
                **lineage,
                "output_captioner": commit["captioner"], "output_tts": commit["tts"],
                "reward_calibration": str((self.run_dir / "reward_calibration.json").resolve()),
                "sft_thresholds": thresholds,
                "audio_only": audio_summary, "caption_only": caption_summary,
                "caption_anchor_rows": len(self._caption_anchors(paired)), "tts_anchor_rows": len(self._tts_anchors(paired, paired_codecs)),
                "evaluation": "disabled; checkpoints are benchmark-ready",
            })
        return self.manager.state["current"]


DualOrchestrator = DualISLOrchestrator
