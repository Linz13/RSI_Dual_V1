from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
from .checkpoints import checkpoint_record
from .config import public_config
from .constants import SOURCE_DOMAIN_TARGET
from .io import atomic_json, dump_yaml, stable_hash, write_jsonl
from .stages import StageManager, StageResult, run_worker


class StageRunner:
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

    def _seed(self, round_index: int, branch: str, sample_id: str, candidate_index: int) -> int:
        digest = stable_hash({
            "base_seed": int(self.config["run"].get("seed", 42)),
            "round": round_index, "branch": branch,
            "sample_id": str(sample_id), "candidate_index": candidate_index,
        })
        return int(digest[:15], 16) % (2**31 - 1)

