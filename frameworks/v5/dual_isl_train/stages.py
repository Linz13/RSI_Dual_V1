from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import _make_shared_writable, atomic_json, hash_path, sha256_file, stable_hash
from .constants import FRAMEWORK_VERSION, RUN_STATE_VERSION


@dataclass
class StageResult:
    name: str
    output_path: Path
    checkpoint_path: Path | None = None
    skipped: bool = False


class StageManager:
    def __init__(self, run_dir: str | Path, config: dict[str, Any]):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "run_state.json"
        package_root = Path(__file__).resolve().parent
        implementation_files = (sorted(package_root.rglob("*.py")) + sorted(package_root.rglob("*.json"))
                                + sorted((package_root.parent / "scripts").glob("*captioner_candidate.py")))
        implementation_hash = stable_hash([
            {"path": str(path.relative_to(package_root.parent)), "sha256": sha256_file(path)} for path in implementation_files
        ])
        self.config_hash = stable_hash({
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
            "implementation_hash": implementation_hash,
        })
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("version") != RUN_STATE_VERSION or self.state.get("framework_version") != FRAMEWORK_VERSION:
                raise RuntimeError("Existing run_state.json is not a DualISL-Train run state")
            if self.state.get("config_hash") != self.config_hash:
                raise RuntimeError("Existing run_state.json was created with a different configuration or implementation")
        else:
            self.state = {
                "version": RUN_STATE_VERSION, "framework_version": FRAMEWORK_VERSION,
                "config_hash": self.config_hash, "stages": {}, "current": {},
            }
            self.save()

    def save(self) -> None:
        atomic_json(self.state_path, self.state)

    def complete(
        self, name: str, input_path: Path, output_path: Path, checkpoint: Path | None = None,
        invocation_hash: str | None = None,
    ) -> None:
        self.state["stages"][name] = {
            "status": "complete", "input_path": str(input_path), "input_sha256": sha256_file(input_path),
            "output_path": str(output_path), "output_sha256": sha256_file(output_path),
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": hash_path(checkpoint), "invocation_hash": invocation_hash,
            "completed_at": time.time(),
        }
        self.save()

    def fail(self, name: str, input_path: Path, error: str) -> None:
        self.state["stages"][name] = {
            "status": "failed", "input_path": str(input_path), "input_sha256": sha256_file(input_path),
            "error": error, "failed_at": time.time(),
        }
        self.save()

    def reusable(self, name: str, input_path: Path, invocation_hash: str | None = None) -> StageResult | None:
        entry = self.state["stages"].get(name)
        if not entry or entry.get("status") != "complete":
            return None
        output = Path(entry["output_path"])
        if entry.get("input_sha256") != sha256_file(input_path) or not output.is_file():
            return None
        if entry.get("invocation_hash") != invocation_hash:
            return None
        if entry.get("output_sha256") != sha256_file(output):
            return None
        checkpoint = Path(entry["checkpoint_path"]) if entry.get("checkpoint_path") else None
        if checkpoint is not None and entry.get("checkpoint_sha256") != hash_path(checkpoint):
            return None
        return StageResult(name, output, checkpoint, skipped=True)

    def set_current(self, caption_checkpoint: str | None, tts_checkpoint: str | None, round_index: int) -> None:
        self.state["current"] = {
            "caption_checkpoint": caption_checkpoint,
            "caption_checkpoint_sha256": hash_path(caption_checkpoint),
            "tts_checkpoint": tts_checkpoint,
            "tts_checkpoint_sha256": hash_path(tts_checkpoint),
            "round": round_index,
        }
        self.save()


def run_worker(
    *, python: str, module: str, action: str, config_path: str, input_path: Path, output_path: Path,
    checkpoint_in: str | None, target_checkpoint: str | None, checkpoint_out: Path | None,
    log_path: Path, project_root: Path, training_phase: str | None = None,
    distributed: dict[str, Any] | None = None,
) -> None:
    worker_arguments = [module, action, "--config", config_path, "--input", str(input_path), "--output", str(output_path)]
    if checkpoint_in:
        worker_arguments.extend(["--checkpoint-in", checkpoint_in])
    if target_checkpoint:
        worker_arguments.extend(["--target-checkpoint", target_checkpoint])
    if checkpoint_out is not None:
        worker_arguments.extend(["--checkpoint-out", str(checkpoint_out)])
    if training_phase:
        worker_arguments.extend(["--training-phase", training_phase])
    use_distributed = bool((distributed or {}).get("enabled", False)) and action in {
        "rollout", "generate-audio", "score", "score-reconstruction", "grpo-update", "sft-update",
    }
    if use_distributed:
        world_size = int(distributed["world_size"])
        command = [
            python, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={world_size}", "--module", *worker_arguments,
        ]
    else:
        command = [python, "-m", *worker_arguments]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(project_root) + (os.pathsep + current_pythonpath if current_pythonpath else "")
    worker_bin = str(Path(python).resolve().parent)
    environment["PATH"] = worker_bin + (os.pathsep + environment.get("PATH", "") if environment.get("PATH") else "")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        subprocess.run(command, cwd=project_root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    # safetensors may write mode 0600 despite the launcher's shared umask.
    # Publish readable checkpoints at the stage boundary, including if the next
    # round is later interrupted before the launcher's atexit cleanup finishes.
    if checkpoint_out is not None:
        for path in checkpoint_out.rglob("*"):
            if path.is_file() and not path.is_symlink():
                _make_shared_writable(path)
