#!/usr/bin/env python3
"""Strict, auditable PEFT adapter loading shared by local benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def describe_adapter(
    adapter_dir: Path | str | None,
    base_model_dir: Path | str,
) -> dict[str, Any] | None:
    """Validate an optional PEFT adapter and return immutable identity metadata."""
    if adapter_dir is None:
        return None
    adapter_path = Path(adapter_dir).expanduser().resolve()
    base_path = Path(base_model_dir).expanduser().resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    config_path = adapter_path / ADAPTER_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"PEFT adapter config not found: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read PEFT adapter config {config_path}: {exc}") from exc

    configured_base = config.get("base_model_name_or_path")
    if not isinstance(configured_base, str) or not configured_base.strip():
        raise ValueError(f"Adapter config has no base_model_name_or_path: {config_path}")
    configured_base_path = Path(configured_base).expanduser().resolve()
    if configured_base_path != base_path:
        raise ValueError(
            "Adapter/base mismatch: "
            f"adapter expects {configured_base_path}, requested base is {base_path}"
        )

    weight_path = next(
        (adapter_path / name for name in ADAPTER_WEIGHT_NAMES if (adapter_path / name).is_file()),
        None,
    )
    if weight_path is None:
        raise FileNotFoundError(
            f"No PEFT adapter weights found in {adapter_path}; "
            f"expected one of {ADAPTER_WEIGHT_NAMES}"
        )
    if not os.access(weight_path, os.R_OK):
        raise PermissionError(
            f"PEFT adapter weights are not readable on this host: {weight_path}"
        )

    return {
        "path": str(adapter_path),
        "base_model_name_or_path": str(configured_base_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "weights_path": str(weight_path),
        "weights_sha256": sha256_file(weight_path),
        "weights_bytes": weight_path.stat().st_size,
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout"),
        "target_modules": config.get("target_modules"),
    }


def load_peft_adapter(
    base_model: Any,
    descriptor: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Attach a frozen default adapter and prove that LoRA parameters exist."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        base_model,
        descriptor["path"],
        is_trainable=False,
        autocast_adapter_dtype=True,
    )
    model.set_adapter("default")
    # PEFT 0.18 can preserve requires_grad=True from an adapter config even
    # when is_trainable=False.  Evaluation must be unconditionally frozen.
    model.requires_grad_(False)
    model.eval()
    adapter_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_" in name and ".default." in name
    ]
    if not adapter_parameters:
        raise RuntimeError(
            f"PEFT reported a loaded adapter but exposed no default LoRA parameters: "
            f"{descriptor['path']}"
        )
    if any(parameter.requires_grad for _, parameter in adapter_parameters):
        raise RuntimeError("Inference adapter unexpectedly contains trainable parameters")
    runtime = {
        **descriptor,
        "adapter_name": "default",
        "parameter_tensors": len(adapter_parameters),
        "parameter_numel": sum(parameter.numel() for _, parameter in adapter_parameters),
        "is_trainable": False,
    }
    return model, runtime
