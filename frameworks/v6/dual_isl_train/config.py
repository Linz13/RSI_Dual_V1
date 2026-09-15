from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import expand_env, load_yaml
from .constants import CYCLE_SFT_SELECTION


ENV_OVERRIDES = {
    "DUALISL_RUN_DIR": ("run", "output_dir"),
    "DUALISL_PAIRED_PATH": ("data", "paired_path"),
    "DUALISL_AUDIO_ONLY_PATH": ("data", "audio_only_path"),
    "DUALISL_CAPTION_ONLY_PATH": ("data", "caption_only_path"),
    "DUALISL_CAPTION_PYTHON": ("captioner", "python"),
    "DUALISL_CAPTION_MODEL": ("captioner", "model_path"),
    "DUALISL_CAPTION_ADAPTER": ("captioner", "adapter_path"),
    "DUALISL_TTS_PYTHON": ("tts", "python"),
    "DUALISL_TTS_MODEL": ("tts", "model_path"),
    "DUALISL_TTS_TOKENIZER": ("tts", "tokenizer_path"),
    "DUALISL_TTS_ADAPTER": ("tts", "adapter_path"),
    "DUALISL_CRITIC_PYTHON": ("critics", "python"),
    "DUALISL_WHISPER_MODEL": ("critics", "whisper_model"),
}

INTEGER_ENV_OVERRIDES = {
    "DUALISL_ROUNDS": ("training", "rounds"),
}


def _resolve_paths(value: Any, base: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: _resolve_paths(item, base, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base, key) for item in value]
    if isinstance(value, str) and value and (
        key.endswith("_path") or key.endswith("_dir") or key.endswith("_root")
        or key in {"python", "whisper_model"}
    ):
        path = Path(value)
        return str(path if path.is_absolute() else (base / path).resolve())
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_config_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Circular config inheritance: {chain}")
    raw = expand_env(load_yaml(path))
    parent = raw.pop("extends", None)
    current = _resolve_paths(raw, path.parent)
    if not parent:
        return current
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(_load_config_tree(parent_path, (*stack, path)), current)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _load_config_tree(config_path)
    for variable, target in ENV_OVERRIDES.items():
        value = os.environ.get(variable)
        if value and value.strip():
            config[target[0]][target[1]] = str(Path(value).expanduser().resolve())
    for variable, target in INTEGER_ENV_OVERRIDES.items():
        value = os.environ.get(variable)
        if value and value.strip():
            try:
                config[target[0]][target[1]] = int(value)
            except ValueError as exc:
                raise ValueError(f"{variable} must be an integer") from exc
    if config.get("version") == 5 and not config.get("labeling", {}).get("cache_dir"):
        config.setdefault("labeling", {})["cache_dir"] = str(Path(config["run"]["output_dir"]) / "label_cache")
    if config.get("version") == 5:
        config["tts"]["codec_cache_dir"] = str(Path(config["run"]["output_dir"]) / "codec_cache/qwen3tts_12hz")
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def _worker_section(config: dict[str, Any], role: str) -> None:
    section = config[role]
    for key in ("python", "worker_module"):
        if not str(section.get(key, "")).strip():
            raise ValueError(f"{role}.{key} is required")


def validate_config(config):
    from .config_v6 import validate
    validate(config)


def public_config(config):
    value = deepcopy(config)
    value.pop("_config_path", None)
    return value
