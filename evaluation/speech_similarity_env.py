#!/usr/bin/env python3
"""Local model paths and a dependency-light loader for SpeechBERTScore.

The upstream ``discrete-speech-metrics`` package imports unrelated MCD/PESQ
modules from its package initializer.  This loader imports only its official
SpeechBERTScore implementation, so the comparison does not require those
optional metrics.
"""

from __future__ import annotations

from importlib import metadata
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WAVLM_LARGE = ROOT.parent / "models/WavLM-Large-SpeechMetrics"
WAV2VEC_XLSR = ROOT.parent / "cache/labeling2/huggingface/hub/models--facebook--wav2vec2-large-xlsr-53/snapshots/c3f9d884181a224a6ac87bf8885c84d1cff3384f"


def load_official_speechbertscore_module():
    """Load only upstream ``speechbertscore.py`` without package side effects."""
    package_root = Path(metadata.distribution("discrete-speech-metrics").locate_file("discrete_speech_metrics"))
    source = package_root / "speechbertscore.py"
    if not source.is_file():
        raise FileNotFoundError(f"SpeechBERTScore source not found: {source}")
    spec = importlib.util.spec_from_file_location("official_speechbertscore", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SpeechBERTScore source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_local_assets() -> dict[str, str]:
    """Check immutable local assets without downloading or using a GPU."""
    required = {
        "wavlm_config": WAVLM_LARGE / "config.json",
        "wavlm_preprocessor": WAVLM_LARGE / "preprocessor_config.json",
        "wavlm_weights": WAVLM_LARGE / "pytorch_model.bin",
        "wav2vec_config": WAV2VEC_XLSR / "config.json",
        "wav2vec_preprocessor": WAV2VEC_XLSR / "preprocessor_config.json",
        "wav2vec_weights": WAV2VEC_XLSR / "pytorch_model.bin",
    }
    missing = [f"{key}={path}" for key, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local speech metric assets: " + "; ".join(missing))
    return {key: str(path) for key, path in required.items()}


if __name__ == "__main__":
    import json

    assets = verify_local_assets()
    module = load_official_speechbertscore_module()
    print(json.dumps({"assets": assets, "speechbertscore_module": str(module.__file__)}, indent=2))
