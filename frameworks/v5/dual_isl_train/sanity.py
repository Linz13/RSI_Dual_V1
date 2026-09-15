from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .dual_space import canonical_synth_caption
from .render import render_qwen_request


def check_caption(value: Any) -> tuple[bool, dict[str, Any]]:
    if value is None:
        return False, {"reason": "missing_caption"}
    try:
        caption = canonical_synth_caption(value)
        transcript = str(caption["semantic_content"]["transcript"]).strip()
        if not transcript or transcript.casefold() == "unknown":
            return False, {"reason": "empty_or_unknown_transcript"}
        request = render_qwen_request(caption)
    except Exception as exc:
        return False, {"reason": "invalid_caption", "error": str(exc)}
    return True, {"reason": "accepted", "request": request}


def check_audio(path: str | Path) -> tuple[bool, dict[str, Any]]:
    target = Path(path)
    if not target.is_file() or target.stat().st_size < 1:
        return False, {"reason": "missing_or_empty_audio", "path": str(target)}
    try:
        import numpy as np
        import soundfile as sf

        info = sf.info(target)
        audio, sample_rate = sf.read(target, dtype="float32", always_2d=False)
        array = np.asarray(audio)
        finite = bool(np.isfinite(array).all())
        duration = float(len(array) / sample_rate) if sample_rate > 0 else 0.0
        valid = sample_rate > 0 and len(array) > 0 and duration > 0 and finite
        return valid, {
            "reason": "accepted" if valid else "invalid_audio_values",
            "sample_rate": int(sample_rate), "frames": int(len(array)), "duration_sec": duration,
            "finite": finite, "channels": int(info.channels),
        }
    except Exception as exc:
        return False, {"reason": "unreadable_audio", "error": str(exc), "path": str(target)}


