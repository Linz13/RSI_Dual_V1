from __future__ import annotations

import math
from typing import Any

from dual_isl_train.metrics import error_rate
from dual_isl_train.distributed import DistributedContext, run_sharded_inference
from dual_isl_train.io import atomic_json
from dual_isl_train.workers.common import load_job, worker_parser, write_output


def audio_health(path: str, min_duration: float, max_duration: float) -> tuple[float, dict[str, float]]:
    import numpy as np
    import soundfile as sf
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype("float32", copy=False)
    duration = len(audio) / max(float(sample_rate), 1.0)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    clipping = float(np.mean(np.abs(audio) >= 0.999)) if len(audio) else 1.0
    silence = float(np.mean(np.abs(audio) < 1e-4)) if len(audio) else 1.0
    duration_score = 1.0 if min_duration <= duration <= max_duration else max(0.0, 1.0 - abs(duration - min(max(duration, min_duration), max_duration)) / max(max_duration, 1.0))
    peak_score = 1.0 if 1e-4 < peak <= 1.0 else 0.0
    clipping_score = max(0.0, 1.0 - clipping / 0.01)
    energy_score = min(1.0, rms / 0.03) * max(0.0, 1.0 - silence)
    score = 0.25 * (duration_score + peak_score + clipping_score + energy_score)
    return score, {"duration": duration, "peak": peak, "rms": rms, "clipping_ratio": clipping, "silence_ratio": silence}


class WhisperCritic:
    def __init__(self, config: dict[str, Any], distributed: DistributedContext | None = None):
        import whisper
        cfg = config["critics"]
        self.model = whisper.load_model(
            cfg.get("whisper_model", "turbo"), device=(distributed.device if distributed and distributed.enabled else cfg.get("device", "cuda:0")),
            download_root=cfg.get("whisper_download_root"),
        )

    def transcribe(self, path: str, language: str) -> str:
        import numpy as np
        import soundfile as sf
        language_code = "zh" if language == "Chinese" else "en" if language == "English" else None
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            divisor = math.gcd(sample_rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype(np.float32)
        result = self.model.transcribe(audio, language=language_code, fp16=True, temperature=0.0)
        return str(result.get("text", "")).strip()


def main() -> None:
    parser = worker_parser("Fixed audio health and Whisper ASR critic", ["score", "preflight"])
    args = parser.parse_args()
    config, rows = load_job(args)
    distributed = DistributedContext.initialize(config)
    try:
        critic = WhisperCritic(config, distributed)
        limits = config["data"].get("duration", {})

        def score(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
            output = []
            cache: dict[tuple[str, str], tuple[float, dict[str, float], str]] = {}
            for row in values:
                key = (str(row["audio_path"]), str(row.get("language", "English")))
                if key not in cache:
                    health, details = audio_health(row["audio_path"], float(limits.get("min", 1.2)), float(limits.get("max", 30.0)))
                    transcript = critic.transcribe(row["audio_path"], row.get("language", "English"))
                    cache[key] = health, details, transcript
                health, details, transcript = cache[key]
                asr_score = 1.0 - error_rate(str(row.get("transcript", "")), transcript, str(row.get("language", "English")))
                output.append({
                    "candidate_id": row["candidate_id"], "audio_health": health,
                    "audio_health_details": details, "asr_score": asr_score, "asr_text": transcript,
                })
            return output

        if distributed.enabled:
            _output, metrics = run_sharded_inference(rows, args.output, distributed, score)
            if distributed.is_main:
                atomic_json(str(args.output) + ".metrics.json", metrics)
        else:
            write_output(args.output, score(rows))
    finally:
        distributed.close()


if __name__ == "__main__":
    main()
