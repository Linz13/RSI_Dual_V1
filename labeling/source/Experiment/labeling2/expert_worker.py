from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("AUDIO_CAPTION_ROOT", Path(__file__).resolve().parents[2])).resolve()

LOW_NORMAL = 0.03331899
NORMAL_HIGH = 0.05054203
EN_SLOW = 11.477411477411476
EN_FAST = 19.129019129019127
ZH_SLOW = 4.0
ZH_FAST = 10.0
_WHISPER_MODELS: dict[tuple[str, str], Any] = {}
_EN_G2P: Any | None = None
_ACCENT_METHODS: dict[str, Any] = {}
_VOXLECT_MODELS: dict[str, Any] = {}


def progress_line(expert: str, completed: int, total: int, elapsed: float) -> str:
    """Return a dependency-free progress bar suitable for tmux and log files."""
    width = 30
    ratio = completed / total if total else 1.0
    filled = min(width, int(width * ratio))
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else 0.0
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"[{expert}] [{bar}] {completed}/{total} ({ratio:6.2%}) "
        f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
    )


def rate_label(phones_per_sec: float, slow_edge: float, fast_edge: float) -> str:
    return "slow" if phones_per_sec < slow_edge else "fast" if phones_per_sec > fast_edge else "moderate"


def textrol_label(energy: float) -> str:
    if energy < LOW_NORMAL:
        return "low"
    if energy > NORMAL_HIGH:
        return "high"
    return "medium"


def volume_label(path: Path) -> tuple[str, dict[str, Any]]:
    import librosa
    y, sr = librosa.load(path)
    if len(y) == 0:
        raise ValueError("empty_audio")
    energy = float(librosa.feature.rms(y=y).mean())
    label = textrol_label(energy)
    return label, {"energy": energy, "sample_rate": sr, "duration_sec": len(y) / sr, "low_normal": LOW_NORMAL, "normal_high": NORMAL_HIGH, "method": "TextrolSpeech librosa.feature.rms(y).mean", "model_version": "TextrolSpeech energy rule"}


def load_brouhaha_vad(device: str) -> Any:
    """Load the same Brouhaha/pyannote VAD used by ParaSpeechCaps."""
    try:
        import torch
        from brouhaha.pipeline import RegressiveActivityDetectionPipeline
        from huggingface_hub import hf_hub_download
        from pyannote.audio import Model
    except ImportError as exc:
        raise RuntimeError(
            "ParaSpeechCaps rate expert requires brouhaha, pyannote.audio and "
            "huggingface_hub in the configured rate_python environment"
        ) from exc
    checkpoint = hf_hub_download(repo_id="ylacombe/brouhaha-best", filename="best.ckpt")
    model = Model.from_pretrained(Path(checkpoint), strict=False)
    pipeline = RegressiveActivityDetectionPipeline(segmentation=model, batch_size=32)
    pipeline.to(torch.device(device))
    return pipeline


def voiced_seconds(path: Path, device: str = "cpu", pipeline: Any | None = None) -> tuple[float, dict[str, Any]]:
    if pipeline is None:
        pipeline = load_brouhaha_vad(device)
    import soundfile as sf
    import torch

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.tensor(audio.mean(axis=1)[None, :], dtype=torch.float32)
    if str(device).startswith("cuda"):
        waveform = waveform.to(device)
    result = pipeline({"sample_rate": sample_rate, "waveform": waveform})
    intervals = []
    seconds = 0.0
    for segment, _track in result["annotation"].itertracks():
        intervals.append({"start_sec": float(segment.start), "end_sec": float(segment.end)})
        seconds += float(segment.duration)
    if seconds <= 0:
        raise ValueError("vad_no_speech")
    return seconds, {"sample_rate": sample_rate, "intervals": intervals, "vad": "Brouhaha RegressiveActivityDetectionPipeline"}


def whisper_transcript(path: Path, device: str, model_name: str) -> str:
    import whisper
    environment_bin = str(Path(sys.executable).resolve().parent)
    if (Path(environment_bin) / "ffmpeg").is_file():
        os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
    key = (str(device), str(model_name))
    model = _WHISPER_MODELS.get(key)
    if model is None:
        checkpoint_root = Path(os.environ.get("LABELING2_WHISPER_CACHE", ROOT / "Experiment/acc_model_pool/Caption_Bench/methods_low/checkpoints/whisper"))
        model = whisper.load_model(model_name, device=device, download_root=str(checkpoint_root))
        _WHISPER_MODELS[key] = model
    result = model.transcribe(str(path), fp16=str(device).startswith("cuda"), verbose=False)
    return str(result.get("text", "")).strip()


def english_rate(transcript: str, path: Path, device: str, model_name: str, vad_pipeline: Any | None = None) -> tuple[str, dict[str, Any]]:
    if not transcript:
        transcript = whisper_transcript(path, device, model_name)
    if not transcript:
        raise ValueError("empty_transcript")
    global _EN_G2P
    if _EN_G2P is None:
        from g2p import make_g2p
        _EN_G2P = make_g2p("eng", "eng-ipa")
    phones = str(_EN_G2P(transcript).output_string)
    count = len(phones)
    duration, vad = voiced_seconds(path, device=device, pipeline=vad_pipeline)
    pps = count / duration
    label = rate_label(pps, EN_SLOW, EN_FAST)
    return label, {"transcript": transcript, "phoneme_string": phones, "phoneme_count": count, "speech_duration_sec": duration, "phones_per_sec": pps, "slow_edge": EN_SLOW, "fast_edge": EN_FAST, **vad, "method": "ParaSpeechCaps/DataSpeech PPS", "model_version": "DataSpeech v02 edges"}


def chinese_phone_count(transcript: str) -> int:
    from pypinyin import Style, pinyin
    chars = "".join(char for char in transcript if "\u3400" <= char <= "\u9fff")
    if not chars:
        return 0
    initials = pinyin(chars, style=Style.INITIALS, strict=False)
    finals = pinyin(chars, style=Style.FINALS, strict=False)
    return sum(bool(item[0].strip()) for item in initials) + sum(bool(item[0].strip()) for item in finals)


def chinese_rate(transcript: str, path: Path, device: str, model_name: str, vad_pipeline: Any | None = None) -> tuple[str, dict[str, Any]]:
    if not transcript:
        transcript = whisper_transcript(path, device, model_name)
    phones = chinese_phone_count(transcript)
    if phones <= 0:
        raise ValueError("no_chinese_phone_units")
    duration, vad = voiced_seconds(path, device=device, pipeline=vad_pipeline)
    pps = phones / duration
    label = rate_label(pps, ZH_SLOW, ZH_FAST)
    return label, {"transcript": transcript, "phone_count": phones, "speech_duration_sec": duration, "phones_per_sec": pps, "slow_edge": ZH_SLOW, "fast_edge": ZH_FAST, "method": "Learning Speech Rate in Speech Recognition inspired pinyin initial/final rate", "model_version": "paper-inspired deployment heuristic", **vad}


def emotion_model(device: str, model_dir: str = ""):
    from funasr import AutoModel
    path = Path(model_dir).expanduser().resolve() if model_dir else None
    if path is None or not path.is_dir():
        raise FileNotFoundError(f"emotion_model_dir_not_found:{path}")
    model = AutoModel(model=str(path), device=device, disable_update=True)
    return model


def emotion_label(model: Any, path: Path) -> tuple[str, dict[str, Any]]:
    result = model.generate(input=str(path), granularity="utterance", extract_embedding=False)
    item = result[0] if isinstance(result, list) else result
    labels = [str(value) for value in item.get("labels", [])]
    scores = [float(value) for value in item.get("scores", [])]
    if not labels:
        raw = str(item.get("text", item.get("label", ""))).casefold()
        labels = [raw]
        scores = [float(item.get("score", 0.0) or 0.0)]
    index = max(range(len(labels)), key=lambda i: scores[i] if i < len(scores) else -math.inf)
    raw = labels[index].casefold()
    aliases = {"ang": "angry", "anger": "angry", "angry": "angry", "hap": "happy", "happy": "happy", "sad": "sad", "sadness": "sad", "neu": "neutral", "neutral": "neutral", "fea": "fearful", "fear": "fearful", "fearful": "fearful", "sur": "surprised", "surprise": "surprised", "surprised": "surprised", "dis": "disgusted", "disgust": "disgusted", "disgusted": "disgusted"}
    mapped = next((value for key, value in aliases.items() if key in raw), "other")
    return mapped, {"raw_label": labels[index], "score": scores[index] if index < len(scores) else None, "labels": labels, "scores": scores, "model": "emotion2vec_plus_large", "model_version": "emotion2vec/emotion2vec_plus_large"}


def normalize_accent_label(raw: str) -> str:
    key = "".join(char for char in raw.casefold() if char.isalnum())
    mapping = {
        "us": "US English", "american": "US English", "generalamerican": "US English",
        "england": "England English", "british": "England English", "english": "England English",
        "australia": "Australian English", "australian": "Australian English",
        "india": "Indian English", "indian": "Indian English",
        "canada": "Canadian English", "canadian": "Canadian English",
        "bermuda": "Bermudian English", "bermudian": "Bermudian English",
        "scotland": "Scottish English", "scottish": "Scottish English",
        "africa": "African English", "african": "African English",
        "ireland": "Irish English", "irish": "Irish English",
        "newzealand": "New Zealand English",
        "wales": "Welsh English", "welsh": "Welsh English",
        "malaysia": "Malaysian English", "malaysian": "Malaysian English",
        "philippines": "Philippine English", "philippine": "Philippine English",
        "singapore": "Singapore English", "singaporean": "Singapore English",
        "hongkong": "Hong Kong English",
        # The released checkpoint contains the misspelled label
        # ``southatlandtic`` in its label encoder.
        "southatlandtic": "South Atlantic English", "southatlantic": "South Atlantic English",
    }
    return mapping.get(key, "other")


def accent_model(path: Path, device: str, model_dir: str = "") -> tuple[str, dict[str, Any]]:
    source = Path(model_dir).expanduser().resolve() if model_dir else None
    if source is None or not source.is_dir():
        raise FileNotFoundError(f"accent_model_dir_not_found:{source}")
    method = _ACCENT_METHODS.get(str(device))
    if method is None:
        try:
            from speechbrain.inference.interfaces import foreign_class
        except ImportError:
            from speechbrain.pretrained.interfaces import foreign_class
        overrides = {"pretrained_path": str(source)}
        # SpeechBrain's YAML uses a separate cache_dir for the base encoder.
        # An explicit local snapshot keeps offline deployment on the pinned model.
        base_model_dir = os.environ.get("LABELING2_ACCENT_BASE_MODEL_DIR")
        if base_model_dir:
            base_source = Path(base_model_dir).expanduser().resolve()
            if not base_source.is_dir():
                raise FileNotFoundError(f"accent_base_model_dir_not_found:{base_source}")
            overrides["wav2vec2_hub"] = str(base_source)
        method = foreign_class(
            source=str(source),
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": device},
            overrides=overrides,
        )
        _ACCENT_METHODS[str(device)] = method
    result = method.classify_file(str(path))
    scores, _, _, labels = result
    raw = str(labels[0])
    mapped = normalize_accent_label(raw)
    return mapped, {"raw_label": raw, "scores": scores.squeeze().detach().cpu().tolist(), "method": "accent_xlsr", "model_version": "Jzuluaga/accent-id-commonaccent_xlsr-en-english"}


def voxlect_model(path: Path, device: str) -> tuple[str, dict[str, Any]]:
    import sys
    import numpy as np
    import soundfile as sf
    import torch
    voxlect_root = ROOT / "Experiment/acc_model_pool/Caption_Bench/third_party/voxlect"
    sys.path.insert(0, str(voxlect_root))
    from src.model.dialect.whisper_dialect import WhisperWrapper
    labels = ["Jiang-Huai", "Jiao-Liao", "Ji-Lu", "Lan-Yin", "Mandarin", "Southwestern", "Zhongyuan", "Cantonese"]
    model = _VOXLECT_MODELS.get(str(device))
    if model is None:
        model = WhisperWrapper.from_pretrained("tiantiaf/voxlect-mandarin-cantonese-dialect-whisper-small", pretrain_model="whisper_small", output_class_num=8).to(device).eval()
        _VOXLECT_MODELS[str(device)] = model
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if np.ndim(audio) == 2:
        audio = np.mean(audio, axis=1)
    with torch.inference_mode():
        logits = model(torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(device))
        logits = logits[0] if isinstance(logits, (tuple, list)) else logits
        logits = logits.squeeze()
        probs = torch.softmax(logits, dim=-1)
    index = int(torch.argmax(probs).item())
    display = {"Jiang-Huai": "Jiang-Huai Mandarin", "Jiao-Liao": "Jiao-Liao Mandarin", "Ji-Lu": "Ji-Lu Mandarin", "Lan-Yin": "Lan-Yin Mandarin", "Mandarin": "Mandarin", "Southwestern": "Southwestern Mandarin", "Zhongyuan": "Zhongyuan Mandarin", "Cantonese": "Cantonese"}
    return display[labels[index]], {"raw_label": labels[index], "confidence": float(probs[index].item()), "sample_rate": sr, "method": "voxlect-whisper-small", "model_version": "tiantiaf/voxlect-mandarin-cantonese-dialect-whisper-small"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=("volume", "emotion", "accent_en", "accent_zh", "rate_en", "rate_zh"), required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--whisper-model", default="turbo")
    parser.add_argument("--emotion-model-dir", default="")
    parser.add_argument("--emotion-model-revision", default="")
    parser.add_argument("--accent-model-dir", default="")
    parser.add_argument("--accent-model-revision", default="")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(tasks)
    progress_started = time.perf_counter()
    print(progress_line(args.expert, 0, total, 0.0), flush=True)
    try:
        model = emotion_model(args.device, args.emotion_model_dir) if args.expert == "emotion" else None
        vad_pipeline = load_brouhaha_vad(args.device) if args.expert in {"rate_en", "rate_zh"} else None
    except Exception as exc:
        with args.output.open("a", encoding="utf-8") as target:
            for completed, task in enumerate(tasks, start=1):
                target.write(json.dumps({
                    "expert": args.expert, "sample_id": task["sample_id"],
                    "audio_path": task["audio_path"], "status": "error",
                    "prediction": "", "error": repr(exc),
                }, ensure_ascii=False) + "\n")
                target.flush()
                print(progress_line(args.expert, completed, total, time.perf_counter() - progress_started), flush=True)
        print(f"[{args.expert}] model load failed: {exc!r}", flush=True)
        return
    with args.output.open("a", encoding="utf-8") as target:
        for completed, task in enumerate(tasks, start=1):
            record = {"expert": args.expert, "sample_id": task["sample_id"], "audio_path": task["audio_path"], "status": "error", "prediction": "", "error": ""}
            started = time.perf_counter()
            try:
                path = Path(task["audio_path"])
                if args.expert == "volume":
                    label, evidence = volume_label(path)
                elif args.expert == "emotion":
                    label, evidence = emotion_label(model, path)
                elif args.expert == "accent_en":
                    label, evidence = accent_model(path, args.device, args.accent_model_dir)
                elif args.expert == "accent_zh":
                    label, evidence = voxlect_model(path, args.device)
                elif args.expert == "rate_en":
                    label, evidence = english_rate(task.get("transcript") or "", path, args.device, args.whisper_model, vad_pipeline)
                else:
                    label, evidence = chinese_rate(task.get("transcript") or "", path, args.device, args.whisper_model, vad_pipeline)
                record.update({"status": "success", "prediction": label, "evidence": evidence})
            except Exception as exc:
                record["error"] = repr(exc)
            record["duration_sec"] = time.perf_counter() - started
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            print(progress_line(args.expert, completed, total, time.perf_counter() - progress_started), flush=True)


if __name__ == "__main__":
    main()
