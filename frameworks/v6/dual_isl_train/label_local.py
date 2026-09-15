"""One local labeling phase per process, so GPU models are released on exit."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path

from .io import read_jsonl, write_jsonl
from .partial_caption import field_schema, normalize_field
from jsonschema import Draft202012Validator


def main():
    p = argparse.ArgumentParser()
    for name in ("phase", "config", "input", "output", "transcripts"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rows = list(read_jsonl(args.input))
    old = {r["id"]: r for r in read_jsonl(args.output)} if Path(args.output).is_file() else {}
    if set(old) == {r["id"] for r in rows} and all(v["status"] == "complete" for v in old.values()):
        return
    device = cfg.get("local_device", "cuda:0")
    cached = [old[r["id"]] for r in rows if old.get(r["id"], {}).get("status") == "complete"]
    rows = [r for r in rows if old.get(r["id"], {}).get("status") != "complete"]
    if args.phase == "asr":
        import torch
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(cfg["asr"]["model_path"], dtype=torch.bfloat16,
            device_map=device, attn_implementation="sdpa", local_files_only=True,
            max_inference_batch_size=cfg["asr"].get("batch_size", 4), max_new_tokens=512)
        outputs = []
        for row in rows:
            try:
                prediction = model.transcribe(audio=row["audio_path"], return_time_stamps=False)[0]
                # Empty transcript is legitimate uncertainty; never use Whisper as an implicit gate/fallback.
                outputs.append({"id": row["id"], "status": "complete", "values": {
                    "semantic_content.transcript": prediction.text.strip() or "unknown"}})
            except Exception as exc:
                outputs.append({"id": row["id"], "status": "pending", "values": {}, "error_type": type(exc).__name__})
    else:
        os.environ["AUDIO_CAPTION_ROOT"] = cfg["source_root"]
        path = Path(cfg["source_root"]) / "Experiment/labeling2/expert_worker.py"
        spec = importlib.util.spec_from_file_location("v5_expert", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ecfg = cfg["backend_experts"]
        model = module.emotion_model(device, ecfg["emotion_model_dir"]) if args.phase == "emotion" else None
        transcripts = {r["id"]: r["values"].get("semantic_content.transcript", "unknown") for r in read_jsonl(args.transcripts)}
        needs_rate = args.phase.startswith("rate") and any(
            r["language"] == ("English" if args.phase.endswith("_en") else "Chinese") and transcripts[r["id"]] != "unknown" for r in rows)
        vad = module.load_brouhaha_vad(device) if needs_rate else None
        outputs = []
        field = {"volume": "paralinguistic.volume_level", "emotion": "paralinguistic.emotion",
            "accent_en": "speaker_profile.accent", "accent_zh": "speaker_profile.accent",
            "rate_en": "paralinguistic.speaking_rate", "rate_zh": "paralinguistic.speaking_rate"}[args.phase]
        for row in rows:
            output = {"id": row["id"], "status": "complete", "values": {}}
            language = row["language"]
            required = not args.phase.endswith(("_en", "_zh")) or language == ("English" if args.phase.endswith("_en") else "Chinese")
            if not required:
                if language not in ("English", "Chinese"):
                    output["values"][field] = "unknown"
                outputs.append(output)
                continue
            try:
                path = Path(row["audio_path"])
                transcript = transcripts[row["id"]]
                if args.phase.startswith("rate") and transcript == "unknown":
                    label, evidence = "unknown", {"reason": "no_transcript"}
                elif args.phase == "volume":
                    label, evidence = module.volume_label(path)
                elif args.phase == "emotion":
                    label, evidence = module.emotion_label(model, path)
                elif args.phase == "accent_en":
                    label, evidence = module.accent_model(path, device, ecfg["accent_model_dir"])
                elif args.phase == "accent_zh":
                    label, evidence = module.voxlect_model(path, device)
                else:
                    fn = module.english_rate if args.phase == "rate_en" else module.chinese_rate
                    label, evidence = fn(transcript, path, device, "turbo", vad)
                label = normalize_field(label, field_schema(field))
                if list(Draft202012Validator(field_schema(field)).iter_errors(label)):
                    raise ValueError("expert returned invalid schema value")
                output["values"][field] = label
                output["evidence"] = evidence
            except Exception as exc:
                if isinstance(exc, ValueError) and str(exc) in {"vad_no_speech", "empty_transcript", "no_chinese_phone_units"}:
                    output["values"][field] = "unknown"
                    output["evidence"] = {"reason": str(exc)}
                else:
                    output.update(status="pending", error_type=type(exc).__name__)
            outputs.append(output)
    outputs = cached + outputs
    write_jsonl(args.output, outputs)
    if any(r["status"] != "complete" for r in outputs):
        raise SystemExit("Local labels pending; see phase output, resume after resolving failure")


if __name__ == "__main__":
    main()
