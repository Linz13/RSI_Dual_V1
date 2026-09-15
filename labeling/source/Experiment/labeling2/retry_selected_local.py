from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from labeling2.manifest import load_manifest
    from labeling2.pipeline import latest_by_id, run_local_model
    from labeling2.run_labeling import load_config
except ModuleNotFoundError:  # pragma: no cover - package invocation
    from .manifest import load_manifest
    from .pipeline import latest_by_id, run_local_model
    from .run_labeling import load_config


LOCAL_MODELS = ("qwen3_captioner", "kimi_audio", "step_audio_r1_1")


def requested_paths(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            values.append(str(Path(value).expanduser().resolve()))
    if not values:
        raise ValueError(f"no audio paths found in {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate audio paths found in {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun a local labeling2 backend for only explicitly selected audio files."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=LOCAL_MODELS, required=True)
    parser.add_argument("--audio-paths-file", type=Path, required=True)
    args = parser.parse_args()

    samples = load_manifest(args.manifest.expanduser().resolve(), check_audio=True)
    wanted = requested_paths(args.audio_paths_file.expanduser().resolve())
    by_audio = {str(Path(sample.audio_path).resolve()): sample for sample in samples}
    missing = [audio for audio in wanted if audio not in by_audio]
    if missing:
        raise ValueError("requested audio is not in the manifest: " + ", ".join(missing))

    selected = [by_audio[audio] for audio in wanted]
    config = load_config(args.config)
    run_dir = args.run_dir.expanduser().resolve()
    print(
        json.dumps(
            {
                "action": "targeted_local_retry",
                "model": args.model,
                "samples": len(selected),
                "sample_ids": [sample.sample_id for sample in selected],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    # Intentionally disable resume for this explicitly selected set: even a
    # formerly schema-valid row may have been selected because it contained a
    # repetition loop. run_local_model remains append-only for raw predictions.
    run_local_model(selected, args.model, run_dir, config, resume=False)

    output = run_dir / "raw_predictions" / f"{args.model}.jsonl"
    latest = latest_by_id(output)
    result = []
    failed = False
    for sample in selected:
        row = latest.get(sample.sample_id, {})
        ok = row.get("status") == "success" and not row.get("field_errors")
        failed = failed or not ok
        result.append(
            {
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "status": row.get("status", "missing"),
                "parse_status": row.get("parse_status", "missing"),
                "response_chars": len(str(row.get("response_text", ""))),
                "field_errors": row.get("field_errors", {}),
            }
        )
    print(json.dumps({"targeted_retry_results": result}, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
