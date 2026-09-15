from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a random_audio_440 sampling manifest to labeling2 format")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    seen: set[str] = set()
    with args.input.resolve().open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            item = json.loads(line)
            dataset = str(item["dataset"])
            index = int(item["sample_index"])
            sample_id = f"random440_{dataset.casefold()}_{index:03d}"
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id at line {line_no}: {sample_id}")
            audio_path = Path(item["output_path"]).expanduser().resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            metadata = dict(item.get("metadata") or {})
            metadata.update({
                "sampling_manifest_line": line_no,
                "source_path": item.get("source_path", ""),
                "relative_source_path": item.get("relative_source_path", ""),
            })
            rows.append({
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "dataset": dataset,
                "metadata": metadata,
                "transcript": metadata.get("value") if dataset == "ExpressiveSpeech" else None,
                "language_hint": "Chinese" if dataset == "ExpressiveSpeech" else None,
            })
            seen.add(sample_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output.resolve()), "samples": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
