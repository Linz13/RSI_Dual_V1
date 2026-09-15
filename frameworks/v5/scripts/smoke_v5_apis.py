"""Two existing audio fixtures through the new remote prompts; no local model execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from dual_isl_train.config import load_config
from dual_isl_train.io import atomic_json, read_jsonl
from dual_isl_train.labeling import LabelService, MODEL_FIELDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[2] / "data/labeling2/cpu_smoke_manifest.jsonl")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/v5_midasheng.yaml"))
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg["labeling"]["cache_dir"] = str(args.output.resolve() / "cache")
    svc = LabelService(cfg)
    try:
        rows = list(read_jsonl(args.manifest))[:2]
        if len(rows) != 2:
            raise ValueError("API smoke requires two existing audio fixtures")
        keyed = [(r, svc.submit(r)) for r in rows]
        results = []
        for row, key in keyed:
            for model in MODEL_FIELDS:
                attrs = svc.futures[(key, model)].result()
                results.append({"audio_path": row["audio_path"], "evaluator": model, "attributes": attrs})
        reference = {"paralinguistic": {"prosody": "Steady intonation and an even rhythm.",
                                       "pause": "There is a brief pause at the end of the sentence."}}
        equivalent = {"paralinguistic": {"prosody": "The rhythm is even and the intonation stays steady.",
                                        "pause": "A short pause occurs at the sentence ending."}}
        judged = svc.judge(reference, equivalent)
        if set(judged.values()) != {1} or len(judged) != 2:
            raise ValueError("Text judge failed equivalent-description smoke")
        report = {"ok": True, "scope": "remote API integration only; no GPU models or audio end-to-end training",
                  "audio_results": results, "text_judge_scores": judged}
        atomic_json(args.output / "summary.json", report)
        print(json.dumps({"ok": True, "audio_requests": 4, "text_judge_pairs": 2, "report": str(args.output / "summary.json")}))
    finally:
        svc.close()


if __name__ == "__main__":
    main()
