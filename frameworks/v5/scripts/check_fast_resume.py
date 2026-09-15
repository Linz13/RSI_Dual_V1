"""Check migrated locks, committed artifacts and exact cache reuse without GPUs/APIs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from dual_isl_train.attribute_reward import JUDGE_FIELDS, field_known
from dual_isl_train.io import atomic_json, load_yaml, read_jsonl
from dual_isl_train.labeling import LabelService
from dual_isl_train.stages import StageManager
from scripts.v5_launcher import check
from scripts.verify_v5 import verify


def audit(run):
    cfg = load_yaml(run / "resolved_config.yaml")
    manager = StageManager(run, cfg)
    assert manager.state["current"]["round"] == 0
    assert cfg["tts"]["generation"]["rollout_batch_size"] == 1
    assert cfg["tts"]["generation"]["synthesis_batch_size"] == 4
    assert not (run / "round_001").exists()
    preserved = []
    for name, stage in manager.state["stages"].items():
        assert manager.reusable(name, Path(stage["input_path"]), stage.get("invocation_hash")), name
        preserved.append(name)
    verified = verify(run)
    assert verified["ok"], verified["errors"]
    source = list(read_jsonl(run / "prepared/audio_pool_codecs.input.jsonl"))
    generated = list(read_jsonl(run / "round_000/collections/audio_attribute_synthesis/round_000_audio_attribute_synthesis.output.jsonl"))
    groups = list(read_jsonl(run / "round_000/collections/round_000_audio_collection.output.jsonl"))
    def forbidden(*args, **kwargs):
        raise AssertionError("Cache verification attempted an API request or local GPU worker")
    with patch("requests.post", forbidden), patch("subprocess.run", forbidden):
        service = LabelService(cfg)
        service._request_with_metadata = forbidden
        service._local = forbidden
        try:
            labels = service.labels(source + generated)
            for future in service.futures.values():
                future.result()
            compared = 0
            for group in groups:
                reference = group["reference_attributes"]
                assert labels[group["id"]] == reference
                for candidate in group["candidates"]:
                    if "generated_attributes" not in candidate:
                        continue
                    predicted = candidate["generated_attributes"]
                    assert labels[candidate["candidate_id"]] == predicted
                    expected = {field: candidate["attribute_reconstruction"]["fields"][field]
                                for field in JUDGE_FIELDS if field_known(reference, field) and field_known(predicted, field)}
                    assert service.judge(reference, predicted) == expected
                    compared += 1
        finally:
            service.close()
    return {"ok": True, "gpu_execution": "not_performed", "api_requests": 0,
            "reusable_stages": preserved, "source_labels_reused": len(source),
            "generated_labels_reused": len(generated), "reconstruction_judgments_checked": compared,
            "next_round_index": 1, "remaining_rounds": cfg["training"]["rounds"] - 1,
            "first_round_verification": verified, "preflight": check(cfg)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0)
    os.environ["DUALISL_SHARED_WRITABLE"] = "1"
    result = audit(args.run_dir.resolve())
    atomic_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("reusable_stages", "first_round_verification")}, indent=2))
