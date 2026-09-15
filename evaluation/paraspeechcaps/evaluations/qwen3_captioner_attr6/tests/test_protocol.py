from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[2]))

from common import (  # noqa: E402
    duration_balanced_partition,
    load_schema,
    parse_prediction,
    parse_prediction_with_violations,
)
from prepare_manifest import build_samples, validate_samples  # noqa: E402
from score import bootstrap_ci, positive_recall, set_f1  # noqa: E402
from model_adapter_utils import describe_adapter, sha256_json  # noqa: E402
from run_qwen3_captioner import evaluation_identity, parse_args  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_manifest_normalization_and_repairs(self):
        schema = load_schema()
        samples = build_samples(ROOT.parents[1] / "data/test_available.csv", schema)
        validate_samples(samples, schema)
        self.assertEqual(len(samples), 140)
        repaired = {
            index
            for sample in samples
            for index in sample["benchmark_indices"]
            if any("tag_of_interest/reference audit" in action for action in sample["normalization_actions"])
        }
        self.assertEqual(repaired, {65, 68, 69})
        for sample in samples:
            self.assertNotIn(sample["gt"]["accent"], sample["gt"]["intrinsic_traits"])
            self.assertNotIn("enunciated", sample["gt"]["situational_traits"])

    def test_parser_accepts_fence_deduplicates_and_normalizes(self):
        raw = """```json
        {"gender":"female","pitch":"high pitched","speaking_rate":"slow speed",
         "accent":"unknown","intrinsic_traits":["vocal fry","vocal-fry","crisp"],
         "situational_traits":["happy","happy"]}
        ```"""
        value = parse_prediction(raw)
        self.assertEqual(value["pitch"], "high-pitched")
        self.assertEqual(value["accent"], "unknown")
        self.assertEqual(value["intrinsic_traits"], ["crisp", "vocal-fry"])
        self.assertEqual(value["situational_traits"], ["happy"])

    def test_parser_rejects_missing_extra_and_unknown_multi_label(self):
        base = {
            "gender": "female", "pitch": "high-pitched", "speaking_rate": "slow speed",
            "accent": "american", "intrinsic_traits": [], "situational_traits": [],
        }
        for value in (
            {key: item for key, item in base.items() if key != "accent"},
            {**base, "explanation": "x"},
            {**base, "intrinsic_traits": ["not-a-label"]},
        ):
            with self.assertRaises(ValueError):
                parse_prediction(json.dumps(value))

    def test_cross_field_candidate_is_scored_not_discarded(self):
        value = {
            "gender": "female", "pitch": "high-pitched", "speaking_rate": "slow speed",
            "accent": "american", "intrinsic_traits": ["whispered"],
            "situational_traits": [],
        }
        parsed, violations = parse_prediction_with_violations(json.dumps(value))
        self.assertEqual(parsed["intrinsic_traits"], ["whispered"])
        self.assertEqual(violations, ["cross_field_label:intrinsic_traits:whispered"])

    def test_set_metrics(self):
        self.assertEqual(set_f1([], []), 1.0)
        self.assertEqual(set_f1(["a"], []), 0.0)
        self.assertAlmostEqual(set_f1(["a", "b"], ["b", "c"]), 0.5)
        self.assertIsNone(positive_recall([], ["a"]))
        self.assertEqual(positive_recall(["a", "b"], ["a"]), 0.5)

    def test_duration_balanced_partition_is_disjoint_and_complete(self):
        samples = [
            {"sample_id": f"s{i}", "duration_seconds": duration}
            for i, duration in enumerate([10, 9, 8, 7, 6, 5])
        ]
        shards = duration_balanced_partition(samples, 2)
        identifiers = [row["sample_id"] for shard in shards for row in shard]
        self.assertEqual(sorted(identifiers), sorted(row["sample_id"] for row in samples))
        self.assertEqual(len(identifiers), len(set(identifiers)))
        totals = [sum(row["duration_seconds"] for row in shard) for shard in shards]
        self.assertLessEqual(abs(totals[0] - totals[1]), 5)

    def test_bootstrap_is_deterministic(self):
        self.assertEqual(bootstrap_ci([0.0, 0.5, 1.0], rounds=100, seed=7), bootstrap_ci([0.0, 0.5, 1.0], rounds=100, seed=7))

    def test_adapter_descriptor_is_hashed_and_rejects_wrong_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            adapter = root / "adapter"
            base.mkdir()
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({
                    "base_model_name_or_path": str(base),
                    "peft_type": "LORA",
                    "r": 4,
                }),
                encoding="utf-8",
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter-test")
            descriptor = describe_adapter(adapter, base)
            self.assertEqual(descriptor["path"], str(adapter.resolve()))
            self.assertEqual(descriptor["weights_bytes"], len(b"adapter-test"))
            self.assertEqual(len(descriptor["weights_sha256"]), 64)
            self.assertEqual(len(sha256_json(descriptor)), 64)
            wrong_base = root / "wrong"
            wrong_base.mkdir()
            with self.assertRaisesRegex(ValueError, "Adapter/base mismatch"):
                describe_adapter(adapter, wrong_base)

    def test_midasheng_defaults_and_identity_include_adapter(self):
        manifest = ROOT / "runs/new_cluster_default/manifest.jsonl"
        with patch.object(
            sys,
            "argv",
            ["runner", "--backend", "midasheng", "--attn-backend", "sdpa"],
        ):
            args = parse_args()
        self.assertEqual(args.backend, "midasheng")
        identity_sha256, identity = evaluation_identity(args, manifest)
        self.assertEqual(identity["backend"], "midasheng")
        self.assertIsNone(identity["adapter"])
        self.assertEqual(len(identity_sha256), 64)


if __name__ == "__main__":
    unittest.main()
