"""Deterministic CPU-only integration fixture; never selected in production."""
from .dual_space import project_synth_caption
from .workers.mock import mock_caption
from .attribute_reward import JUDGE_FIELDS


class MockLabelService:
    def __init__(self, config):
        assert config["run"]["inline_mock"] is True

    def labels(self, rows):
        return {r["id"]: project_synth_caption(mock_caption(
            int(str(r["id"]).rsplit("::", 1)[1]) if "::" in str(r["id"]) else 0)) for r in rows}

    def judge(self, reference, generated):
        return {field: 1.0 for field in JUDGE_FIELDS}

    def watch(self, directory, stop):
        stop.wait()

    def close(self):
        pass
