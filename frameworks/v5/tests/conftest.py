from __future__ import annotations

import json
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_caption() -> dict:
    return json.loads((PROJECT / "tests/fixtures/paired.jsonl").read_text(encoding="utf-8").splitlines()[0])["caption"]

