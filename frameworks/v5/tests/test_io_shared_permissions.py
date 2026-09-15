from __future__ import annotations

import stat

from dual_isl_train.io import atomic_json, write_jsonl


def test_atomic_outputs_are_shared_writable_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("DUALISL_SHARED_WRITABLE", "1")
    json_path = tmp_path / "state.json"
    jsonl_path = tmp_path / "rows.jsonl"

    atomic_json(json_path, {"ok": True})
    write_jsonl(jsonl_path, [{"id": "one"}])

    assert stat.S_IMODE(json_path.stat().st_mode) == 0o666
    assert stat.S_IMODE(jsonl_path.stat().st_mode) == 0o666
