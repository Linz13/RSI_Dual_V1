from pathlib import Path

from dual_isl_train.io import atomic_json, dump_yaml, load_yaml
from scripts.migrate_label_rules import atomic_yaml


def test_shared_yaml_write_does_not_chmod_already_shared_foreign_file(tmp_path, monkeypatch):
    path = tmp_path / "resolved_config.yaml"
    path.write_text("fixture: old\n")
    path.chmod(0o666)
    monkeypatch.setenv("DUALISL_SHARED_WRITABLE", "1")
    def forbidden(*args, **kwargs):
        raise PermissionError("fixture: foreign owner cannot chmod")
    monkeypatch.setattr(Path, "chmod", forbidden)
    dump_yaml(path, {"fixture": "new"})
    assert load_yaml(path) == {"fixture": "new"}


def test_atomic_yaml_replaces_foreign_file_without_chmodding_it(tmp_path, monkeypatch):
    path = tmp_path / "resolved_config.yaml"
    path.write_text("fixture: old\n")
    original_chmod = Path.chmod
    def guarded(self, *args, **kwargs):
        if self == path:
            raise PermissionError("fixture: foreign owner cannot chmod")
        return original_chmod(self, *args, **kwargs)
    monkeypatch.setattr(Path, "chmod", guarded)
    atomic_yaml(path, {"fixture": "new"})
    assert load_yaml(path) == {"fixture": "new"}
    assert path.stat().st_mode & 0o777 == 0o666


def test_new_atomic_json_is_still_shared_writable(tmp_path, monkeypatch):
    monkeypatch.setenv("DUALISL_SHARED_WRITABLE", "1")
    path = tmp_path / "state.json"
    atomic_json(path, {"ok": True})
    assert path.stat().st_mode & 0o666 == 0o666
