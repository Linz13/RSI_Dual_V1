import json
from pathlib import Path

import pytest
import requests
import yaml

from scripts import resume_v5_with_judge_check as cli


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("V5_JUDGE_URL", "https://judge.example/v1")
    monkeypatch.setenv("V5_JUDGE_KEY", "fixture-secret-123")
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump({
        "labeling": {"attempts": 2, "judge_timeout": 5}}))
    (tmp_path / "run_state.json").write_text('{"fixture": true}')
    return tmp_path


def response(status=200, model="gpt-5.5"):
    result = requests.Response()
    result.status_code = status
    if status == 200:
        body = {"model": model, "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "results": [{"id": f"paralinguistic.{field}", "score": 1, "reason": "Same description"}
                        for field in ("prosody", "pause")]})}}]}
    else:
        body = {"error": {"code": "invalid_api_key", "message": "Bad key fixture-secret-123"}}
    result._content = json.dumps(body).encode()
    return result


def summary(run_dir):
    return json.loads(next((run_dir / "diagnostics").glob("*/summary.json")).read_text())


def test_http_failure_is_diagnostic_not_reward(run_dir, monkeypatch, capsys):
    original = (run_dir / "run_state.json").read_bytes()
    monkeypatch.setattr(requests, "post", lambda *a, **k: response(401))
    assert cli.check(run_dir) is False
    report = summary(run_dir)
    assert len(report["requests"]) == 2
    assert all(r["http_status"] == 401 for r in report["requests"])
    assert "fixture-secret-123" not in json.dumps(report)
    assert "fixture-secret-123" not in capsys.readouterr().out
    assert (run_dir / "run_state.json").read_bytes() == original
    assert not list((run_dir / "diagnostics").glob("*/judge/*.json"))


def test_check_uses_real_judge_request_contract(run_dir, monkeypatch):
    def post(url, **kwargs):
        assert url == "https://judge.example/v1/chat/completions"
        assert kwargs["json"]["model"] == "gpt-5.5"
        assert kwargs["json"]["response_format"]["type"] == "json_schema"
        assert kwargs["allow_redirects"] is False
        return response()
    monkeypatch.setattr(requests, "post", post)
    assert cli.check(run_dir) is True
    assert summary(run_dir)["scores"] == {"paralinguistic.prosody": 1, "paralinguistic.pause": 1}


def test_wrong_returned_model_fails_preflight(run_dir, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: response(model="other-model"))
    assert cli.check(run_dir) is False


def test_transport_failure_does_not_log_exception_credentials(run_dir, monkeypatch, capsys):
    def post(*args, **kwargs):
        raise requests.ConnectTimeout("https://user:fixture-secret-123@judge.example/private")
    monkeypatch.setattr(requests, "post", post)
    assert cli.check(run_dir) is False
    assert summary(run_dir)["requests"][0]["error_type"] == "ConnectTimeout"
    assert "fixture-secret-123" not in capsys.readouterr().out


def test_non_json_error_body_is_not_logged():
    result = requests.Response()
    result.status_code = 403
    result._content = b"<html>private proxy credentials</html>"
    assert cli.http_diagnostic(result, []) == {"http_status": 403, "response_format": "not_json"}


@pytest.mark.parametrize("ok,check_only,expected_calls", [(False, False, 0), (True, True, 0), (True, False, 1)])
def test_resume_requires_successful_check(tmp_path, monkeypatch, ok, check_only, expected_calls):
    monkeypatch.setattr(cli, "check", lambda path: ok)
    monkeypatch.setattr(cli.os, "umask", lambda value: 0)
    calls = []
    monkeypatch.setattr(cli.os, "execvp", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", ["probe", str(tmp_path)] + (["--check-only"] if check_only else []))
    assert cli.main() == (0 if ok else 1)
    assert len(calls) == expected_calls
    if calls:
        assert calls[0][1] == ["bash", str(cli.ROOT / "scripts/run_v5.sh"), "resume", str(tmp_path)]


def test_redaction_happens_before_truncation():
    secret = "sensitive" * 200
    result = cli.safe_text(secret + " Bearer unexpected-secret https://host/path?key=another", [secret])
    assert "sensitive" not in result
    assert "unexpected-secret" not in result
    assert "another" not in result
