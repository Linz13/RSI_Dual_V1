"""Offline request/response checks for the two deployed API transports."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Experiment.labeling2.pipeline import api_module


class APITransportTests(unittest.TestCase):
    def test_gemini_labeling_identity_and_native_endpoint(self):
        module = api_module("gemini")
        args = SimpleNamespace(api_key="example-only", base_url=module.DEFAULT_BASE_URL,
                               api_model="gemini-3.1-pro-preview", timeout=300,
                               request_retries=1, retry_backoff_sec=0, large_audio_threshold_mb=20)
        data = {"modelVersion": args.api_model, "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"ok": true}'}]}}]}
        response = SimpleNamespace(ok=True, status_code=200, json=lambda: data)
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "test.wav"
            audio.write_bytes(b"example")
            bundle = module.load_model(args)
            with patch.object(module.requests, "post", return_value=response) as post:
                self.assertEqual(module.request_text(bundle, audio, "original prompt", args), '{"ok": true}')
        self.assertEqual(post.call_args.args[0], "https://api.videoimagent.com/v1beta/models/gemini-3.1-pro-preview:generateContent")
        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(payload["contents"][0]["parts"][0], {"text": "original prompt"})
        self.assertEqual(bundle["response_metadata"]["model_version"], args.api_model)
        self.assertEqual(bundle["response_metadata"]["finish_reasons"], ["STOP"])

    def test_qwen_stream_preserves_utf8_and_usage(self):
        module = api_module("qwen35")
        events = [
            {"model": "qwen3.5-omni-plus", "choices": [{"delta": {"content": "中文"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " output"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"total_tokens": 10}},
        ]
        lines = [("data: " + json.dumps(event, ensure_ascii=False)).encode() for event in events] + [b"data: [DONE]"]
        response = SimpleNamespace(status_code=200, iter_lines=lambda **kwargs: iter(lines))
        metadata = {}
        self.assertEqual(module._stream_text(response, metadata), "中文 output")
        self.assertEqual(metadata["model"], "qwen3.5-omni-plus")
        self.assertEqual(metadata["usage"]["total_tokens"], 10)
        self.assertEqual(metadata["finish_reasons"], ["stop"])
        self.assertTrue(metadata["stream_done"])

    def test_qwen_payload_targets_dashscope_text_stream(self):
        module = api_module("qwen35")
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "test.wav"
            audio.write_bytes(b"example")
            payload = module.build_payload(audio, "original prompt", "qwen3.5-omni-plus")
        self.assertEqual(module.DEFAULT_BASE_URL, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(payload["model"], "qwen3.5-omni-plus")
        self.assertEqual(payload["modalities"], ["text"])
        self.assertTrue(payload["stream"])
        content = payload["messages"][0]["content"]
        self.assertTrue(content[0]["input_audio"]["data"].startswith("data:audio/wav;base64,"))
        self.assertEqual(content[1]["text"], "original prompt")


if __name__ == "__main__":
    unittest.main()
