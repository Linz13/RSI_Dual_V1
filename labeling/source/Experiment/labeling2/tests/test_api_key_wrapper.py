from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Experiment.labeling2.run_with_local_api_key import PROXY_ENV_VARS, api_environment, load_key


class ApiEnvironmentTests(unittest.TestCase):
    def test_removes_proxies_and_keeps_provider_keys_separate(self) -> None:
        inherited = {name: "http://broken-proxy" for name in PROXY_ENV_VARS}
        with patch.dict(os.environ, inherited, clear=True):
            env = api_environment("gemini-secret", "qwen-secret")
        self.assertTrue(all(name not in env for name in PROXY_ENV_VARS))
        self.assertEqual(env["GEMINI_API_KEY"], "gemini-secret")
        self.assertEqual(env["QWEN_API_KEY"], "qwen-secret")

    def test_reads_literal_without_executing_example(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "example.py"
            source.write_text('raise RuntimeError("must not execute")\nAPI_KEY = "example-key"\n')
            with patch.dict(os.environ, {"LABELING2_GEMINI_KEY_SOURCE": str(source)}, clear=True):
                self.assertEqual(load_key("gemini"), "example-key")

    def test_provider_environment_overrides_unavailable_source(self) -> None:
        with patch.dict(os.environ, {"QWEN_API_KEY": "own-key", "LABELING2_QWEN_KEY_SOURCE": "/missing/example.py"}, clear=True):
            self.assertEqual(load_key("qwen35"), "own-key")


if __name__ == "__main__":
    unittest.main()
