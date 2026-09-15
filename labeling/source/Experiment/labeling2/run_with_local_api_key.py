from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROVIDERS = {
    "gemini": ("GEMINI_API_KEY", "LABELING2_GEMINI_KEY_SOURCE", "/data/L202500147/api/gemini_audio.py"),
    "qwen35": ("QWEN_API_KEY", "LABELING2_QWEN_KEY_SOURCE", "/data/L202500147/api/api_demo/api_demo/qwen_omni_wav_to_text.py"),
}
PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")


def load_key(model: str = "gemini") -> str:
    """Read each provider separately, without executing its example script."""
    env_name, source_env, default_path = PROVIDERS[model]
    if os.environ.get(env_name, "").strip():
        return os.environ[env_name].strip()
    path = Path(os.environ.get(source_env, default_path))
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "API_KEY" for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and node.value.value.strip():
                return node.value.value.strip()
    raise ValueError(f"No literal API_KEY in {path}; set {env_name} instead")


def api_environment(gemini_key: str, qwen_key: str) -> dict[str, str]:
    """Use distinct provider credentials and bypass legacy incompatible proxies."""
    env = dict(os.environ)
    for name in PROXY_ENV_VARS:
        env.pop(name, None)
    env["GEMINI_API_KEY"] = gemini_key
    env["QWEN_API_KEY"] = qwen_key
    return env


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_with_local_api_key.py COMMAND [ARG ...]")
    env = api_environment(load_key("gemini"), load_key("qwen35"))
    result = subprocess.run(sys.argv[1:], env=env, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
