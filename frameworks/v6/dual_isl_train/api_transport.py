"""Deployed audio API transport, with strict model and completion checks."""
import os
from pathlib import Path
from types import SimpleNamespace

def request_with_metadata(self, model, path, prompt):
    module = self.pipeline.api_module(model)
    api = self.backend_config["api"]
    backend = self.backend_config.get("backends", {}).get(model, {})
    args = SimpleNamespace(
        api_key=os.environ.get(backend.get("api_key_env", "GEMINI_API_KEY" if model == "gemini" else "QWEN_API_KEY")),
        base_url=api["gemini_base_url" if model == "gemini" else "qwen_base_url"],
        api_model=api["gemini_model" if model == "gemini" else "qwen_model"],
        timeout=int(api.get("timeout_sec", 300)), request_retries=1,
        retry_backoff_sec=0, large_audio_threshold_mb=20.0)
    bundle = module.load_model(args)
    raw = module.request_text(bundle, Path(path), prompt, args)
    metadata = bundle.get("response_metadata", {})
    returned = metadata.get("model_version" if model == "gemini" else "model")
    finishes = metadata.get("finish_reasons", [])
    if returned != args.api_model or not finishes or any(v.casefold() != "stop" for v in finishes):
        raise ValueError("wrong model or incomplete evaluator response")
    if model == "qwen35" and not metadata.get("stream_done"):
        raise ValueError("incomplete evaluator stream")
    return raw, metadata
