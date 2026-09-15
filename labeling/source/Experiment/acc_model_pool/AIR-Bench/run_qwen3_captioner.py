from __future__ import annotations

import os
from pathlib import Path

from airbench_utils import AirBenchSample, build_common_parser, build_prompt, run_airbench_experiment


MODEL_NAME = "qwen3_captioner"
MODEL_DIR = Path("/F00120250029/lixiang_share/linzhiyu_share/Qwen3-Captioner/Qwen3-Omni-30B-A3B-Captioner")


def choose_torch_dtype(torch):
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def patch_initializer_range(config) -> None:
    if not hasattr(config, "initializer_range"):
        config.initializer_range = 0.02
    for sub_name in ["thinker_config", "talker_config", "code2wav_config"]:
        if hasattr(config, sub_name):
            sub_cfg = getattr(config, sub_name)
            if sub_cfg is not None and not hasattr(sub_cfg, "initializer_range"):
                sub_cfg.initializer_range = 0.02


def resolve_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return model.device


def load_model(args):
    import torch
    from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    model_path = args.model_dir.expanduser().resolve()
    dtype = choose_torch_dtype(torch)
    device_map = "auto" if args.device == "auto" else {"": args.device}
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    patch_initializer_range(config)
    load_kwargs = {}
    max_memory_gib = int(os.environ.get("QWEN3_MAX_MEMORY_GIB", "0"))
    if device_map == "auto" and max_memory_gib > 0:
        load_kwargs["max_memory"] = {
            index: f"{max_memory_gib}GiB" for index in range(torch.cuda.device_count())
        } | {"cpu": os.environ.get("QWEN3_CPU_MAX_MEMORY", "128GiB")}
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        config=config,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=args.attn_backend,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
        **load_kwargs,
    ).eval()
    if os.environ.get("QWEN3_REQUIRE_GPU_ONLY", "0") == "1":
        device_map_values = {str(value) for value in getattr(model, "hf_device_map", {}).values()}
        forbidden = {"cpu", "disk", "meta"} & device_map_values
        if forbidden:
            raise RuntimeError(
                f"Qwen3 distributed load used forbidden offload devices: {sorted(forbidden)}; "
                f"device_map={getattr(model, 'hf_device_map', {})}"
            )
        if torch.cuda.device_count() >= 2 and not {"0", "1"}.issubset(device_map_values):
            raise RuntimeError(
                f"Qwen3 did not use both requested GPUs; "
                f"device_map={getattr(model, 'hf_device_map', {})}"
            )
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    return {"model": model, "processor": processor}


def infer_one(bundle, sample: AirBenchSample, args) -> str:
    import torch
    from qwen_omni_utils import process_mm_info

    model = bundle["model"]
    processor = bundle["processor"]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": sample.audio_path},
                {"type": "text", "text": build_prompt(sample, args.prompt_prefix)},
            ],
        }
    ]
    rendered = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=rendered,
        audio=audios,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    input_device = resolve_input_device(model)
    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to(input_device)
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(dtype=model.dtype)

    with torch.inference_mode():
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": getattr(args, "repetition_penalty", 1.0),
            "no_repeat_ngram_size": getattr(args, "no_repeat_ngram_size", 0),
            "do_sample": getattr(args, "do_sample", False),
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs.update(
                temperature=getattr(args, "temperature", 0.7),
                top_p=getattr(args, "top_p", 0.9),
            )
        generation_output = model.generate(**inputs, **generation_kwargs)
    text_ids = generation_output[0] if isinstance(generation_output, tuple) else generation_output
    seq = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
    generated_ids = seq[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0].strip() if decoded else ""


def main() -> None:
    parser = build_common_parser(MODEL_NAME)
    parser.set_defaults(device="auto")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--attn-backend", default="flash_attention_2", choices=["flash_attention_2", "sdpa"])
    args = parser.parse_args()
    run_airbench_experiment(args=args, model_name=MODEL_NAME, load_model=load_model, infer_one=infer_one)


if __name__ == "__main__":
    main()
