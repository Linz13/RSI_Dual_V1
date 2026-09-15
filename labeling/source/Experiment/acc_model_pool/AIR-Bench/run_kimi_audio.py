from __future__ import annotations

import os
import sys
from pathlib import Path

from airbench_utils import AirBenchSample, build_common_parser, build_prompt, run_airbench_experiment


MODEL_NAME = "kimi_audio"
ROOT = Path(os.environ.get("AUDIO_CAPTION_ROOT", Path(__file__).resolve().parents[3])).resolve()
REPO_DIR = Path(os.environ.get("LABELING2_KIMI_SOURCE_DIR", ROOT / "third_party/Kimi-Audio"))
MODEL_DIR = REPO_DIR / "Kimi-Audio-7B-Instruct"
GLM4_TOKENIZER_DIR = REPO_DIR / "glm-4-voice-tokenizer"

SAMPLING_PARAMS = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}


def maybe_set_cuda_visible_devices(device: str) -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    if device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]


def load_model(args):
    maybe_set_cuda_visible_devices(args.device)
    if args.glm4_tokenizer_path:
        os.environ["KIMI_AUDIO_GLM4_TOKENIZER_PATH"] = str(args.glm4_tokenizer_path.expanduser().resolve())
    if args.allow_online_downloads:
        os.environ["KIMI_AUDIO_LOCAL_FILES_ONLY"] = "0"
    else:
        os.environ.setdefault("KIMI_AUDIO_LOCAL_FILES_ONLY", "1")
    sys.path.insert(0, str(REPO_DIR))
    from kimia_infer.api.kimia import KimiAudio

    return KimiAudio(model_path=str(args.model_dir.expanduser().resolve()), load_detokenizer=False)


def infer_one(model, sample: AirBenchSample, args) -> str:
    messages = [
        {"role": "user", "message_type": "text", "content": build_prompt(sample, args.prompt_prefix)},
        {"role": "user", "message_type": "audio", "content": sample.audio_path},
    ]
    # KimiAudio defaults to ``max_new_tokens=-1`` (up to ~7,500 decoding
    # steps).  That is unsafe for structured-labeling prompts: if EOS is not
    # emitted, one sample can monopolize the GPU for several minutes.  Honor
    # the shared AIR-Bench/local-worker limit instead; 1,024 is ample for the
    # requested JSON fields and keeps resume runs bounded.
    sampling_params = dict(SAMPLING_PARAMS)
    sampling_params["text_repetition_penalty"] = max(
        1.0, float(getattr(args, "repetition_penalty", 1.15))
    )
    sampling_params["text_repetition_window_size"] = max(
        1, int(getattr(args, "repetition_window_size", 64))
    )
    _, text_output = model.generate(
        messages,
        **sampling_params,
        max_new_tokens=int(getattr(args, "max_new_tokens", 256)),
        output_type="text",
    )
    return str(text_output).strip()


def main() -> None:
    parser = build_common_parser(MODEL_NAME)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--glm4-tokenizer-path", type=Path, default=GLM4_TOKENIZER_DIR)
    parser.add_argument("--allow-online-downloads", action="store_true")
    args = parser.parse_args()
    run_airbench_experiment(args=args, model_name=MODEL_NAME, load_model=load_model, infer_one=infer_one)


if __name__ == "__main__":
    main()
