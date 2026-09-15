from __future__ import annotations

import sys
import os
from pathlib import Path

from airbench_utils import AirBenchSample, build_common_parser, build_prompt, run_airbench_experiment


MODEL_NAME = "step_audio_r1_1"
ROOT = Path(os.environ.get("AUDIO_CAPTION_ROOT", Path(__file__).resolve().parents[3])).resolve()
REPO_DIR = Path(os.environ.get("LABELING2_STEP_SOURCE_DIR", ROOT / "lzy/Step-Audio-R1"))


def load_model(args):
    env_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{env_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    sys.path.insert(0, str(REPO_DIR))
    from stepaudior1vllm import StepAudioR1

    return StepAudioR1(args.api_url, args.step_model_name)


def infer_one(model, sample: AirBenchSample, args) -> str:
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append(
        {
            "role": "human",
            "content": [
                {"type": "text", "text": build_prompt(sample, args.prompt_prefix)},
                {"type": "audio", "audio": sample.audio_path},
            ],
        }
    )
    chunks: list[str] = []
    try:
        for _, text, _ in model.stream(
            messages,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_token_ids=[151665],
        ):
            if text:
                chunks.append(text)
    except Exception as exc:
        if "Connection refused" in repr(exc) or "Failed to establish a new connection" in repr(exc):
            raise RuntimeError(
                "Step-Audio-R1.1 server is not reachable. Start it first with "
                "`conda run -p /F00120250029/lixiang_share/Audio_caption_share/miniconda3/envs/stepaudio "
                "python /F00120250029/lixiang_share/Audio_caption_share/lzy/Step-Audio-R1/serve_step_audio_r1_1.py`, "
                f"then rerun this script. api_url={args.api_url}"
            ) from exc
        raise
    return "".join(chunks).strip()


def main() -> None:
    parser = build_common_parser(MODEL_NAME)
    parser.add_argument("--api-url", default="http://127.0.0.1:9999/v1/chat/completions")
    parser.add_argument("--step-model-name", default="Step-Audio-R1.1")
    parser.add_argument("--system", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.set_defaults(max_new_tokens=32)
    args = parser.parse_args()
    run_airbench_experiment(args=args, model_name=MODEL_NAME, load_model=load_model, infer_one=infer_one)


if __name__ == "__main__":
    main()
