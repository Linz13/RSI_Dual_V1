"""labeling2 adapter for the original StepAudio weights via Transformers."""
import os
from pathlib import Path
from airbench_utils import build_prompt
from Experiment.labeling2.stepaudio_hf import StepAudioHF


def load_model(args):
    root = Path(os.environ['AUDIO_CAPTION_ROOT'])
    return StepAudioHF(args.model_dir, root/'lzy/Step-Audio-R1/chat_template_step_audio_r1_1.jinja',
                       device=args.device)


def infer_one(model, sample, args):
    return model.infer(sample.audio_path, build_prompt(sample,args.prompt_prefix),
                       max_new_tokens=args.max_new_tokens,temperature=args.temperature,
                       top_p=args.top_p)['response_text']
