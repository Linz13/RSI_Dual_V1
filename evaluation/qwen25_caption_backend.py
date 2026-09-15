"""Qwen2.5-Omni thinker inference shared by the three Captioner benchmarks."""
from __future__ import annotations

from pathlib import Path
from model_adapter_utils import describe_adapter, load_peft_adapter


class Qwen25Captioner:
    def __init__(self, model_path, adapter_dir=None, attn_backend="flash_attention_2", **_):
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        self.torch = torch
        self.process_mm_info = process_mm_info
        self.model_path = str(Path(model_path).resolve())
        self.attn_backend = attn_backend or "flash_attention_2"
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_path, dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation=self.attn_backend, low_cpu_mem_usage=True,
            local_files_only=True, trust_remote_code=True,
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=True,
        )
        self.processor.tokenizer.padding_side = "left"
        self.adapter = describe_adapter(adapter_dir, self.model_path)
        if self.adapter is not None:
            self.model, self.adapter = load_peft_adapter(self.model, self.adapter)
        self.model.requires_grad_(False)
        self.model.eval()

    def generate_batch(self, audio_paths, prompts, max_new_tokens):
        if len(audio_paths) != len(prompts):
            raise ValueError("Audio/prompt batch lengths differ")
        if not prompts:
            return []
        conversations = [[{"role": "user", "content": [
            {"type": "audio", "audio": str(Path(audio).resolve())},
            {"type": "text", "text": prompt},
        ]}] for audio, prompt in zip(audio_paths, prompts)]
        rendered = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False,
        )
        audios, _, _ = self.process_mm_info(conversations, use_audio_in_video=False)
        inputs = self.processor(text=rendered, audio=audios, return_tensors="pt",
                                padding=True, use_audio_in_video=False)
        parameter = next(self.model.parameters())
        inputs = {key: value.to(parameter.device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        for key in ("input_features", "pixel_values", "pixel_values_videos"):
            if key in inputs:
                inputs[key] = inputs[key].to(dtype=parameter.dtype)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                return_dict_in_generate=True,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        sequences = output.sequences if hasattr(output, "sequences") else output
        generated = sequences[:, inputs["input_ids"].shape[1]:]
        predictions = [text.strip() for text in self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )]
        if len(predictions) != len(prompts):
            raise RuntimeError("Qwen2.5 output batch length differs from input")
        return predictions


def load_bundle(args):
    captioner = Qwen25Captioner(args.model_dir, args.adapter_dir, args.attn_backend)
    return {"backend": "qwen25", "captioner": captioner,
            "adapter": captioner.adapter, "torch": captioner.torch}


def infer_one(bundle, audio_path, prompt, args):
    return bundle["captioner"].generate_batch([audio_path], [prompt], args.max_new_tokens)[0]
