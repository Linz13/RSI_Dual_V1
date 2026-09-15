"""Step-Audio-R1.1 direct PyTorch/Transformers inference, no vLLM imports.

Uses the pinned checkpoint's original model modules. Audio preprocessing and
placeholder lengths follow the pinned StepAudio service processor. Qwen2's
native KV cache avoids the checkpoint wrapper's uncached generation path.
"""
from pathlib import Path
import importlib
import json
import time
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper


@contextmanager
def checkpoint_loader_compat(model_dir):
    """Accept valid metadata-free safetensors only for this indexed checkpoint.

    Transformers 4.46.3 assumes f.metadata() is a dict. The original StepAudio
    shards legitimately omit that optional field. Keep the same safetensors
    tensor loader, without changing weight files or installed packages.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import modeling_utils
    root = Path(model_dir).resolve()
    index = json.loads((root/'model.safetensors.index.json').read_text())
    paths = {(root/name).resolve() for name in index['weight_map'].values()}
    metadata_free = set()
    for path in paths:
        if path.parent != root or path.suffix != '.safetensors':
            raise ValueError('Unexpected shard path in StepAudio checkpoint index')
        with safe_open(str(path),framework='pt') as f:
            if f.metadata() is None:
                metadata_free.add(path)
    original = modeling_utils.load_state_dict

    def load(checkpoint_file, *args, **kwargs):
        if Path(checkpoint_file).resolve() in metadata_free:
            print('Loading metadata-free safetensors:',Path(checkpoint_file).name,flush=True)
            # Exactly the tensor loading operation used by Transformers 4.46.3
            # after its metadata check; its safetensors branch also loads on CPU.
            return load_file(str(checkpoint_file))
        return original(checkpoint_file,*args,**kwargs)

    with patch.object(modeling_utils,'load_state_dict',load):
        yield


def load_config(model_dir):
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    # The checkpoint stores separate embedding and lm_head tensors. The outer
    # custom config otherwise inherits PretrainedConfig's tied-weight default.
    config.tie_word_embeddings = False
    config.text_config.tie_word_embeddings = False
    config.text_config._attn_implementation = 'sdpa'
    return config


def load_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    expected = {'<audio_start>':151688, '<audio_end>':151689, '<audio_patch>':151690, '<|EOT|>':151665}
    for token, token_id in expected.items():
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f'Unexpected special token mapping: {token}')
    return tokenizer


def audio_token_count(mel_frames):
    # Same length rule as pinned vLLM mm_step_audio.Step1fProcessor.
    encoder_length = (mel_frames + 1) // 2 // 2
    return (encoder_length - 1) // 2 + 1


def prepare_audio(path, model_module, sample_rate=16000):
    import librosa
    audio, original_rate = sf.read(str(path), dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Empty or non-finite audio')
    if original_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=original_rate, target_sr=sample_rate)
    chunk_samples = int(29.9 * sample_rate)
    chunks = []
    for offset in range(0, len(audio), chunk_samples):
        chunk = torch.from_numpy(np.array(audio[offset:offset+chunk_samples], dtype=np.float32, copy=True))
        # CPU feature extraction keeps the pinned model's mel filter on CPU.
        mel = model_module.log_mel_spectrogram(chunk, n_mels=128, padding=479)
        if (mel.shape[-1] + 1) // 2 > 1500:
            raise ValueError('Audio chunk exceeds encoder positional context')
        chunks.append(mel)
    return chunks


def prepare_input(tokenizer, prompt, mels, template):
    content = [{'type':'text', 'text':prompt}] + [{'type':'audio'} for _ in mels]
    text = tokenizer.apply_chat_template([{'role':'user','content':content}],
                                         chat_template=template, add_generation_prompt=True, tokenize=False)
    counts = [audio_token_count(mel.shape[-1]) for mel in mels]
    parts = text.split('<audio_patch>')
    if len(parts) != len(counts)+1:
        raise ValueError('Audio placeholder count mismatch')
    expanded = parts[0]
    for count, rest in zip(counts, parts[1:]):
        expanded += '<audio_start>' + '<audio_patch>'*count + '<audio_end>' + rest
    ids = tokenizer(expanded, add_special_tokens=False, return_tensors='pt').input_ids
    if int((ids == 151690).sum()) != sum(counts):
        raise ValueError('Audio patch tokenization mismatch')
    return ids, counts, expanded


@torch.inference_mode()
def audio_embeddings(model, mels):
    device = model.model.embed_tokens.weight.device
    dtype = model.encoder.conv1.weight.dtype
    result = []
    for mel in mels:
        frames = mel.shape[-1]
        encoded, lengths = model.encoder(mel[None].to(device=device, dtype=dtype),
                                         torch.tensor([frames], dtype=torch.int32, device=device))
        adapted = model.adapter(encoded)
        count = int(((lengths[0]-1)//2+1).item())
        if count != audio_token_count(frames):
            raise ValueError('Audio encoder length does not match placeholder count')
        result.append(adapted[0,:count])
    return result


def merge_embeddings(model, ids, features):
    embeddings = model.model.embed_tokens(ids)
    audio = torch.cat(features, dim=0)
    if int((ids==151690).sum()) != audio.shape[0]:
        raise ValueError('Encoded audio and placeholder lengths disagree')
    embeddings[ids==151690] = audio.to(embeddings.dtype)
    return embeddings


@torch.inference_mode()
def decode(model, ids, features, max_new_tokens=1024, temperature=0.7, top_p=0.9,
           stop_ids=(151643,151645,151665), progress=None):
    if max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError('Invalid decoding parameters')
    embeddings = merge_embeddings(model, ids, features)
    mask = torch.ones_like(ids)
    out = model.model(inputs_embeds=embeddings, attention_mask=mask, use_cache=True, return_dict=True)
    generated = []
    reason = 'max_new_tokens'
    temperature_warper = TemperatureLogitsWarper(temperature) if temperature > 0 else None
    top_p_warper = TopPLogitsWarper(top_p) if top_p < 1 else None
    for index in range(max_new_tokens):
        scores = model.lm_head(out.last_hidden_state[:,-1,:]).float()
        if temperature_warper is None:
            next_token = scores.argmax(dim=-1)
        else:
            scores = temperature_warper(None, scores)
            if top_p_warper is not None:
                scores = top_p_warper(None, scores)
            next_token = torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1).squeeze(-1)
        token_id = int(next_token.item())
        generated.append(token_id)
        if progress and (index==0 or (index+1)%32==0):
            progress(index+1)
        if token_id in stop_ids:
            reason = 'stop_token'
            break
        if index+1 == max_new_tokens:
            break
        if out.past_key_values is None:
            raise RuntimeError('Qwen2 did not return its KV cache')
        mask = torch.cat([mask, torch.ones((1,1), dtype=mask.dtype, device=mask.device)], dim=1)
        out = model.model(input_ids=next_token[:,None], attention_mask=mask,
                          past_key_values=out.past_key_values, use_cache=True, return_dict=True)
    return generated, reason


class StepAudioHF:
    def __init__(self, model_dir, template, device='cuda:0'):
        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Real StepAudio inference requires an available CUDA device')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one physical GPU using CUDA_VISIBLE_DEVICES')
        self.device = torch.device(device)
        self.tokenizer = load_tokenizer(model_dir)
        self.template = Path(template).read_text()
        config = load_config(model_dir)
        print('Loading original StepAudio checkpoint onto', device, flush=True)
        with checkpoint_loader_compat(model_dir):
            self.model, info = AutoModelForCausalLM.from_pretrained(
                str(model_dir), config=config, trust_remote_code=True, local_files_only=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={'':device},
                output_loading_info=True)
        self.loading_info = info
        if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
            raise RuntimeError('Checkpoint loading mismatch: ' + json.dumps(info, default=str))
        self.model.eval()
        self.model_module = importlib.import_module(self.model.__class__.__module__)
        print('Checkpoint loaded; CUDA allocated GiB:', torch.cuda.memory_allocated()/2**30, flush=True)

    def infer(self, audio_path, prompt, max_new_tokens=1024, temperature=0.7, top_p=0.9, seed=0):
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        mels = prepare_audio(audio_path, self.model_module)
        ids, counts, rendered = prepare_input(self.tokenizer, prompt, mels, self.template)
        if ids.shape[-1] + max_new_tokens > 4096:
            raise ValueError('Input plus requested output exceeds smoke-test context 4096; audio is not truncated')
        ids = ids.to(self.device)
        features = audio_embeddings(self.model, mels)
        tokens, reason = decode(self.model, ids, features, max_new_tokens, temperature, top_p,
                                progress=lambda n: print('Generated tokens:', n, flush=True))
        # Match the service's termination strings: do not include terminal EOT/EOS.
        text_tokens = tokens[:-1] if reason=='stop_token' else tokens
        text = self.tokenizer.decode(text_tokens, skip_special_tokens=False)
        return {'response_text':text, 'generated_token_ids':tokens, 'finish_reason':reason,
                'input_tokens':ids.shape[-1], 'audio_tokens_per_chunk':counts,
                'rendered_prompt':rendered, 'duration_sec':round(time.monotonic()-started,3),
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
