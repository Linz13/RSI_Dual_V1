"""One shared audio encoding and genuinely batched MiDasheng decoding."""
from __future__ import annotations


def generate_batch(worker, audio_path, prompt, seeds, temperature=None):
    if not seeds:
        return []
    if len(seeds) == 1:
        return worker._serial_generate_batch(audio_path, prompt, seeds, temperature=temperature)
    from transformers import LogitsProcessorList
    t = worker.torch
    cfg = worker.cfg.get("generation", {})
    temperature = float(cfg.get("temperature", 1.0) if temperature is None else temperature)
    worker.model.set_adapter("default")
    worker.model.eval()
    worker.freeze_batchnorm()
    with t.no_grad():
        values = worker._inputs(audio_path, prompt)
        model = worker.model.get_base_model()
        prompt_ids = values.pop("input_ids")
        embeds = model._prepare_inputs_embeds(input_ids=prompt_ids,
            input_values=values.pop("input_values", None), inputs_embeds=values.pop("inputs_embeds", None),
            audio_length=values.pop("audio_length", None))
        embeds = embeds.expand(len(seeds), -1, -1).contiguous()
        values = {k: v.expand(len(seeds), *v.shape[1:]).contiguous()
                  if hasattr(v, "shape") and v.ndim > 0 and v.shape[0] == 1 else v for k, v in values.items()}
        generators = [t.Generator(device=worker.device).manual_seed(int(seed)) for seed in seeds]
        ids, logps = [[] for _ in seeds], [[] for _ in seeds]

        class SampleAndForce:
            def __call__(self, input_ids, scores):
                if scores.shape[0] != len(seeds):
                    raise RuntimeError("MiDasheng batch dimension mismatch")
                forced = t.full_like(scores, -t.inf)
                for i, generator in enumerate(generators):
                    lp = worker._warped_logprobs(input_ids[i:i+1], scores[i:i+1], temperature)
                    token = (t.multinomial(lp.exp(), 1, generator=generator)[0, 0]
                             if temperature > 0 else lp.argmax(dim=-1)[0])
                    ids[i].append(token.detach())
                    logps[i].append(lp[0, token].detach())
                    forced[i, token] = 0
                return forced

        output = model.decoder.generate(inputs_embeds=embeds, generation_config=model.generation_config,
            **values, max_new_tokens=int(cfg.get("max_new_tokens", 384)), do_sample=False, use_cache=True,
            logits_processor=LogitsProcessorList([SampleAndForce()]), eos_token_id=worker.eos_token_ids,
            pad_token_id=worker.pad_token_id, return_dict_in_generate=True)
    results = []
    for i in range(len(seeds)):
        sampled = t.stack(ids[i]).cpu().tolist()
        eos = [j for j, token in enumerate(sampled) if token in worker.eos_token_ids]
        count = eos[0] + 1 if eos else len(sampled)
        sampled = sampled[:count]
        # Decoder inputs_embeds generation emits completions only; tolerate prompt-prefixed backends explicitly.
        sequence = output.sequences[i]
        expected = t.tensor(sampled, device=sequence.device, dtype=sequence.dtype)
        if t.equal(sequence[:count], expected):
            mode = "completion_only"
        elif t.equal(sequence[:prompt_ids.shape[1]], prompt_ids[0]) and t.equal(sequence[prompt_ids.shape[1]:prompt_ids.shape[1]+count], expected):
            mode = "prompt_plus_completion"
        else:
            raise RuntimeError("MiDasheng batch output differs from sampled tokens")
        results.append({"raw_text": worker.tokenizer.decode(sampled, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip(),
            "sampled_token_ids": sampled, "old_token_logprobs": t.stack(logps[i][:count]).float().cpu().tolist(),
            "finish_reason": "eos" if eos else "length", "terminal_token_id": sampled[-1],
            "terminated_by_eos": bool(eos), "post_eos_token_count": 0, "generated_token_count": count,
            "eos_token_id": worker.eos_token_id, "sequence_mode": mode, "generation_batch_size": len(seeds)})
    return results
