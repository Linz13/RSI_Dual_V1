from __future__ import annotations


def extract_generated_ids(sequences, prompt_input_ids):
    if sequences.ndim != 2 or prompt_input_ids.ndim != 2:
        return sequences
    prompt_length = prompt_input_ids.shape[1]
    if sequences.shape[1] >= prompt_length:
        import torch

        if torch.equal(sequences[:, :prompt_length], prompt_input_ids):
            return sequences[:, prompt_length:]
    return sequences


def completion_token_span(prompt_ids, full_ids, truncated_ids=None) -> tuple[int, int]:
    def flatten(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1).tolist()
        elif value and isinstance(value[0], list):
            value = value[0]
        return [int(item) for item in value]

    prompt = flatten(prompt_ids)
    full = flatten(full_ids)
    if full[:len(prompt)] != prompt:
        raise ValueError("Processor completion does not preserve the generation prompt prefix")
    cutoff = len(full)
    if truncated_ids is not None:
        truncated = flatten(truncated_ids)
        cutoff = 0
        for left, right in zip(full, truncated):
            if left != right:
                break
            cutoff += 1
        if cutoff < len(prompt):
            raise ValueError("Completion cutoff diverges inside the processor prompt")
    return len(prompt), cutoff
