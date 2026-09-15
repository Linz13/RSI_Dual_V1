from __future__ import annotations

from pathlib import Path
from typing import Any


def _adapter_parameters(model: Any, adapter_name: str) -> dict[str, Any]:
    marker = f".{adapter_name}."
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if marker in name and ("lora_" in name or "modules_to_save" in name)
    }


def promote_adapter_fp32(model: Any, adapter_name: str) -> None:
    """Promote an attached adapter before loading weights into it.

    PEFT's ``load_adapter`` loads into modules created with the base-model dtype
    and only casts afterwards.  With a BF16 base this permanently rounds an FP32
    checkpoint.  Casting the destination first avoids that lossy intermediate.
    """
    parameters = _adapter_parameters(model, adapter_name)
    if not parameters:
        raise RuntimeError(f"Adapter {adapter_name!r} has no LoRA parameters")
    for parameter in parameters.values():
        parameter.data = parameter.data.float()
    non_fp32 = [name for name, value in parameters.items() if str(value.dtype) != "torch.float32"]
    if non_fp32:
        raise RuntimeError(f"Adapter {adapter_name!r} was not promoted to FP32: {non_fp32[:3]}")


def load_reference_adapter_exact(model: Any, checkpoint: str | Path, adapter_name: str = "reference") -> dict[str, Any]:
    """Attach and load a frozen adapter without a BF16 load round-trip."""
    from peft import PeftConfig
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    checkpoint = Path(checkpoint).resolve()
    if not (checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing PEFT adapter_config.json in {checkpoint}")
    config = PeftConfig.from_pretrained(str(checkpoint), local_files_only=True)
    config.inference_mode = True
    model.add_adapter(adapter_name, config)
    promote_adapter_fp32(model, adapter_name)
    on_disk = load_peft_weights(str(checkpoint), device="cpu")
    load_result = set_peft_model_state_dict(model, on_disk, adapter_name=adapter_name)
    missing = [key for key in load_result.missing_keys if f".{adapter_name}." in key]
    if missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"Exact adapter load failed: missing={missing[:3]} unexpected={load_result.unexpected_keys[:3]}"
        )
    for parameter in _adapter_parameters(model, adapter_name).values():
        parameter.requires_grad_(False)
    audit = audit_adapter_checkpoint(model, checkpoint, adapter_name)
    if not audit["exact"]:
        raise RuntimeError(f"Adapter {adapter_name!r} changed while loading: {audit}")
    return audit


def audit_adapter_checkpoint(model: Any, checkpoint: str | Path, adapter_name: str) -> dict[str, Any]:
    from peft.utils.save_and_load import get_peft_model_state_dict, load_peft_weights

    checkpoint = Path(checkpoint).resolve()
    expected = load_peft_weights(str(checkpoint), device="cpu")
    actual = get_peft_model_state_dict(model, adapter_name=adapter_name)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    max_abs_diff = 0.0
    dtype_mismatches: list[str] = []
    shape_mismatches: list[str] = []
    exact = not missing and not unexpected
    for name in sorted(set(expected) & set(actual)):
        left = expected[name].detach().cpu()
        right = actual[name].detach().cpu()
        if left.shape != right.shape:
            shape_mismatches.append(name)
            exact = False
            continue
        if str(right.dtype) != "torch.float32":
            dtype_mismatches.append(name)
            exact = False
        difference = float((left.float() - right.float()).abs().max()) if left.numel() else 0.0
        max_abs_diff = max(max_abs_diff, difference)
        exact = exact and difference == 0.0
    return {
        "exact": exact,
        "adapter_name": adapter_name,
        "checkpoint": str(checkpoint),
        "checked_tensors": len(set(expected) & set(actual)),
        "max_abs_diff": max_abs_diff,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
    }


def audit_adapter_pair(model: Any, left_name: str = "default", right_name: str = "reference") -> dict[str, Any]:
    left = _adapter_parameters(model, left_name)
    right = _adapter_parameters(model, right_name)
    normalized_left = {name.replace(f".{left_name}.", ".<adapter>."): value for name, value in left.items()}
    normalized_right = {name.replace(f".{right_name}.", ".<adapter>."): value for name, value in right.items()}
    missing = sorted(set(normalized_left) ^ set(normalized_right))
    max_abs_diff = 0.0
    exact = not missing
    for name in sorted(set(normalized_left) & set(normalized_right)):
        lvalue = normalized_left[name].detach().float()
        rvalue = normalized_right[name].detach().float()
        if lvalue.shape != rvalue.shape:
            exact = False
            continue
        difference = float((lvalue - rvalue).abs().max()) if lvalue.numel() else 0.0
        max_abs_diff = max(max_abs_diff, difference)
        exact = exact and difference == 0.0
    return {
        "exact": exact,
        "checked_tensors": len(set(normalized_left) & set(normalized_right)),
        "max_abs_diff": max_abs_diff,
        "layout_mismatches": missing,
    }

