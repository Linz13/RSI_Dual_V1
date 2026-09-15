#!/usr/bin/env python3
"""Run MiDasheng Attr6 inference with native audio preprocessing."""

from common import DEFAULT_MIDASHENG_MODEL_DIR, MIDASHENG_ENV_PYTHON
from run_qwen3_captioner import main


if __name__ == "__main__":
    main(
        default_backend="midasheng",
        default_model_dir=DEFAULT_MIDASHENG_MODEL_DIR,
        default_python=MIDASHENG_ENV_PYTHON,
        default_attn_backend="sdpa",
    )
