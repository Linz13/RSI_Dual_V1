"""Compatibility exports for the existing AIR-Bench model backends.

Do not run this file directly.  The local ``run_*.py`` wrappers import the
original backend implementations and bind these functions in their place.
"""
from caption_bench_utils import CaptionBenchSample as AirBenchSample
from caption_bench_utils import build_common_parser, build_prompt, run_airbench_experiment

__all__ = ["AirBenchSample", "build_common_parser", "build_prompt", "run_airbench_experiment"]
