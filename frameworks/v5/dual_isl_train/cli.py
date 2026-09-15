from __future__ import annotations

import argparse
import json

from .config import load_config, public_config
from .data import load_records, validate_records
from .dual_space import dual_subspace_report
from .orchestrator import DualISLOrchestrator


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dual-isl-train")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "resume", "validate-config"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    data = commands.add_parser("validate-data")
    data.add_argument("--input", required=True)
    data.add_argument("--allow-missing-audio", action="store_true")
    data.add_argument("--skip-audio-inspection", action="store_true")
    data.add_argument("--min-duration", type=float, default=1.2)
    data.add_argument("--max-duration", type=float, default=30.0)
    commands.add_parser("show-dual-space")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "show-dual-space":
        _print(dual_subspace_report())
        return
    if args.command == "validate-data":
        report = validate_records(
            load_records(args.input), require_audio=not args.allow_missing_audio,
            inspect_audio=not args.skip_audio_inspection,
            min_duration=args.min_duration, max_duration=args.max_duration,
        ).as_dict()
        _print(report)
        raise SystemExit(0 if report["ok"] else 2)
    config = load_config(args.config)
    if args.command == "validate-config":
        _print({"ok": True, "config": public_config(config)})
        return
    orchestrator = DualISLOrchestrator(config)
    _print(orchestrator.train(resume_only=args.command == "resume"))


if __name__ == "__main__":
    main()
