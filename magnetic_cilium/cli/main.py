from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from magnetic_cilium.config.loader import load_simulation_config, validate_config_file
from magnetic_cilium.io.run_context import RunContext


LEGACY_MODE_BY_COMMAND = {
    "run": "full",
    "full": "full",
    "mechanics": "mechanics",
    "magnetics": "magnetics",
    "magnetics-fem": "magnetics-fem",
    "sweep": "magnetics-fem-validation",
    "validation": "validation",
    "magnetic-only": "magnetic-only",
    "interpolate": "interpolate-magnetic-results",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cilium",
        description="Magnetic cilium simulation pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate a YAML config without running solvers.")
    validate.add_argument("config")
    validate.add_argument("--json", action="store_true", help="Print validation report as JSON.")

    dry_run = sub.add_parser("dry-run", help="Validate config and show resolved run context.")
    dry_run.add_argument("config")
    dry_run.add_argument("--root-dir", default=None)

    for command, mode in LEGACY_MODE_BY_COMMAND.items():
        run_parser = sub.add_parser(command, help=f"Run mode {mode!r}.")
        run_parser.add_argument("config")
        run_parser.add_argument(
            "--engine",
            choices=["runtime", "new", "legacy"],
            default="runtime",
            help="Execution engine. runtime preserves current solver behavior; new uses magnetic_cilium.pipeline.",
        )
        run_parser.add_argument("--dry-run", action="store_true", help="Only show translated argv / plan.")
        run_parser.add_argument("runtime_args", nargs=argparse.REMAINDER, help="Extra arguments passed to runtime mode.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        report = validate_config_file(args.config)
        if args.json:
            print(report.to_json())
        else:
            status = "ok" if report.ok else "failed"
            print(f"config validation: {status}")
            print(f"config = {report.config_path}")
            print(f"mode = {report.mode}")
            print(f"is_sweep = {report.is_sweep}")
            for error in report.errors:
                print(f"error: {error}")
        return 0 if report.ok else 2

    if args.command == "dry-run":
        config = load_simulation_config(args.config)
        ctx = RunContext.create(config, root_dir=args.root_dir, create_dirs=False)
        payload = {
            "config": config.to_dict(),
            "run_context": ctx.to_dict(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    _consume_wrapper_options(args)

    mode = LEGACY_MODE_BY_COMMAND[args.command]
    dry_run_requested = bool(args.dry_run) or "--dry-run" in args.runtime_args
    runtime_argv = ["main.py", "--mode", mode, "--config", args.config]
    runtime_argv.extend(_clean_remainder(args.runtime_args))

    if dry_run_requested:
        report = validate_config_file(args.config)
        print(report.to_json())
        if args.engine == "new":
            _print_architecture_plan(args.command, args.config)
        print("runtime argv:")
        print(" ".join(runtime_argv))
        return 0 if report.ok else 2

    if args.engine == "new":
        return _run_architecture_engine(args.command, args.config)

    return _run_runtime_main(runtime_argv)


def _clean_remainder(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        values = values[1:]
    return [value for value in values if value != "--dry-run"]


def _consume_wrapper_options(args) -> None:
    values = list(args.runtime_args)
    cleaned: list[str] = []
    i = 0
    while i < len(values):
        value = values[i]
        if value == "--dry-run":
            args.dry_run = True
            i += 1
            continue
        if value == "--engine" and i + 1 < len(values):
            args.engine = values[i + 1]
            i += 2
            continue
        if value.startswith("--engine="):
            args.engine = value.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(value)
        i += 1
    if args.engine == "legacy":
        args.engine = "runtime"
    if args.engine not in {"runtime", "new"}:
        raise SystemExit(f"invalid --engine value: {args.engine!r}")
    args.runtime_args = cleaned


def _run_runtime_main(runtime_argv: list[str]) -> int:
    _ensure_runtime_root()
    previous_argv = sys.argv[:]
    try:
        sys.argv = runtime_argv
        from magnetic_cilium.cli.runtime import main as runtime_main

        runtime_main()
    finally:
        sys.argv = previous_argv
    return 0


def _print_architecture_plan(command: str, config_path: str) -> None:
    print("architecture plan:")
    if command in {"run", "full", "mechanics"}:
        from magnetic_cilium.pipeline.full import run_full_pipeline

        plan = run_full_pipeline(config_path, dry_run=True)
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    elif command in {"magnetics", "magnetic-only"}:
        from magnetic_cilium.pipeline.magnetic_only import run_magnetic_only_pipeline

        plan = run_magnetic_only_pipeline(config_path, dry_run=True)
        print(json.dumps(plan, indent=2, sort_keys=True))
    elif command == "sweep":
        from magnetic_cilium.pipeline.sweep import build_sweep_plan

        plan = build_sweep_plan(config_path)
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(json.dumps({"supported": False, "command": command}, indent=2, sort_keys=True))


def _run_architecture_engine(command: str, config_path: str) -> int:
    if command in {"run", "full", "mechanics"}:
        from magnetic_cilium.pipeline.full import run_full_pipeline

        result = run_full_pipeline(config_path)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if command in {"magnetics", "magnetic-only"}:
        from magnetic_cilium.pipeline.magnetic_only import run_magnetic_only_pipeline

        result = run_magnetic_only_pipeline(config_path)
        payload = None if result is None else result.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if command == "sweep":
        from magnetic_cilium.pipeline.sweep import run_magnetics_fem_validation_from_config

        run_magnetics_fem_validation_from_config(config_path)
        return 0
    raise RuntimeError(f"Architecture engine does not support command: {command}")


def _ensure_runtime_root() -> None:
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


if __name__ == "__main__":
    raise SystemExit(main())
