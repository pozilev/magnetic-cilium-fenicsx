from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

try:
    from magnetic_cilium_pipeline.config.loader import load_simulation_config, validate_config_file
    from magnetic_cilium_pipeline.io.run_context import RunContext
except ModuleNotFoundError:
    from config.loader import load_simulation_config, validate_config_file  # type: ignore
    from magnetic_cilium.io.run_context import RunContext  # type: ignore


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
        description="Subcommand wrapper around the magnetic cilium simulation pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate a YAML config without running solvers.")
    validate.add_argument("config")
    validate.add_argument("--json", action="store_true", help="Print validation report as JSON.")

    dry_run = sub.add_parser("dry-run", help="Validate config and show resolved run context.")
    dry_run.add_argument("config")
    dry_run.add_argument("--root-dir", default=None)

    for command, mode in LEGACY_MODE_BY_COMMAND.items():
        run_parser = sub.add_parser(command, help=f"Run legacy mode {mode!r}.")
        run_parser.add_argument("config")
        run_parser.add_argument("--dry-run", action="store_true", help="Only show translated legacy argv.")
        run_parser.add_argument("legacy_args", nargs=argparse.REMAINDER, help="Extra arguments passed to main.py.")

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

    mode = LEGACY_MODE_BY_COMMAND[args.command]
    dry_run_requested = bool(args.dry_run) or "--dry-run" in args.legacy_args
    legacy_argv = ["main.py", "--mode", mode, "--config", args.config]
    legacy_argv.extend(_clean_remainder(args.legacy_args))

    if dry_run_requested:
        report = validate_config_file(args.config)
        print(report.to_json())
        print("legacy argv:")
        print(" ".join(legacy_argv))
        return 0 if report.ok else 2

    return _run_legacy_main(legacy_argv)


def _clean_remainder(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        values = values[1:]
    return [value for value in values if value != "--dry-run"]


def _run_legacy_main(legacy_argv: list[str]) -> int:
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)

    previous_argv = sys.argv[:]
    try:
        sys.argv = legacy_argv
        from main import main as legacy_main

        legacy_main()
    finally:
        sys.argv = previous_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
