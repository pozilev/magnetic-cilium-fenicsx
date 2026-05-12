from __future__ import annotations

try:
    from magnetic_cilium_pipeline.cli.main import main
except ModuleNotFoundError:
    from cli.main import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())
