from __future__ import annotations

from importlib import import_module
from typing import Any


def legacy_module(name: str) -> Any:
    """Import a legacy module whether the project is run as a package or script."""
    candidates = [f"magnetic_cilium_pipeline.{name}", name]
    last_error: ModuleNotFoundError | None = None
    for candidate in candidates:
        try:
            return import_module(candidate)
        except ModuleNotFoundError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ModuleNotFoundError(name)


def legacy_attr(module_name: str, attr_name: str) -> Any:
    return getattr(legacy_module(module_name), attr_name)
