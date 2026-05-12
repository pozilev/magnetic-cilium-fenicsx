from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

MECHANICS_RESTART_VERSION = "1.0"
REQUIRED_RESTART_FILES = ("mechanics_restart.npz", "params.json")
OPTIONAL_RESTART_FILES = ("mechanics_result.json", "restart_manifest.json")


@dataclass(frozen=True)
class RestartManifest:
    version: str
    restart_type: str
    restart_dir: str
    required_files: tuple[str, ...] = REQUIRED_RESTART_FILES
    optional_files: tuple[str, ...] = OPTIONAL_RESTART_FILES
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RestartStatus:
    restart_dir: str
    ok: bool
    missing_required_files: tuple[str, ...]
    present_optional_files: tuple[str, ...]
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def restart_manifest_path(restart_dir: str) -> str:
    return os.path.join(restart_dir, "restart_manifest.json")


def inspect_restart_dir(restart_dir: str) -> RestartStatus:
    missing = tuple(name for name in REQUIRED_RESTART_FILES if not os.path.exists(os.path.join(restart_dir, name)))
    present_optional = tuple(name for name in OPTIONAL_RESTART_FILES if os.path.exists(os.path.join(restart_dir, name)))
    return RestartStatus(
        restart_dir=restart_dir,
        ok=not missing,
        missing_required_files=missing,
        present_optional_files=present_optional,
        manifest_path=restart_manifest_path(restart_dir),
    )


def ensure_restart_dir(restart_dir: str) -> RestartStatus:
    status = inspect_restart_dir(restart_dir)
    if not status.ok:
        missing = ", ".join(status.missing_required_files)
        raise FileNotFoundError(f"Invalid mechanics restart directory {restart_dir!r}; missing: {missing}")
    return status


def write_restart_manifest(
    restart_dir: str,
    *,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    os.makedirs(restart_dir, exist_ok=True)
    manifest = RestartManifest(
        version=MECHANICS_RESTART_VERSION,
        restart_type="mechanics",
        restart_dir=restart_dir,
        run_id=run_id,
        metadata=metadata or {},
    )
    path = restart_manifest_path(restart_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
    return path


def read_restart_manifest(restart_dir: str) -> RestartManifest | None:
    path = restart_manifest_path(restart_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RestartManifest(
        version=str(data.get("version", "")),
        restart_type=str(data.get("restart_type", "")),
        restart_dir=str(data.get("restart_dir", restart_dir)),
        required_files=tuple(data.get("required_files", REQUIRED_RESTART_FILES)),
        optional_files=tuple(data.get("optional_files", OPTIONAL_RESTART_FILES)),
        run_id=data.get("run_id"),
        metadata=dict(data.get("metadata") or {}),
    )


def save_mechanics_restart(*args, **kwargs):
    from magnetic_cilium.pipeline.execution import save_mechanics_restart as _save_mechanics_restart

    restart_npz = _save_mechanics_restart(*args, **kwargs)
    write_restart_manifest(os.path.dirname(restart_npz))
    return restart_npz


def load_mechanics_restart(*args, **kwargs):
    if args:
        ensure_restart_dir(str(args[0]))
    from magnetic_cilium.pipeline.execution import load_mechanics_restart as _load_mechanics_restart

    return _load_mechanics_restart(*args, **kwargs)
