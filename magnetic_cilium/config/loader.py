from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .schema import ConfigValidationError, SimulationConfig


@dataclass(frozen=True)
class ConfigValidationReport:
    config_path: str
    ok: bool
    errors: list[str]
    mode: str
    is_sweep: bool
    output_dir: str | None
    restart_dir: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "ok": self.ok,
            "errors": self.errors,
            "mode": self.mode,
            "is_sweep": self.is_sweep,
            "output_dir": self.output_dir,
            "restart_dir": self.restart_dir,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def load_simulation_config(path: str) -> SimulationConfig:
    raw = load_yaml_like(path)
    config = SimulationConfig.from_mapping(raw, config_path=path)
    config.ensure_valid()
    return config


def validate_config_file(path: str) -> ConfigValidationReport:
    try:
        config = load_simulation_config(path)
    except ConfigValidationError as exc:
        errors = [part.strip() for part in str(exc).split(";") if part.strip()]
        return ConfigValidationReport(
            config_path=path,
            ok=False,
            errors=errors,
            mode="unknown",
            is_sweep=False,
            output_dir=None,
            restart_dir=None,
        )
    except Exception as exc:
        return ConfigValidationReport(
            config_path=path,
            ok=False,
            errors=[str(exc)],
            mode="unknown",
            is_sweep=False,
            output_dir=None,
            restart_dir=None,
        )
    return ConfigValidationReport(
        config_path=path,
        ok=True,
        errors=[],
        mode=config.run.mode,
        is_sweep=config.sweep.enabled,
        output_dir=config.output.outdir or config.output.output_dir,
        restart_dir=config.output.restart_dir,
    )


def load_yaml_like(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        with open(path, "r", encoding="utf-8") as f:
            return _load_minimal_yaml(f.read())

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigValidationError(f"Config must be a YAML mapping: {path}")
    return _normalize_keys(data)


def _normalize_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k).replace("-", "_"): _normalize_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_keys(v) for v in value]
    return value


def _load_minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by project configs.

    This fallback keeps lightweight validation available on systems where
    PyYAML is not installed. Full solver runs should still use the documented
    FEniCSx environment from environment.yaml.
    """
    lines = []
    for raw_line in text.splitlines():
        stripped = _strip_comment(raw_line).rstrip()
        if stripped.strip():
            lines.append((len(stripped) - len(stripped.lstrip(" ")), stripped.strip()))
    parser = _MinimalYamlParser(lines)
    result = parser.parse_block(0)
    if not isinstance(result, dict):
        raise ConfigValidationError("Config must be a YAML mapping")
    return _normalize_keys(result)


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


class _MinimalYamlParser:
    def __init__(self, lines: list[tuple[int, str]]) -> None:
        self.lines = lines
        self.index = 0

    def parse_block(self, indent: int) -> Any:
        if self.index >= len(self.lines):
            return {}
        if self.lines[self.index][1].startswith("- "):
            return self.parse_list(indent)
        return self.parse_mapping(indent)

    def parse_mapping(self, indent: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while self.index < len(self.lines):
            current_indent, content = self.lines[self.index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ConfigValidationError(f"Unexpected indentation near: {content}")
            if content.startswith("- "):
                break
            if ":" not in content:
                raise ConfigValidationError(f"Expected key/value pair near: {content}")
            key, value = content.split(":", 1)
            key = key.strip().replace("-", "_")
            value = value.strip()
            self.index += 1
            if value:
                result[key] = _parse_scalar(value)
            else:
                if self.index >= len(self.lines) or self.lines[self.index][0] <= current_indent:
                    result[key] = {}
                else:
                    result[key] = self.parse_block(self.lines[self.index][0])
        return result

    def parse_list(self, indent: int) -> list[Any]:
        result: list[Any] = []
        while self.index < len(self.lines):
            current_indent, content = self.lines[self.index]
            if current_indent < indent:
                break
            if current_indent != indent or not content.startswith("- "):
                break
            item = content[2:].strip()
            self.index += 1
            if not item:
                value: Any = self.parse_block(self.lines[self.index][0]) if self.index < len(self.lines) else {}
            elif ":" in item:
                key, raw_value = item.split(":", 1)
                value = {key.strip().replace("-", "_"): _parse_scalar(raw_value.strip())}
                if self.index < len(self.lines) and self.lines[self.index][0] > current_indent:
                    nested = self.parse_mapping(self.lines[self.index][0])
                    value.update(nested)
            else:
                value = _parse_scalar(item)
            result.append(value)
        return result


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value
