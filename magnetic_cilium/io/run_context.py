from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{utc_timestamp()}_{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class RunContext:
    run_id: str
    experiment_id: str
    root_dir: str
    run_dir: str
    logs_dir: str
    plots_dir: str
    tables_dir: str
    fields_dir: str
    resolved_config_path: str
    result_json_path: str

    @classmethod
    def create(
        cls,
        config: Any,
        *,
        root_dir: str | None = None,
        create_dirs: bool = True,
        run_id: str | None = None,
    ) -> "RunContext":
        experiment_id = config.output.experiment_id or "debug"
        base_root = root_dir or config.output.outdir or config.output.output_dir or "runs"
        actual_run_id = run_id or make_run_id(config.run.mode.replace("-", "_"))
        run_dir = os.path.join(base_root, experiment_id, actual_run_id)
        ctx = cls(
            run_id=actual_run_id,
            experiment_id=experiment_id,
            root_dir=base_root,
            run_dir=run_dir,
            logs_dir=os.path.join(run_dir, "logs"),
            plots_dir=os.path.join(run_dir, "plots"),
            tables_dir=os.path.join(run_dir, "tables"),
            fields_dir=os.path.join(run_dir, "fields"),
            resolved_config_path=os.path.join(run_dir, "resolved_config.json"),
            result_json_path=os.path.join(run_dir, "result.json"),
        )
        if create_dirs:
            ctx.ensure_dirs()
        return ctx

    def ensure_dirs(self) -> None:
        for path in (self.run_dir, self.logs_dir, self.plots_dir, self.tables_dir, self.fields_dir):
            os.makedirs(path, exist_ok=True)

    def write_resolved_config(self, config: Any) -> None:
        self.ensure_dirs()
        with open(self.resolved_config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
