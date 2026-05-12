from __future__ import annotations

import os
from typing import Any

from magnetic_cilium._compat import legacy_attr


def append_master_result(*args, **kwargs):
    return legacy_attr("magnetic_results", "append_master_result")(*args, **kwargs)


def append_master_results(*args, **kwargs):
    return legacy_attr("magnetic_results", "append_master_results")(*args, **kwargs)


def build_magnetic_master_row(*args, **kwargs):
    return legacy_attr("magnetic_results", "build_magnetic_master_row")(*args, **kwargs)


def sync_master_parquet(master_csv_path: str, parquet_path: str | None = None) -> dict[str, Any]:
    """Mirror master.csv to parquet when pandas/pyarrow are available.

    The CSV remains the canonical lightweight format. Parquet creation is best
    effort so solver runs do not fail on minimal FEniCS environments.
    """
    target = parquet_path or os.path.splitext(master_csv_path)[0] + ".parquet"
    if not os.path.exists(master_csv_path):
        return {"ok": False, "path": target, "reason": "missing_master_csv"}
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError:
        return {"ok": False, "path": target, "reason": "pandas_not_installed"}
    try:
        frame = pd.read_csv(master_csv_path)
        frame.to_parquet(target, index=False)
    except Exception as exc:  # pragma: no cover - depends on optional engines.
        return {"ok": False, "path": target, "reason": str(exc)}
    return {"ok": True, "path": target, "reason": "ok"}
