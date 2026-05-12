from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PlotSpec:
    name: str
    filename: str
    kind: str
    required_fields: tuple[str, ...]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
