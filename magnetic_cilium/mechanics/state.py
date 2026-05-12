from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MechanicsState:
    domain: Any
    material: Any
    u_vertices: Any


@dataclass(frozen=True)
class MechanicsResult:
    values: dict[str, Any] = field(default_factory=dict)
