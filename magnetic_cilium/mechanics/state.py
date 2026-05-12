from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class MechanicsState:
    domain: Any
    material: Any
    u_vertices: Any
    params: Any | None = None


@dataclass(frozen=True)
class MechanicsResult:
    run_id: str | None = None
    run_index: int | None = None
    study: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    state: MechanicsState | None = None

    @classmethod
    def from_result_dict(
        cls,
        values: Mapping[str, Any],
        *,
        run_id: str | None = None,
        params: Any | None = None,
    ) -> "MechanicsResult":
        public_values = {str(k): v for k, v in values.items() if not str(k).startswith("_")}
        state = None
        if "_domain" in values and "_material" in values and "_u_vertices" in values:
            state = MechanicsState(
                domain=values["_domain"],
                material=values["_material"],
                u_vertices=values["_u_vertices"],
                params=params,
            )
        return cls(
            run_id=run_id,
            run_index=_optional_int(public_values.get("run")),
            study=_optional_str(public_values.get("study")),
            values=public_values,
            state=state,
        )

    @property
    def restart_file(self) -> str | None:
        return _optional_str(self.values.get("restart_file"))

    @property
    def reaction_force_x_uN(self) -> float | None:
        return _optional_float(self.values.get("reaction_force_x_uN"))

    @property
    def max_top_u_x_m(self) -> float | None:
        return _optional_float(self.values.get("max_top_u_x_m"))

    @property
    def J_min(self) -> float | None:
        return _optional_float(self.values.get("J_min"))

    @property
    def J_max(self) -> float | None:
        return _optional_float(self.values.get("J_max"))

    @property
    def von_mises_max_Pa(self) -> float | None:
        return _optional_float(self.values.get("von_mises_max_Pa"))

    def to_dict(self, *, include_state: bool = False) -> dict[str, Any]:
        data = {
            "run_id": self.run_id,
            "run_index": self.run_index,
            "study": self.study,
            "values": dict(self.values),
        }
        if include_state and self.state is not None:
            data["state"] = asdict(self.state)
        return data


def _optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value in ("", None):
        return None
    return str(value)
