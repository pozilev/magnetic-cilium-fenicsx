from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BVector:
    x_T: float | None = None
    y_T: float | None = None
    z_T: float | None = None

    @property
    def norm_T(self) -> float | None:
        if self.x_T is None or self.y_T is None or self.z_T is None:
            return None
        return math.sqrt(self.x_T * self.x_T + self.y_T * self.y_T + self.z_T * self.z_T)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "x_T": self.x_T,
            "y_T": self.y_T,
            "z_T": self.z_T,
            "norm_T": self.norm_T,
        }


@dataclass(frozen=True)
class SensorResponse:
    sensor_position_m: tuple[float | None, float | None, float | None]
    B_initial: BVector
    B_deformed: BVector
    delta_B: BVector
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, values: Mapping[str, Any]) -> "SensorResponse":
        return cls(
            sensor_position_m=(
                _optional_float(values.get("sensor_x_m")),
                _optional_float(values.get("sensor_y_m")),
                _optional_float(values.get("sensor_z_m")),
            ),
            B_initial=BVector(
                _optional_float(values.get("B0_sensor_x_T"), scale_uT_fallback=values.get("B0_sensor_x_uT")),
                _optional_float(values.get("B0_sensor_y_T"), scale_uT_fallback=values.get("B0_sensor_y_uT")),
                _optional_float(values.get("B0_sensor_z_T"), scale_uT_fallback=values.get("B0_sensor_z_uT")),
            ),
            B_deformed=BVector(
                _optional_float(values.get("B1_sensor_x_T"), scale_uT_fallback=values.get("B1_sensor_x_uT")),
                _optional_float(values.get("B1_sensor_y_T"), scale_uT_fallback=values.get("B1_sensor_y_uT")),
                _optional_float(values.get("B1_sensor_z_T"), scale_uT_fallback=values.get("B1_sensor_z_uT")),
            ),
            delta_B=BVector(
                _optional_float(values.get("dB_sensor_x_T"), scale_uT_fallback=values.get("dB_sensor_x_uT")),
                _optional_float(values.get("dB_sensor_y_T"), scale_uT_fallback=values.get("dB_sensor_y_uT")),
                _optional_float(values.get("dB_sensor_z_T"), scale_uT_fallback=values.get("dB_sensor_z_uT")),
            ),
            values=dict(values),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_position_m": self.sensor_position_m,
            "B_initial": self.B_initial.to_dict(),
            "B_deformed": self.B_deformed.to_dict(),
            "delta_B": self.delta_B.to_dict(),
            "values": dict(self.values),
        }


@dataclass(frozen=True)
class MagneticResult:
    model_type: str
    run_id: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    sensor_response: SensorResponse | None = None

    @classmethod
    def from_legacy(
        cls,
        values: Mapping[str, Any],
        *,
        model_type: str,
        run_id: str | None = None,
    ) -> "MagneticResult":
        public_values = {str(k): v for k, v in values.items() if not str(k).startswith("_")}
        return cls(
            model_type=model_type,
            run_id=run_id,
            values=public_values,
            sensor_response=SensorResponse.from_legacy(public_values),
        )

    @property
    def delta_B_norm_uT(self) -> float | None:
        return _optional_float(self.values.get("dB_sensor_norm_uT"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_type": self.model_type,
            "values": dict(self.values),
            "sensor_response": None if self.sensor_response is None else self.sensor_response.to_dict(),
        }


def _optional_float(value: Any, *, scale_uT_fallback: Any = None) -> float | None:
    if value in ("", None):
        if scale_uT_fallback in ("", None):
            return None
        try:
            return float(scale_uT_fallback) * 1.0e-6
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
