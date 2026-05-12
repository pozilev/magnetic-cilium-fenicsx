from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, init=False)
class CiliumGeometry:
    radius: float
    lower_length: float
    magnetic_length: float
    substrate_radius: float
    substrate_thickness: float

    def __init__(
        self,
        *,
        radius: float | None = None,
        diameter: float | None = None,
        lower_length: float,
        magnetic_length: float,
        substrate_radius: float = 0.0,
        substrate_thickness: float = 0.0,
    ) -> None:
        if radius is None and diameter is None:
            raise ValueError("Either radius or diameter must be provided.")
        resolved_radius = radius if radius is not None else float(diameter) / 2.0
        object.__setattr__(self, "radius", float(resolved_radius))
        object.__setattr__(self, "lower_length", float(lower_length))
        object.__setattr__(self, "magnetic_length", float(magnetic_length))
        object.__setattr__(self, "substrate_radius", float(substrate_radius))
        object.__setattr__(self, "substrate_thickness", float(substrate_thickness))

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def total_length(self) -> float:
        return self.lower_length + self.magnetic_length


@dataclass(frozen=True, init=False)
class HallSensorGeometry:
    center: tuple[float, float, float]
    size: tuple[float, float, float] | None
    average_radius: float | None = None
    average_n: int | None = None

    def __init__(
        self,
        *,
        center: Sequence[float] | None = None,
        size: Sequence[float] | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        average_radius: float | None = None,
        average_n: int | None = None,
    ) -> None:
        if center is None:
            if x is None or y is None or z is None:
                raise ValueError("Either center or x/y/z coordinates must be provided.")
            center = (x, y, z)
        resolved_center = tuple(float(value) for value in center)
        if len(resolved_center) != 3:
            raise ValueError("Hall sensor center must contain exactly 3 coordinates.")
        resolved_size = None if size is None else tuple(float(value) for value in size)
        if resolved_size is not None and len(resolved_size) != 3:
            raise ValueError("Hall sensor size must contain exactly 3 values.")
        object.__setattr__(self, "center", resolved_center)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "average_radius", average_radius)
        object.__setattr__(self, "average_n", average_n)

    @property
    def x(self) -> float:
        return self.center[0]

    @property
    def y(self) -> float:
        return self.center[1]

    @property
    def z(self) -> float:
        return self.center[2]
