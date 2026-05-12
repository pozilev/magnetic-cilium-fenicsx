from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when a user configuration is structurally invalid."""


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _float(mapping: dict[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return float(mapping[key])
    return float(default)


def _int(mapping: dict[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return int(mapping[key])
    return int(default)


def _str(mapping: dict[str, Any], *keys: str, default: str | None = None) -> str | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return str(mapping[key])
    return default


def _require_positive(name: str, value: float, errors: list[str]) -> None:
    if value <= 0.0:
        errors.append(f"{name} must be positive, got {value!r}")


def _require_nonnegative(name: str, value: float, errors: list[str]) -> None:
    if value < 0.0:
        errors.append(f"{name} must be non-negative, got {value!r}")


def _require_poisson(name: str, value: float, errors: list[str]) -> None:
    if not (-1.0 < value < 0.5):
        errors.append(f"{name} must satisfy -1 < nu < 0.5, got {value!r}")


@dataclass(frozen=True)
class GeometryConfig:
    D: float = 120e-6
    L1: float = 2e-3
    L2: float = 2e-3
    substrate_radius: float = 0.60e-3
    substrate_thickness: float = 0.50e-3

    @property
    def R(self) -> float:
        return self.D / 2.0

    @property
    def L(self) -> float:
        return self.L1 + self.L2

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> GeometryConfig:
        return cls(
            D=_float(data, "D", default=cls.D),
            L1=_float(data, "L1", default=cls.L1),
            L2=_float(data, "L2", default=cls.L2),
            substrate_radius=_float(data, "substrate_radius", default=cls.substrate_radius),
            substrate_thickness=_float(data, "substrate_thickness", default=cls.substrate_thickness),
        )

    def validate(self, errors: list[str]) -> None:
        _require_positive("geometry.D", self.D, errors)
        _require_positive("geometry.L1", self.L1, errors)
        _require_positive("geometry.L2", self.L2, errors)
        _require_positive("geometry.substrate_radius", self.substrate_radius, errors)
        _require_positive("geometry.substrate_thickness", self.substrate_thickness, errors)


@dataclass(frozen=True)
class MaterialConfig:
    E_pdms: float = 1.5e6
    nu_pdms: float = 0.49
    E_magnetic: float = 16.6e6
    nu_magnetic: float = 0.49

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> MaterialConfig:
        shared_nu = data.get("nu")
        return cls(
            E_pdms=_float(data, "E_pdms", default=cls.E_pdms),
            nu_pdms=float(data.get("nu_pdms", shared_nu if shared_nu is not None else cls.nu_pdms)),
            E_magnetic=_float(data, "E_magnetic", default=cls.E_magnetic),
            nu_magnetic=float(data.get("nu_magnetic", shared_nu if shared_nu is not None else cls.nu_magnetic)),
        )

    def validate(self, errors: list[str]) -> None:
        _require_positive("materials.E_pdms", self.E_pdms, errors)
        _require_positive("materials.E_magnetic", self.E_magnetic, errors)
        _require_poisson("materials.nu_pdms", self.nu_pdms, errors)
        _require_poisson("materials.nu_magnetic", self.nu_magnetic, errors)


@dataclass(frozen=True)
class MeshConfig:
    h_cilium: float = 30e-6
    h_substrate: float = 100e-6
    element_degree: int = 2
    h_air: float = 100e-6
    h_air_near: float | None = None
    h_air_far: float | None = None
    max_air_cells: int = 700000
    allow_large_air_mesh: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> MeshConfig:
        h_air = _float(data, "h_air", default=cls.h_air)
        return cls(
            h_cilium=_float(data, "h_cilium", default=cls.h_cilium),
            h_substrate=_float(data, "h_substrate", default=cls.h_substrate),
            element_degree=_int(data, "element_degree", default=cls.element_degree),
            h_air=h_air,
            h_air_near=float(data["h_air_near"]) if data.get("h_air_near") is not None else h_air,
            h_air_far=float(data["h_air_far"]) if data.get("h_air_far") is not None else 3.0 * h_air,
            max_air_cells=_int(data, "max_air_cells", default=cls.max_air_cells),
            allow_large_air_mesh=_as_bool(data.get("allow_large_air_mesh"), cls.allow_large_air_mesh),
        )

    def validate(self, errors: list[str]) -> None:
        _require_positive("mesh.h_cilium", self.h_cilium, errors)
        _require_positive("mesh.h_substrate", self.h_substrate, errors)
        _require_positive("mesh.h_air", self.h_air, errors)
        if self.h_air_near is not None:
            _require_positive("mesh.h_air_near", self.h_air_near, errors)
        if self.h_air_far is not None:
            _require_positive("mesh.h_air_far", self.h_air_far, errors)
        if self.element_degree < 1:
            errors.append(f"mesh.element_degree must be >= 1, got {self.element_degree!r}")
        if self.max_air_cells <= 0:
            errors.append(f"mesh.max_air_cells must be positive, got {self.max_air_cells!r}")


@dataclass(frozen=True)
class MechanicsConfig:
    load_type: str = "displacement"
    delta_x: float = 0.32e-3
    n_steps: int = 12
    target_reaction_uN: float = 60.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> MechanicsConfig:
        return cls(
            load_type=str(data.get("load_type", cls.load_type)),
            delta_x=_float(data, "delta_x", "delta", default=cls.delta_x),
            n_steps=_int(data, "n_steps", default=cls.n_steps),
            target_reaction_uN=_float(data, "target_reaction_uN", default=cls.target_reaction_uN),
        )

    def validate(self, errors: list[str]) -> None:
        if self.load_type != "displacement":
            errors.append(f"mechanics.load_type={self.load_type!r} is not supported yet")
        if self.n_steps < 1:
            errors.append(f"mechanics.n_steps must be >= 1, got {self.n_steps!r}")
        _require_nonnegative("mechanics.delta_x", self.delta_x, errors)
        _require_positive("mechanics.target_reaction_uN", self.target_reaction_uN, errors)


@dataclass(frozen=True)
class MagneticConfig:
    Br_magnetic: float = 0.10
    sensor_x: float = 0.0
    sensor_y: float = 0.0
    sensor_z: float = -50e-6
    rotate_magnetization: bool = True
    magnetic_boundary: str = "natural"
    sensor_average: bool = False
    sensor_average_radius: float = 25e-6
    sensor_average_n: int = 5
    projection_mode: str = "cell_center"
    air_radius_factor: float = 8.0
    air_below_factor: float = 4.0
    air_above_factor: float = 4.0
    near_radius_factor: float = 3.0
    near_source_padding_factor: float = 2.0
    near_sensor_padding_factor: float = 1.0
    target_sensitivity_uT_per_uN: float = 1.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any], geometry: GeometryConfig) -> MagneticConfig:
        sensor_x = data.get("sensor_x")
        sensor_y = data.get("sensor_y")
        sensor_z = data.get("sensor_z")
        if sensor_x is None and data.get("sensor_x_over_r") is not None:
            sensor_x = float(data["sensor_x_over_r"]) * geometry.R
        if sensor_y is None and data.get("sensor_y_over_r") is not None:
            sensor_y = float(data["sensor_y_over_r"]) * geometry.R
        if sensor_z is None and data.get("sensor_z_over_r") is not None:
            sensor_z = float(data["sensor_z_over_r"]) * geometry.R
        return cls(
            Br_magnetic=_float(data, "Br_magnetic", "Br", default=cls.Br_magnetic),
            sensor_x=float(sensor_x if sensor_x is not None else cls.sensor_x),
            sensor_y=float(sensor_y if sensor_y is not None else cls.sensor_y),
            sensor_z=float(sensor_z if sensor_z is not None else cls.sensor_z),
            rotate_magnetization=_as_bool(data.get("rotate_magnetization"), cls.rotate_magnetization),
            magnetic_boundary=str(data.get("magnetic_boundary", cls.magnetic_boundary)),
            sensor_average=_as_bool(data.get("sensor_average"), cls.sensor_average),
            sensor_average_radius=_float(data, "sensor_average_radius", default=cls.sensor_average_radius),
            sensor_average_n=_int(data, "sensor_average_n", default=cls.sensor_average_n),
            projection_mode=str(data.get("projection_mode", cls.projection_mode)),
            air_radius_factor=_float(data, "air_radius_factor", default=cls.air_radius_factor),
            air_below_factor=_float(data, "air_below_factor", default=cls.air_below_factor),
            air_above_factor=_float(data, "air_above_factor", default=cls.air_above_factor),
            near_radius_factor=_float(data, "near_radius_factor", default=cls.near_radius_factor),
            near_source_padding_factor=_float(data, "near_source_padding_factor", default=cls.near_source_padding_factor),
            near_sensor_padding_factor=_float(data, "near_sensor_padding_factor", default=cls.near_sensor_padding_factor),
            target_sensitivity_uT_per_uN=_float(
                data, "target_sensitivity_uT_per_uN", default=cls.target_sensitivity_uT_per_uN
            ),
        )

    def validate(self, errors: list[str]) -> None:
        _require_nonnegative("magnetics.Br_magnetic", self.Br_magnetic, errors)
        if self.magnetic_boundary not in {"natural", "dirichlet_zero"}:
            errors.append(f"magnetics.magnetic_boundary must be natural or dirichlet_zero, got {self.magnetic_boundary!r}")
        if self.projection_mode not in {"cell_center"}:
            errors.append(f"magnetics.projection_mode={self.projection_mode!r} is not supported yet")
        if self.sensor_average:
            _require_positive("magnetics.sensor_average_radius", self.sensor_average_radius, errors)
            if self.sensor_average_n < 1:
                errors.append(f"magnetics.sensor_average_n must be >= 1, got {self.sensor_average_n!r}")
        _require_positive("magnetics.air_radius_factor", self.air_radius_factor, errors)
        _require_positive("magnetics.air_below_factor", self.air_below_factor, errors)
        _require_positive("magnetics.air_above_factor", self.air_above_factor, errors)
        _require_positive("magnetics.near_radius_factor", self.near_radius_factor, errors)
        _require_positive("magnetics.target_sensitivity_uT_per_uN", self.target_sensitivity_uT_per_uN, errors)


@dataclass(frozen=True)
class OutputConfig:
    outdir: str | None = None
    restart_dir: str | None = None
    master_csv_path: str = "../results/magnetic_results_master.csv"
    local_summary_path: str | None = None
    results_write_mode: str = "debug"
    experiment_id: str | None = None
    output_dir: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> OutputConfig:
        return cls(
            outdir=_str(data, "outdir", "output_dir", default=cls.outdir),
            restart_dir=_str(data, "restart_dir", default=cls.restart_dir),
            master_csv_path=_str(data, "master_csv_path", default=cls.master_csv_path) or cls.master_csv_path,
            local_summary_path=_str(data, "local_summary_path", default=cls.local_summary_path),
            results_write_mode=_str(data, "results_write_mode", default=cls.results_write_mode) or cls.results_write_mode,
            experiment_id=_str(data, "experiment_id", default=cls.experiment_id),
            output_dir=_str(data, "output_dir", default=cls.output_dir),
        )

    def validate(self, errors: list[str]) -> None:
        if self.results_write_mode not in {"debug", "experiment"}:
            errors.append(
                f"output.results_write_mode must be debug or experiment, got {self.results_write_mode!r}"
            )


@dataclass(frozen=True)
class RunConfig:
    mode: str = "validation"
    config_path: str | None = None
    master_mode: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any], config_path: str | None = None) -> RunConfig:
        mode = str(data.get("mode", cls.mode))
        if mode == "final":
            mode = "full"
        if mode == "magnetic-validation":
            mode = "magnetic-only"
        if mode == "magnetic-sensor-position-sweep":
            mode = "magnetics-fem-validation"
        return cls(mode=mode, config_path=config_path, master_mode=_str(data, "master_mode", default=None))

    def validate(self, errors: list[str]) -> None:
        allowed = {
            "validation",
            "full",
            "mechanics",
            "magnetics",
            "magnetic-only",
            "magnetics-fem",
            "magnetics-fem-validation",
            "interpolate-magnetic-results",
        }
        if self.mode not in allowed:
            errors.append(f"run.mode must be one of {sorted(allowed)}, got {self.mode!r}")


@dataclass(frozen=True)
class SweepConfig:
    enabled: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
    fixed_physical_parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> SweepConfig:
        sweep = data.get("sweep")
        fixed = data.get("fixed_physical_parameters") or {}
        return cls(enabled=isinstance(sweep, dict), raw=sweep or {}, fixed_physical_parameters=fixed)

    def validate(self, errors: list[str]) -> None:
        if not self.enabled:
            return
        if not self.raw:
            errors.append("sweep must contain at least one group")
            return
        for group_name, group in self.raw.items():
            if not isinstance(group, dict):
                errors.append(f"sweep.{group_name} must be a mapping")
                continue
            cases = group.get("cases")
            if not isinstance(cases, list) or not cases:
                errors.append(f"sweep.{group_name}.cases must be a non-empty list")
                continue
            for idx, case in enumerate(cases):
                if not isinstance(case, dict) or not case.get("id"):
                    errors.append(f"sweep.{group_name}.cases[{idx}] must be a mapping with an id")


@dataclass(frozen=True)
class SimulationConfig:
    run: RunConfig = field(default_factory=RunConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    materials: MaterialConfig = field(default_factory=MaterialConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    mechanics: MechanicsConfig = field(default_factory=MechanicsConfig)
    magnetics: MagneticConfig = field(default_factory=MagneticConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], config_path: str | None = None) -> SimulationConfig:
        merged = _flatten_config_mapping(data)
        geometry = GeometryConfig.from_mapping(merged)
        return cls(
            run=RunConfig.from_mapping(merged, config_path=config_path),
            geometry=geometry,
            materials=MaterialConfig.from_mapping(merged),
            mesh=MeshConfig.from_mapping(merged),
            mechanics=MechanicsConfig.from_mapping(merged),
            magnetics=MagneticConfig.from_mapping(merged, geometry),
            output=OutputConfig.from_mapping(merged),
            sweep=SweepConfig.from_mapping(data),
            raw=data,
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        self.run.validate(errors)
        self.geometry.validate(errors)
        self.materials.validate(errors)
        self.mesh.validate(errors)
        self.mechanics.validate(errors)
        self.magnetics.validate(errors)
        self.output.validate(errors)
        self.sweep.validate(errors)
        if self.run.mode in {"magnetics", "magnetics-fem", "magnetics-fem-validation"}:
            if not self.output.restart_dir:
                errors.append(f"run.mode={self.run.mode!r} requires output.restart_dir")
        if self.sweep.enabled and self.run.mode != "magnetics-fem-validation":
            errors.append("configs with a sweep section should use mode magnetics-fem-validation")
        return errors

    def ensure_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ConfigValidationError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flatten_config_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Merge current flat YAML and nested sweep YAML into one validated view."""
    if not {"global", "fixed_physical_parameters", "sweep"}.intersection(data):
        return dict(data)

    merged: dict[str, Any] = {}
    global_config = data.get("global") or {}
    fixed_physical = data.get("fixed_physical_parameters") or {}
    if isinstance(global_config, dict):
        merged.update(global_config)
    if isinstance(fixed_physical, dict):
        merged.update(fixed_physical)
    if "sweep" in data:
        merged["mode"] = "magnetics-fem-validation"
    return merged
