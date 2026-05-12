from __future__ import annotations

from magnetic_cilium._compat import legacy_attr
from magnetic_cilium.magnetics.state import MagneticResult
from magnetic_cilium.mechanics.state import MechanicsResult


def compute_dipole_response(*args, **kwargs):
    raw = legacy_attr("magnetics_dipoles", "compute_magnetic_dipole_diagnostics")(*args, **kwargs)
    return MagneticResult.from_legacy(raw, model_type="dipole")


def compute_fem_response(*args, **kwargs):
    raw = legacy_attr("magnetics_fem", "compute_magnetostatic_fem_diagnostics")(*args, **kwargs)
    return MagneticResult.from_legacy(raw, model_type="fem")


def compute_dipole_response_from_mechanics(
    mechanics: MechanicsResult,
    params,
    *,
    sensor_average: bool = False,
    sensor_average_radius: float = 25e-6,
    sensor_average_n: int = 5,
    run_id: str | None = None,
) -> MagneticResult:
    if mechanics.state is None:
        raise ValueError("MechanicsResult.state is required for dipole magnetic response.")
    raw = legacy_attr("pipeline", "run_magnetics_case_from_objects")(
        mechanics.state.domain,
        mechanics.state.material,
        mechanics.state.u_vertices,
        params,
        mechanics.values,
        sensor_average=sensor_average,
        sensor_average_radius=sensor_average_radius,
        sensor_average_n=sensor_average_n,
    )
    return MagneticResult.from_legacy(raw, model_type="dipole", run_id=run_id)
