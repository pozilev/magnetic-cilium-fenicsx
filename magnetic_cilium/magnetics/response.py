from __future__ import annotations

from magnetic_cilium.magnetics.state import MagneticResult
from magnetic_cilium.mechanics.state import MechanicsResult


def compute_dipole_response(*args, **kwargs):
    from magnetic_cilium.magnetics.dipole import compute_magnetic_dipole_diagnostics

    raw = compute_magnetic_dipole_diagnostics(*args, **kwargs)
    return MagneticResult.from_result_dict(raw, model_type="dipole")


def compute_fem_response(*args, **kwargs):
    from magnetic_cilium.magnetics.fem_scalar_potential import compute_magnetostatic_fem_diagnostics

    raw = compute_magnetostatic_fem_diagnostics(*args, **kwargs)
    return MagneticResult.from_result_dict(raw, model_type="fem")


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
    from magnetic_cilium.pipeline.execution import run_magnetics_case_from_objects

    raw = run_magnetics_case_from_objects(
        mechanics.state.domain,
        mechanics.state.material,
        mechanics.state.u_vertices,
        params,
        mechanics.values,
        sensor_average=sensor_average,
        sensor_average_radius=sensor_average_radius,
        sensor_average_n=sensor_average_n,
    )
    return MagneticResult.from_result_dict(raw, model_type="dipole", run_id=run_id)
