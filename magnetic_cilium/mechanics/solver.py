from __future__ import annotations

from magnetic_cilium._compat import legacy_attr
from magnetic_cilium.config.adapters import simulation_config_to_model_params
from magnetic_cilium.config.schema import SimulationConfig
from magnetic_cilium.mechanics.state import MechanicsResult


def solve_mechanics(domain, E, nu, params):
    return legacy_attr("mechanics_model", "solve_hyperelasticity_displacement_control_3d")(domain, E, nu, params)


def solve_hyperelasticity_displacement_control_3d(domain, E, nu, params):
    return solve_mechanics(domain, E, nu, params)


def solve_mechanics_case(params, *, run_index: int = 1, study: str = "mechanics", run_id: str | None = None) -> MechanicsResult:
    legacy_result = legacy_attr("pipeline", "run_mechanics_case")(params, run_index, study)
    return MechanicsResult.from_legacy(legacy_result, run_id=run_id, params=params)


def solve_mechanics_from_config(
    config: SimulationConfig,
    *,
    run_index: int = 1,
    study: str | None = None,
    run_id: str | None = None,
) -> MechanicsResult:
    params = simulation_config_to_model_params(config)
    return solve_mechanics_case(
        params,
        run_index=run_index,
        study=study or config.run.mode,
        run_id=run_id,
    )
