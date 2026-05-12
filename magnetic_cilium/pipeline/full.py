from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from magnetic_cilium.config.adapters import simulation_config_to_model_params
from magnetic_cilium.config.loader import load_simulation_config
from magnetic_cilium.config.schema import SimulationConfig
from magnetic_cilium.io.json_io import write_json
from magnetic_cilium.io.run_context import RunContext
from magnetic_cilium.magnetics.response import compute_dipole_response_from_mechanics
from magnetic_cilium.magnetics.state import MagneticResult
from magnetic_cilium.mechanics.solver import solve_mechanics_from_config
from magnetic_cilium.mechanics.state import MechanicsResult
from magnetic_cilium.postprocess.quality import QualityReport, evaluate_quality


@dataclass(frozen=True)
class PipelineStep:
    name: str
    enabled: bool = True
    backend: str = "architecture"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    mode: str
    run_dir: str
    dry_run: bool
    steps: list[PipelineStep]
    mechanics: MechanicsResult | None = None
    magnetics: MagneticResult | None = None
    quality: QualityReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "run_dir": self.run_dir,
            "dry_run": self.dry_run,
            "steps": [step.to_dict() for step in self.steps],
            "mechanics": None if self.mechanics is None else self.mechanics.to_dict(),
            "magnetics": None if self.magnetics is None else self.magnetics.to_dict(),
            "quality": None if self.quality is None else self.quality.to_dict(),
        }


def build_full_pipeline_plan(config: SimulationConfig, ctx: RunContext) -> list[PipelineStep]:
    magnetic_enabled = config.run.mode == "full"
    return [
        PipelineStep("validate_config", details={"config_path": config.run.config_path}),
        PipelineStep("create_run_context", details=ctx.to_dict()),
        PipelineStep("save_resolved_config", enabled=True),
        PipelineStep("solve_mechanics", enabled=config.run.mode in {"full", "mechanics"}, backend="mechanics-fem"),
        PipelineStep("postprocess_mechanics", enabled=config.run.mode in {"full", "mechanics"}, backend="mechanics-fem"),
        PipelineStep("save_restart", enabled=config.run.mode in {"full", "mechanics"}, backend="mechanics-fem"),
        PipelineStep("solve_magnetics_dipole", enabled=magnetic_enabled, backend="dipole"),
        PipelineStep("evaluate_quality", enabled=True),
        PipelineStep("save_result_json", enabled=True),
    ]


def run_full_pipeline(
    config_or_path: SimulationConfig | str,
    *,
    dry_run: bool = False,
    root_dir: str | None = None,
    run_id: str | None = None,
) -> PipelineResult:
    config = _load_config(config_or_path)
    if config.run.mode not in {"full", "mechanics"}:
        raise ValueError(f"run_full_pipeline supports full/mechanics modes, got {config.run.mode!r}")

    ctx = RunContext.create(config, root_dir=root_dir, create_dirs=not dry_run, run_id=run_id)
    steps = build_full_pipeline_plan(config, ctx)
    if dry_run:
        return PipelineResult(ctx.run_id, config.run.mode, ctx.run_dir, True, steps)

    ctx.write_resolved_config(config)
    mechanics = solve_mechanics_from_config(config, run_index=1, study=config.run.mode, run_id=ctx.run_id)
    magnetics = None
    merged_values = dict(mechanics.values)

    if config.run.mode == "full":
        params = simulation_config_to_model_params(config)
        magnetics = compute_dipole_response_from_mechanics(
            mechanics,
            params,
            sensor_average=config.magnetics.sensor_average,
            sensor_average_radius=config.magnetics.sensor_average_radius,
            sensor_average_n=config.magnetics.sensor_average_n,
            run_id=ctx.run_id,
        )
        merged_values.update(magnetics.values)

    quality = evaluate_quality(merged_values, model_type="dipole")
    result = PipelineResult(ctx.run_id, config.run.mode, ctx.run_dir, False, steps, mechanics, magnetics, quality)
    write_json(ctx.result_json_path, result.to_dict())
    return result


def _load_config(config_or_path: SimulationConfig | str) -> SimulationConfig:
    if isinstance(config_or_path, SimulationConfig):
        return config_or_path
    return load_simulation_config(config_or_path)
