"""Configuration schema and loading helpers for magnetic cilium runs."""

from .loader import load_simulation_config, validate_config_file
from .schema import (
    ConfigValidationError,
    GeometryConfig,
    MagneticConfig,
    MaterialConfig,
    MechanicsConfig,
    MeshConfig,
    OutputConfig,
    RunConfig,
    SimulationConfig,
    SweepConfig,
)

__all__ = [
    "ConfigValidationError",
    "GeometryConfig",
    "MagneticConfig",
    "MaterialConfig",
    "MechanicsConfig",
    "MeshConfig",
    "OutputConfig",
    "RunConfig",
    "SimulationConfig",
    "SweepConfig",
    "load_simulation_config",
    "validate_config_file",
]
