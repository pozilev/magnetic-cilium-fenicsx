"""Configuration facade."""

try:
    from magnetic_cilium_pipeline.config import *  # noqa: F401,F403
except ModuleNotFoundError:
    from config import *  # type: ignore # noqa: F401,F403

from .adapters import simulation_config_to_legacy_namespace, simulation_config_to_model_params

__all__ = [
    name for name in globals()
    if not name.startswith("_")
]
