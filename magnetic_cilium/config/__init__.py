"""Configuration facade."""

try:
    from magnetic_cilium_pipeline.config import *  # noqa: F401,F403
except ModuleNotFoundError:
    from config import *  # type: ignore # noqa: F401,F403
