try:
    from magnetic_cilium_pipeline.config.schema import *  # noqa: F401,F403
except ModuleNotFoundError:
    from config.schema import *  # type: ignore # noqa: F401,F403
