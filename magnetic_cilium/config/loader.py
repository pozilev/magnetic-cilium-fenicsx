try:
    from magnetic_cilium_pipeline.config.loader import *  # noqa: F401,F403
except ModuleNotFoundError:
    from config.loader import *  # type: ignore # noqa: F401,F403
