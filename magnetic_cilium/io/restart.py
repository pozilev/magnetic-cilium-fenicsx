from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def save_mechanics_restart(*args, **kwargs):
    return legacy_attr("pipeline", "save_mechanics_restart")(*args, **kwargs)


def load_mechanics_restart(*args, **kwargs):
    return legacy_attr("pipeline", "load_mechanics_restart")(*args, **kwargs)
