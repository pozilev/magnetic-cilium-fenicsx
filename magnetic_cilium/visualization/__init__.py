"""Visualization module namespace."""
from .geometry import geometry_plot_specs
from .magnetics import magnetics_plot_specs
from .mechanics import mechanics_plot_specs
from .specs import PlotSpec
from .sweeps import sweep_plot_specs

__all__ = [
    "PlotSpec",
    "geometry_plot_specs",
    "mechanics_plot_specs",
    "magnetics_plot_specs",
    "sweep_plot_specs",
]
