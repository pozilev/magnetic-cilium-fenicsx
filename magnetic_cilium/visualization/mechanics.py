from __future__ import annotations

import os
from typing import Any

from .specs import PlotSpec


def mechanics_plot_specs() -> list[PlotSpec]:
    return [
        PlotSpec(
            name="deformed_geometry",
            filename="mechanics_deformed_geometry.png",
            kind="mechanics",
            required_fields=("max_top_u_x_m", "reaction_force_x_N"),
            description="Deformed cilium shape under prescribed displacement.",
        ),
        PlotSpec(
            name="J_detF",
            filename="mechanics_J_detF.png",
            kind="mechanics",
            required_fields=("J_min", "J_max"),
            description="det(F) quality field.",
        ),
        PlotSpec(
            name="von_mises",
            filename="mechanics_von_mises.png",
            kind="mechanics",
            required_fields=("von_mises_max_Pa",),
            description="von Mises stress distribution.",
        ),
    ]


def _cell_edges(cells, max_edges: int = 20000):
    edges = set()
    for cell in cells:
        ids = [int(v) for v in cell[:4]]
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            edge = tuple(sorted((ids[i], ids[j])))
            edges.add(edge)
            if len(edges) >= max_edges:
                return list(edges)
    return list(edges)


def plot_deformation_step_frame(
    points,
    cells,
    u_vertices,
    output_path: str,
    *,
    step: int,
    total_steps: int,
    alpha: float,
    newton_iterations: int | None = None,
    scalar_values=None,
    scalar_label: str | None = None,
) -> str:
    """Save one Russian-labeled mechanics deformation frame in x-z projection."""
    import matplotlib.pyplot as plt

    import numpy as np

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    X = np.asarray(points, dtype=float)[:, :3]
    U = np.asarray(u_vertices, dtype=float)[:, :3]
    x_def = X + U
    cells_arr = np.asarray(cells)

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    edges = _cell_edges(cells_arr)
    scale = 1.0e6
    for i, j in edges:
        ax.plot([X[i, 0] * scale, X[j, 0] * scale], [X[i, 2] * scale, X[j, 2] * scale], color="#9ca3af", linewidth=0.25, alpha=0.22)
        ax.plot([x_def[i, 0] * scale, x_def[j, 0] * scale], [x_def[i, 2] * scale, x_def[j, 2] * scale], color="#2563eb", linewidth=0.35, alpha=0.45)

    if scalar_values is not None:
        values = np.asarray(scalar_values, dtype=float)
        sc = ax.scatter(x_def[:, 0] * scale, x_def[:, 2] * scale, c=values[: x_def.shape[0]], s=8, cmap="viridis", label=scalar_label or "поле")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(scalar_label or "значение")
    else:
        ax.scatter(X[:, 0] * scale, X[:, 2] * scale, s=5, color="#6b7280", alpha=0.35, label="недеформированная геометрия")
        ax.scatter(x_def[:, 0] * scale, x_def[:, 2] * scale, s=5, color="#1d4ed8", alpha=0.65, label="деформированная геометрия")

    stride = max(1, X.shape[0] // 250)
    ax.quiver(
        X[::stride, 0] * scale,
        X[::stride, 2] * scale,
        U[::stride, 0] * scale,
        U[::stride, 2] * scale,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#b91c1c",
        width=0.002,
        alpha=0.75,
        label="перемещения",
    )

    title = f"Механическая деформация: шаг {step}/{total_steps}, нагрузка {alpha:.3f}"
    if newton_iterations is not None:
        title += f", итераций Ньютона: {newton_iterations}"
    ax.set_title(title)
    ax.set_xlabel("x, мкм")
    ax.set_ylabel("z, мкм")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d1d5db", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


class MechanicsFrameRecorder:
    """Callback object for continuation-step deformation frames.

    The current nonlinear backend exposes converged load-continuation states.
    Per-Newton-iteration states are not available from DOLFINx's solver wrapper,
    so this recorder is deliberately attached at the safe post-step hook.
    """

    def __init__(self, outdir: str, *, every: int = 1, make_animation: bool = False):
        self.outdir = outdir
        self.every = max(1, int(every))
        self.make_animation = bool(make_animation)
        self.frames: list[str] = []
        os.makedirs(self.outdir, exist_ok=True)

    def __call__(
        self,
        *,
        step: int,
        total_steps: int,
        alpha: float,
        displacement,
        domain,
        params,
        newton_iterations: int,
        converged: bool,
        **_: Any,
    ) -> None:
        if step % self.every != 0 and step != total_steps:
            return
        from magnetic_cilium.mechanics.backend import evaluate_displacement_at_geometry_vertices, extract_cells_from_domain

        u_vertices = evaluate_displacement_at_geometry_vertices(domain, displacement)
        points = domain.geometry.x[:, :3].copy()
        cells = extract_cells_from_domain(domain)
        frame_path = os.path.join(self.outdir, f"mechanics_step_{step:03d}.png")
        plot_deformation_step_frame(
            points,
            cells,
            u_vertices,
            frame_path,
            step=step,
            total_steps=total_steps,
            alpha=alpha,
            newton_iterations=newton_iterations,
        )
        self.frames.append(frame_path)

    def finalize(self) -> str | None:
        if not self.make_animation or len(self.frames) < 2:
            return None
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError:
            return None
        gif_path = os.path.join(self.outdir, "mechanics_deformation.gif")
        images = [imageio.imread(path) for path in self.frames]
        imageio.mimsave(gif_path, images, duration=0.35)
        return gif_path
