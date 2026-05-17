from __future__ import annotations

import csv
import logging
import os
from typing import Any

from .specs import PlotSpec


log = logging.getLogger("magnetic_cilium_3d")


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


def _cilium_cells(points, cells):
    import numpy as np

    X = np.asarray(points, dtype=float)[:, :3]
    cells_arr = np.asarray(cells, dtype=int)
    if cells_arr.size == 0:
        return cells_arr.reshape((0, 4))
    z = X[cells_arr[:, :4], 2]
    cilium_mask = np.mean(z, axis=1) >= -1e-12
    selected = cells_arr[cilium_mask]
    return selected if selected.size else cells_arr


def _set_3d_limits(ax, coords):
    import numpy as np

    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    centers = 0.5 * (mins + maxs)
    extents = np.maximum(maxs - mins, 1.0)
    extents[0] = max(extents[0], 0.22 * extents[2])
    extents[1] = max(extents[1], 0.14 * extents[2])
    pad = 0.06 * extents
    half = 0.5 * extents + pad
    ax.set_xlim(centers[0] - half[0], centers[0] + half[0])
    ax.set_ylim(centers[1] - half[1], centers[1] + half[1])
    ax.set_zlim(centers[2] - half[2], centers[2] + half[2])
    try:
        ax.set_box_aspect(extents)
    except AttributeError:
        pass


def _binned_centerline(coords, *, bins: int = 80):
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 2:
        return coords
    z = coords[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max <= z_min:
        return coords[np.argsort(z)]
    edges = np.linspace(z_min, z_max, bins + 1)
    centers = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (z >= lo) & (z < hi)
        if not np.any(mask):
            continue
        centers.append(np.mean(coords[mask], axis=0))
    mask = z >= edges[-2]
    if np.any(mask):
        centers.append(np.mean(coords[mask], axis=0))
    if not centers:
        return coords[np.argsort(z)]
    return np.asarray(centers)


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


def plot_deformation_step_frame_3d(
    points,
    cells,
    u_vertices,
    output_path: str,
    *,
    step: int,
    total_steps: int,
    alpha: float,
    prescribed_delta_x: float | None = None,
    reaction_force_x_uN: float | None = None,
    newton_iterations: int | None = None,
) -> str:
    """Save one 3D deformation frame for the cilium part of the mesh."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    X = np.asarray(points, dtype=float)[:, :3]
    U = np.asarray(u_vertices, dtype=float)[:, :3]
    x_def = X + U
    cells_arr = _cilium_cells(X, cells)
    vertex_ids = np.unique(cells_arr[:, :4].ravel()) if cells_arr.size else np.arange(X.shape[0])

    fig = plt.figure(figsize=(8.0, 6.4), dpi=150, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    scale = 1.0e6

    sample = vertex_ids
    if sample.size > 2600:
        sample = sample[:: max(1, sample.size // 2600)]
    disp_norm = np.linalg.norm(U[sample], axis=1) * scale
    points_plot = x_def[sample] * scale
    sc = ax.scatter(
        points_plot[:, 0],
        points_plot[:, 1],
        points_plot[:, 2],
        c=disp_norm,
        s=8,
        cmap="viridis",
        depthshade=True,
        alpha=0.72,
        linewidths=0.0,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label("|u|, мкм")

    initial_centerline = _binned_centerline(X[vertex_ids] * scale)
    deformed_centerline = _binned_centerline(x_def[vertex_ids] * scale)
    ax.plot(
        initial_centerline[:, 0],
        initial_centerline[:, 1],
        initial_centerline[:, 2],
        color="#6b7280",
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        label="исходная ось",
    )
    ax.plot(
        deformed_centerline[:, 0],
        deformed_centerline[:, 1],
        deformed_centerline[:, 2],
        color="#1d4ed8",
        linewidth=2.0,
        label="деформированная ось",
    )

    title_parts = [f"Шаг {step}/{total_steps}, нагрузка {alpha:.3f}"]
    if prescribed_delta_x is not None:
        title_parts.append(f"delta={prescribed_delta_x * 1e3:.4f} мм")
    if reaction_force_x_uN is not None:
        title_parts.append(f"Rx={reaction_force_x_uN:.3f} мкН")
    if newton_iterations is not None:
        title_parts.append(f"Newton={newton_iterations}")
    ax.set_title("Деформированная ресничка\n" + ", ".join(title_parts), fontsize=10, pad=8)
    ax.set_xlabel("x, мкм")
    ax.set_ylabel("y, мкм")
    ax.set_zlabel("z, мкм")
    ax.view_init(elev=18.0, azim=-64.0)
    ax.tick_params(axis="both", which="major", labelsize=8, pad=1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.legend(loc="upper left", fontsize=8)
    _set_3d_limits(ax, (x_def[vertex_ids] * scale) if vertex_ids.size else (x_def * scale))
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path


class MechanicsFrameRecorder:
    """Callback object for continuation-step deformation frames.

    The nonlinear backend exposes converged load-continuation states together
    with the Newton iteration count used to reach each state. Those states are
    the useful points for the reaction-vs-displacement reporting curve.
    """

    def __init__(
        self,
        outdir: str,
        *,
        table_path: str | None = None,
        every: int = 1,
        make_animation: bool = False,
        save_frames: bool = True,
        save_xz_frames: bool = False,
    ):
        self.outdir = outdir
        self.table_path = table_path or os.path.join(outdir, "mechanics_newton_steps.csv")
        self.every = max(1, int(every))
        self.make_animation = bool(make_animation)
        self.save_frames = bool(save_frames)
        self.save_xz_frames = bool(save_xz_frames)
        self.frames: list[str] = []
        self.frames_3d: list[str] = []
        self.rows: list[dict[str, Any]] = []
        self.animation_path: str | None = None
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
        prescribed_delta_x: float | None = None,
        max_top_u_x: float | None = None,
        reaction_force_x: float | None = None,
        reaction_force_x_uN: float | None = None,
        **_: Any,
    ) -> None:
        if not self._should_save_frame(step, total_steps):
            self.record_step(
                step=step,
                total_steps=total_steps,
                alpha=alpha,
                prescribed_delta_x=prescribed_delta_x,
                max_top_u_x=max_top_u_x,
                reaction_force_x=reaction_force_x,
                reaction_force_x_uN=reaction_force_x_uN,
                newton_iterations=newton_iterations,
                converged=converged,
            )
            return

        from magnetic_cilium.mechanics.backend import evaluate_displacement_at_geometry_vertices, extract_cells_from_domain

        u_vertices = evaluate_displacement_at_geometry_vertices(domain, displacement)
        points = domain.geometry.x[:, :3].copy()
        cells = extract_cells_from_domain(domain)
        self.record_arrays(
            points,
            cells,
            u_vertices,
            step=step,
            total_steps=total_steps,
            alpha=alpha,
            prescribed_delta_x=prescribed_delta_x,
            max_top_u_x=max_top_u_x,
            reaction_force_x=reaction_force_x,
            reaction_force_x_uN=reaction_force_x_uN,
            newton_iterations=newton_iterations,
            converged=converged,
        )

    def record_arrays(
        self,
        points,
        cells,
        u_vertices,
        *,
        step: int,
        total_steps: int,
        alpha: float,
        prescribed_delta_x: float | None = None,
        max_top_u_x: float | None = None,
        reaction_force_x: float | None = None,
        reaction_force_x_uN: float | None = None,
        newton_iterations: int | None = None,
        converged: bool = True,
    ) -> None:
        frame_path = ""
        frame_3d_path = ""
        if self._should_save_frame(step, total_steps):
            xz_path = os.path.join(self.outdir, f"mechanics_step_{step:03d}.png")
            three_d_path = os.path.join(self.outdir, f"mechanics_step_{step:03d}_3d.png")
            try:
                if self.save_xz_frames:
                    frame_path = plot_deformation_step_frame(
                        points,
                        cells,
                        u_vertices,
                        xz_path,
                        step=step,
                        total_steps=total_steps,
                        alpha=alpha,
                        newton_iterations=newton_iterations,
                    )
                frame_3d_path = plot_deformation_step_frame_3d(
                    points,
                    cells,
                    u_vertices,
                    three_d_path,
                    step=step,
                    total_steps=total_steps,
                    alpha=alpha,
                    prescribed_delta_x=prescribed_delta_x,
                    reaction_force_x_uN=reaction_force_x_uN,
                    newton_iterations=newton_iterations,
                )
            except ModuleNotFoundError as exc:
                log.warning("mechanics: frame plotting skipped, missing dependency %s", exc.name)
            except Exception as exc:
                log.warning("mechanics: frame plotting skipped at step %s: %s", step, exc)
            if frame_path:
                self.frames.append(frame_path)
            if frame_3d_path:
                self.frames_3d.append(frame_3d_path)

        self.record_step(
            step=step,
            total_steps=total_steps,
            alpha=alpha,
            prescribed_delta_x=prescribed_delta_x,
            max_top_u_x=max_top_u_x,
            reaction_force_x=reaction_force_x,
            reaction_force_x_uN=reaction_force_x_uN,
            newton_iterations=newton_iterations,
            converged=converged,
            frame_path=frame_path,
            frame_3d_path=frame_3d_path,
        )

    def record_step(
        self,
        *,
        step: int,
        total_steps: int,
        alpha: float,
        prescribed_delta_x: float | None = None,
        max_top_u_x: float | None = None,
        reaction_force_x: float | None = None,
        reaction_force_x_uN: float | None = None,
        newton_iterations: int | None = None,
        converged: bool = True,
        frame_path: str = "",
        frame_3d_path: str = "",
    ) -> None:
        self.rows.append(
            {
                "step": int(step),
                "total_steps": int(total_steps),
                "alpha": float(alpha),
                "prescribed_delta_x_m": _optional_float(prescribed_delta_x),
                "prescribed_delta_x_mm": _optional_float(prescribed_delta_x, scale=1.0e3),
                "max_top_u_x_m": _optional_float(max_top_u_x),
                "max_top_u_x_mm": _optional_float(max_top_u_x, scale=1.0e3),
                "reaction_force_x_N": _optional_float(reaction_force_x),
                "reaction_force_x_uN": _optional_float(
                    reaction_force_x_uN if reaction_force_x_uN is not None else (
                        None if reaction_force_x is None else reaction_force_x * 1.0e6
                    )
                ),
                "newton_iterations": "" if newton_iterations is None else int(newton_iterations),
                "converged": bool(converged),
                "frame_xz_png": frame_path,
                "frame_3d_png": frame_3d_path,
            }
        )
        self.write_table()

    def _should_save_frame(self, step: int, total_steps: int) -> bool:
        return self.save_frames and (step % self.every == 0 or step == total_steps)

    def write_table(self) -> str:
        os.makedirs(os.path.dirname(self.table_path) or ".", exist_ok=True)
        fieldnames = [
            "step",
            "total_steps",
            "alpha",
            "prescribed_delta_x_m",
            "prescribed_delta_x_mm",
            "max_top_u_x_m",
            "max_top_u_x_mm",
            "reaction_force_x_N",
            "reaction_force_x_uN",
            "newton_iterations",
            "converged",
            "frame_xz_png",
            "frame_3d_png",
        ]
        with open(self.table_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return self.table_path

    def finalize(self) -> str | None:
        self.write_table()
        animation_frames = self.frames if self.frames else self.frames_3d
        if not self.make_animation or len(animation_frames) < 2:
            return None
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError:
            return None
        gif_path = os.path.join(self.outdir, "mechanics_deformation.gif")
        images = [imageio.imread(path) for path in animation_frames]
        imageio.mimsave(gif_path, images, duration=0.35)
        self.animation_path = gif_path
        return gif_path


def _optional_float(value, *, scale: float = 1.0):
    if value is None:
        return ""
    return float(value) * scale
