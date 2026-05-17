from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .specs import PlotSpec


def magnetics_plot_specs() -> list[PlotSpec]:
    return [
        PlotSpec(
            name="B_initial_vs_deformed",
            filename="magnetics_B_initial_vs_deformed.png",
            kind="magnetics",
            required_fields=("B0_sensor_norm_uT", "B1_sensor_norm_uT", "dB_sensor_norm_uT"),
            description="Initial and deformed magnetic response at Hall sensor.",
        ),
        PlotSpec(
            name="delta_B_components",
            filename="magnetics_delta_B_components.png",
            kind="magnetics",
            required_fields=("dB_sensor_x_uT", "dB_sensor_y_uT", "dB_sensor_z_uT"),
            description="Delta-B component comparison.",
        ),
    ]


def _float_value(data: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = data.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _str_value(data: Mapping[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if value in (None, ""):
        return default
    return str(value)


def _resolve_existing_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None
    raw = Path(raw_path)
    candidates = [raw, Path.cwd() / raw]
    if raw_path.startswith("../"):
        candidates.append(Path.cwd() / raw_path[3:])
        candidates.append(Path.cwd() / "magnetic_cilium_pipeline" / raw)
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


def _restart_path_from_result(result: Mapping[str, Any]) -> Path | None:
    for key in ("restart_file",):
        path = _resolve_existing_path(_str_value(result, key))
        if path and path.name == "mechanics_restart.npz":
            return path
    pvd_path = _resolve_existing_path(_str_value(result, "pvd_file"))
    if pvd_path is not None:
        restart = pvd_path.parent / "mechanics_restart.npz"
        if restart.exists():
            return restart
    return None


def _tet_volume(vertices):
    import numpy as np

    return abs(float(np.linalg.det(np.vstack([
        vertices[1] - vertices[0],
        vertices[2] - vertices[0],
        vertices[3] - vertices[0],
    ])))) / 6.0


def _unit(vector, fallback=None):
    import numpy as np

    vec = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-30:
        return np.asarray([0.0, 0.0, 1.0] if fallback is None else fallback, dtype=float)
    return vec / norm


def _angle_between_vectors_deg(left, right) -> float:
    import numpy as np

    left_u = _unit(left)
    right_u = _unit(right)
    return float(np.rad2deg(np.arccos(np.clip(float(np.dot(left_u, right_u)), -1.0, 1.0))))


def _rotation_from_F(F):
    import numpy as np

    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _cell_deformation_gradient(X, x_current):
    import numpy as np

    A = np.vstack([X[1] - X[0], X[2] - X[0], X[3] - X[0]]).T
    B = np.vstack([x_current[1] - x_current[0], x_current[2] - x_current[0], x_current[3] - x_current[0]]).T
    return B @ np.linalg.inv(A)


def _direction_from_theta(theta_rad: float, geometry_axis):
    import numpy as np

    ez = np.asarray([0.0, 0.0, 1.0], dtype=float)
    ex = np.asarray([1.0, 0.0, 0.0], dtype=float)
    axis = _unit(geometry_axis)
    transverse = axis - float(np.dot(axis, ez)) * ez
    transverse = _unit(transverse, fallback=ex)
    return _unit(np.cos(theta_rad) * ez + np.sin(theta_rad) * transverse)


def _magnetization_direction(F, result: Mapping[str, Any]):
    import numpy as np

    ez = np.asarray([0.0, 0.0, 1.0], dtype=float)
    model = _str_value(result, "magnetization_model", "fixed_global")
    geometry_axis = _unit(F @ ez)
    geometry_theta = float(np.arccos(np.clip(np.dot(geometry_axis, ez), -1.0, 1.0)))
    if model == "fixed_global":
        return ez.copy()
    if model == "rotate_with_material":
        return _unit(_rotation_from_F(F) @ ez)
    if model == "prescribed_theta_mu":
        return _direction_from_theta(_float_value(result, "theta_mu_rad", 0.0), geometry_axis)
    if model == "follow_factor":
        return _direction_from_theta(_float_value(result, "follow_factor_alpha", 1.0) * geometry_theta, geometry_axis)
    return _unit([
        _float_value(result, "magnetization_mean_unit_x"),
        _float_value(result, "magnetization_mean_unit_y"),
        _float_value(result, "magnetization_mean_unit_z", 1.0),
    ])


def _convex_hull_xz(points_xz):
    points = sorted({(float(x), float(z)) for x, z in points_xz})
    if len(points) <= 2:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _centerline_profile_xz(points, current_points, vertex_ids, n_sections: int = 160):
    import numpy as np

    ref = points[vertex_ids]
    current = current_points[vertex_ids]
    ref_z = ref[:, 2]
    if ref_z.size == 0:
        return None

    z_min = float(np.min(ref_z))
    z_max = float(np.max(ref_z))
    if abs(z_max - z_min) <= 1.0e-30:
        return None

    edges = np.linspace(z_min, z_max, n_sections + 1)
    section_ref_z = []
    section_centers = []
    section_points = []
    for idx in range(n_sections):
        if idx == n_sections - 1:
            mask = (ref_z >= edges[idx]) & (ref_z <= edges[idx + 1])
        else:
            mask = (ref_z >= edges[idx]) & (ref_z < edges[idx + 1])
        if int(np.count_nonzero(mask)) < 3:
            continue
        xz = current[mask][:, [0, 2]]
        section_ref_z.append(float(np.mean(ref_z[mask])))
        section_centers.append(np.mean(xz, axis=0))
        section_points.append(xz)

    if len(section_centers) < 2:
        return None

    centers = np.asarray(section_centers, dtype=float)
    tangents = np.zeros_like(centers)
    tangents[0] = centers[1] - centers[0]
    tangents[-1] = centers[-1] - centers[-2]
    if len(centers) > 2:
        tangents[1:-1] = centers[2:] - centers[:-2]

    normals = []
    half_widths = []
    adjusted_centers = []
    for center, tangent, xz in zip(centers, tangents, section_points):
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1.0e-30:
            tangent = np.asarray([0.0, 1.0], dtype=float)
        else:
            tangent = tangent / tangent_norm
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        projection = xz @ normal
        lower = float(np.min(projection))
        upper = float(np.max(projection))
        width = max(0.5 * (upper - lower), 1.0e-12)
        normal_center = 0.5 * (upper + lower)
        center = center + (normal_center - float(center @ normal)) * normal
        normals.append(normal)
        half_widths.append(width)
        adjusted_centers.append(center)

    return {
        "ref_z": np.asarray(section_ref_z, dtype=float),
        "centers": np.asarray(adjusted_centers, dtype=float),
        "normals": np.asarray(normals, dtype=float),
        "half_widths": np.asarray(half_widths, dtype=float),
    }


def _profile_outline_xz(profile, ref_z_min: float | None = None, ref_z_max: float | None = None):
    import numpy as np

    if profile is None:
        return []
    ref_z = profile["ref_z"]
    mask = np.ones_like(ref_z, dtype=bool)
    if ref_z_min is not None:
        mask &= ref_z >= ref_z_min
    if ref_z_max is not None:
        mask &= ref_z <= ref_z_max
    if int(np.count_nonzero(mask)) < 2:
        return []

    centers = profile["centers"][mask]
    normals = profile["normals"][mask]
    half_widths = profile["half_widths"][mask]
    left = centers + normals * half_widths[:, None]
    right = centers - normals * half_widths[:, None]
    outline = np.vstack([left, right[::-1]])
    return [(float(x), float(z)) for x, z in outline]


def _load_visual_geometry(result: Mapping[str, Any]):
    import numpy as np

    restart = _restart_path_from_result(result)
    if restart is None:
        return None
    data = np.load(restart)
    points = np.asarray(data["points"], dtype=float)[:, :3]
    cells = np.asarray(data["cells"], dtype=int)
    material = np.asarray(data["material_cell_ids"], dtype=int)
    u_vertices = np.asarray(data["u_vertices"], dtype=float)[:, :3]
    current_points = points + u_vertices

    cilium_vertex_ids = np.unique(cells[np.isin(material, [1, 2])].reshape(-1))
    magnetic_mask = material == 2
    magnetic_vertex_ids = np.unique(cells[magnetic_mask].reshape(-1))
    cilium_hull = _convex_hull_xz(current_points[cilium_vertex_ids][:, [0, 2]])
    magnetic_hull = _convex_hull_xz(current_points[magnetic_vertex_ids][:, [0, 2]])
    cilium_profile = _centerline_profile_xz(points, current_points, cilium_vertex_ids)
    magnetic_ref_z = points[magnetic_vertex_ids, 2]
    magnetic_ref_z_min = float(np.min(magnetic_ref_z))
    magnetic_ref_z_max = float(np.max(magnetic_ref_z))
    cilium_outline = _profile_outline_xz(cilium_profile)
    magnetic_outline = _profile_outline_xz(cilium_profile, magnetic_ref_z_min, magnetic_ref_z_max)

    magnetic_cells = cells[magnetic_mask]
    centers = []
    volumes = []
    directions = []
    ref_z = []
    for cell in magnetic_cells:
        X = points[cell]
        x_cell = current_points[cell]
        volume = _tet_volume(x_cell)
        if volume <= 0.0:
            continue
        F = _cell_deformation_gradient(X, x_cell)
        centers.append(x_cell.mean(axis=0))
        volumes.append(volume)
        directions.append(_magnetization_direction(F, result))
        ref_z.append(float(X[:, 2].mean()))

    return {
        "restart": restart,
        "points": points,
        "current_points": current_points,
        "magnetic_cells": magnetic_cells,
        "magnetic_vertex_ids": magnetic_vertex_ids,
        "cilium_hull": cilium_hull,
        "magnetic_hull": magnetic_hull,
        "cilium_outline": cilium_outline or cilium_hull,
        "magnetic_outline": magnetic_outline or magnetic_hull,
        "centers": np.asarray(centers, dtype=float),
        "volumes": np.asarray(volumes, dtype=float),
        "directions": np.asarray(directions, dtype=float),
        "ref_z": np.asarray(ref_z, dtype=float),
    }


def _dipole_vectors_for_plot(result: Mapping[str, Any], geometry):
    import numpy as np

    fallback_point = np.asarray([
        _float_value(result, "magnetization_moment_center_x_m"),
        _float_value(result, "magnetization_moment_center_y_m"),
        _float_value(result, "magnetization_moment_center_z_m"),
    ], dtype=float)
    fallback_direction = _unit([
        _float_value(result, "magnetization_mean_unit_x"),
        _float_value(result, "magnetization_mean_unit_y"),
        _float_value(result, "magnetization_mean_unit_z", 1.0),
    ])
    if geometry is None or geometry["centers"].size == 0:
        return fallback_point.reshape(1, 3), fallback_direction.reshape(1, 3)

    mode = _str_value(result, "dipole_mode", "tetrahedral")
    centers = geometry["centers"]
    volumes = geometry["volumes"]
    directions = geometry["directions"]
    if mode == "tetrahedral":
        return fallback_point.reshape(1, 3), fallback_direction.reshape(1, 3)

    if mode == "tip_single_dipole":
        points = geometry["points"]
        current = geometry["current_points"]
        vertex_ids = geometry["magnetic_vertex_ids"]
        ref_vertices = points[vertex_ids]
        z_max = float(np.max(ref_vertices[:, 2]))
        z_span = float(np.ptp(ref_vertices[:, 2]))
        tol = max(1.0e-12, 1.0e-6 * max(z_span, 1.0e-12))
        tip_ids = vertex_ids[np.abs(ref_vertices[:, 2] - z_max) <= tol]
        point = current[tip_ids].mean(axis=0) if tip_ids.size else fallback_point
        direction = _unit(np.sum(directions * volumes[:, None], axis=0), fallback=fallback_direction)
        return point.reshape(1, 3), direction.reshape(1, 3)

    n_points = max(1, int(_float_value(result, "n_point_dipoles", 8)))
    ref_z = geometry["ref_z"]
    z_min = float(np.min(ref_z))
    z_max = float(np.max(ref_z))
    if abs(z_max - z_min) <= 1.0e-30:
        bins = np.zeros_like(ref_z, dtype=int)
        n_bins = 1
    else:
        edges = np.linspace(z_min, z_max, n_points + 1)
        bins = np.searchsorted(edges, ref_z, side="right") - 1
        bins = np.clip(bins, 0, n_points - 1)
        n_bins = n_points

    plot_points = []
    plot_directions = []
    for bin_id in range(n_bins):
        mask = bins == bin_id
        if not np.any(mask):
            continue
        weights = volumes[mask]
        plot_points.append(np.average(centers[mask], axis=0, weights=weights))
        plot_directions.append(_unit(np.sum(directions[mask] * weights[:, None], axis=0), fallback=fallback_direction))
    return np.asarray(plot_points, dtype=float), np.asarray(plot_directions, dtype=float)


def plot_magnetization_hall_schematic(result: Mapping[str, Any], output_base_path: str) -> tuple[str, str]:
    """Save a Russian-labeled schematic of net magnetization and Hall sensor."""
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon, Rectangle

    png_path = output_base_path + ".png"
    svg_path = output_base_path + ".svg"
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    geometry = _load_visual_geometry(result)
    dipole_points, dipole_dirs = _dipole_vectors_for_plot(result, geometry)

    sensor = (
        _float_value(result, "sensor_x_m"),
        _float_value(result, "sensor_z_m"),
    )
    segment_min = (
        _float_value(result, "magnetic_segment_min_x_m", _float_value(result, "magnetization_moment_center_x_m")),
        _float_value(result, "magnetic_segment_min_z_m", _float_value(result, "magnetization_moment_center_z_m")),
    )
    segment_max = (
        _float_value(result, "magnetic_segment_max_x_m", _float_value(result, "magnetization_moment_center_x_m")),
        _float_value(result, "magnetic_segment_max_z_m", _float_value(result, "magnetization_moment_center_z_m")),
    )
    segment_center = (
        _float_value(result, "magnetic_segment_centroid_x_m", _float_value(result, "magnetization_moment_center_x_m")),
        _float_value(result, "magnetic_segment_centroid_z_m", _float_value(result, "magnetization_moment_center_z_m")),
    )
    model = str(result.get("magnetization_model", ""))
    mode = str(result.get("dipole_mode", ""))

    x_values = [sensor[0], segment_min[0], segment_max[0], segment_center[0], *dipole_points[:, 0].tolist()]
    z_values = [sensor[1], segment_min[1], segment_max[1], segment_center[1], *dipole_points[:, 2].tolist()]
    if geometry is not None:
        x_values.extend([point[0] for point in geometry["cilium_outline"]])
        z_values.extend([point[1] for point in geometry["cilium_outline"]])

    raw_span = max(max(x_values) - min(x_values), max(z_values) - min(z_values), 1.0e-6)
    if mode == "tip_single_dipole":
        arrow_len = 0.12 * raw_span
    elif len(dipole_points) > 1:
        points_xz = dipole_points[:, [0, 2]]
        ordered = points_xz[np.argsort(points_xz[:, 1])]
        spacing = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
        min_spacing = float(np.min(spacing)) if spacing.size else raw_span
        arrow_len = min(0.055 * raw_span, 0.50 * min_spacing)
    else:
        arrow_len = 0.15 * raw_span

    arrow_end_points = dipole_points + dipole_dirs * arrow_len
    x_values.extend(arrow_end_points[:, 0].tolist())
    z_values.extend(arrow_end_points[:, 2].tolist())
    z_values.append(sensor[1] - 0.12 * raw_span)

    span = max(max(x_values) - min(x_values), max(z_values) - min(z_values), 1.0e-6)
    x_pad = 0.34 * span
    z_pad = 0.18 * span
    scale = 1.0e6

    fig, ax = plt.subplots(figsize=(15.5, 8.8), dpi=170)
    ax.set_aspect("equal", adjustable="box")

    if geometry is not None and len(geometry["cilium_outline"]) >= 3:
        ax.add_patch(
            Polygon(
                [(x * scale, z * scale) for x, z in geometry["cilium_outline"]],
                closed=True,
                facecolor="none",
                edgecolor="#111827",
                linewidth=1.5,
                linestyle=(0, (5, 3)),
                label="контур деформированной реснички",
            )
        )
    if geometry is not None and len(geometry["magnetic_outline"]) >= 3:
        ax.add_patch(
            Polygon(
                [(x * scale, z * scale) for x, z in geometry["magnetic_outline"]],
                closed=True,
                facecolor="#8fbce6",
                edgecolor="#1f5f99",
                linewidth=1.5,
                alpha=0.38,
                label="магнитный сегмент",
            )
        )
    else:
        seg_width = max(segment_max[0] - segment_min[0], 0.08 * span)
        seg_height = max(segment_max[1] - segment_min[1], 0.08 * span)
        seg_x = 0.5 * (segment_min[0] + segment_max[0]) - 0.5 * seg_width
        seg_z = 0.5 * (segment_min[1] + segment_max[1]) - 0.5 * seg_height
        ax.add_patch(
            Rectangle(
                (seg_x * scale, seg_z * scale),
                seg_width * scale,
                seg_height * scale,
                facecolor="#8fbce6",
                edgecolor="#1f5f99",
                linewidth=1.6,
                alpha=0.45,
                label="магнитный сегмент",
            )
        )

    sensor_w = 0.22 * span
    sensor_h = 0.09 * span
    ax.add_patch(
        Rectangle(
            ((sensor[0] - 0.5 * sensor_w) * scale, (sensor[1] - 0.5 * sensor_h) * scale),
            sensor_w * scale,
            sensor_h * scale,
            facecolor="#f7c873",
            edgecolor="#9a6a13",
            linewidth=1.6,
            label="датчик Холла",
        )
    )

    for idx, (point, direction) in enumerate(zip(dipole_points, dipole_dirs)):
        ax.scatter(point[0] * scale, point[2] * scale, s=18, color="#b21f2d", zorder=5)
        moment_width = (0.0023 if len(dipole_points) > 1 else 0.0032) * span * scale
        moment_head_width = min(0.022 * span, 0.22 * arrow_len) * scale
        moment_head_length = min(0.035 * span, 0.35 * arrow_len) * scale
        ax.arrow(
            point[0] * scale,
            point[2] * scale,
            direction[0] * arrow_len * scale,
            direction[2] * arrow_len * scale,
            width=moment_width,
            head_width=moment_head_width,
            head_length=moment_head_length,
            color="#b21f2d",
            length_includes_head=True,
            label="вектор магнитного момента" if idx == 0 else "_nolegend_",
            zorder=6,
        )

    field_start = np.mean(dipole_points, axis=0)
    ax.arrow(
        field_start[0] * scale,
        field_start[2] * scale,
        (sensor[0] - field_start[0]) * scale,
        (sensor[1] - field_start[2]) * scale,
        width=0.0018 * span * scale,
        head_width=0.026 * span * scale,
        head_length=0.040 * span * scale,
        color="#2563eb",
        alpha=0.95,
        linestyle="-",
        length_includes_head=True,
        label="поле на датчике",
    )

    ax.text(
        sensor[0] * scale,
        sensor[1] * scale,
        "датчик Холла",
        ha="center",
        va="center",
        fontsize=9,
        color="#5c3a06",
    )
    ax.text(
        (dipole_points[0, 0] + 0.60 * dipole_dirs[0, 0] * arrow_len) * scale,
        (dipole_points[0, 2] + 0.60 * dipole_dirs[0, 2] * arrow_len) * scale,
        "Σm",
        color="#b21f2d",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        0.5 * (field_start[0] + sensor[0]) * scale,
        0.5 * (field_start[2] + sensor[1]) * scale,
        "поле на датчике",
        color="#1d4ed8",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )

    ax.set_title(f"Суммарная намагниченность и датчик Холла\nмодель: {model}, диполи: {mode}")
    ax.set_xlabel("x, мкм")
    ax.set_ylabel("z, мкм")
    ax.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.7)
    ax.set_xlim((min(x_values) - x_pad) * scale, (max(x_values) + x_pad) * scale)
    ax.set_ylim((min(z_values) - z_pad) * scale, (max(z_values) + z_pad) * scale)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(
            unique.values(),
            unique.keys(),
            loc="upper left",
            bbox_to_anchor=(1.18, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
    fig.tight_layout(rect=(0.0, 0.0, 0.64, 1.0))
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return png_path, svg_path


def plot_magnetization_angle_zoom(
    result: Mapping[str, Any],
    output_base_path: str,
    *,
    angle_threshold_deg: float = 1.0e-6,
) -> tuple[str, str] | None:
    """Save a zoomed Russian-labeled angle schematic for nonzero M-to-normal angles."""
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Arc

    geometry = _load_visual_geometry(result)
    dipole_points, dipole_dirs = _dipole_vectors_for_plot(result, geometry)
    fallback_dir = _unit(np.mean(dipole_dirs, axis=0), fallback=[0.0, 0.0, 1.0])
    moment_dir = _unit(
        [
            _float_value(result, "magnetization_mean_unit_x", fallback_dir[0]),
            _float_value(result, "magnetization_mean_unit_y", fallback_dir[1]),
            _float_value(result, "magnetization_mean_unit_z", fallback_dir[2]),
        ],
        fallback=fallback_dir,
    )
    sensor_normal = _unit(
        [
            _float_value(result, "sensor_normal_unit_x", 0.0),
            _float_value(result, "sensor_normal_unit_y", 0.0),
            _float_value(result, "sensor_normal_unit_z", 1.0),
        ],
        fallback=[0.0, 0.0, 1.0],
    )
    angle_value = _float_value(
        result,
        "dipole_magnetization_angle_to_normal_mean_deg",
        _float_value(result, "theta_mu_to_global_z_mean_deg", _angle_between_vectors_deg(moment_dir, sensor_normal)),
    )
    if abs(angle_value) <= angle_threshold_deg:
        return None

    origin = np.asarray(
        [
            _float_value(result, "magnetization_moment_center_x_m", float(np.mean(dipole_points[:, 0]))),
            _float_value(result, "magnetization_moment_center_y_m", float(np.mean(dipole_points[:, 1]))),
            _float_value(result, "magnetization_moment_center_z_m", float(np.mean(dipole_points[:, 2]))),
        ],
        dtype=float,
    )

    png_path = output_base_path + ".png"
    svg_path = output_base_path + ".svg"
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)

    scale = 1.0e6
    vector_len = 520.0e-6
    normal_end = origin + sensor_normal * vector_len
    moment_end = origin + moment_dir * vector_len

    normal_xz = _unit([sensor_normal[0], sensor_normal[2]], fallback=[0.0, 1.0])
    moment_xz = _unit([moment_dir[0], moment_dir[2]], fallback=[0.0, 1.0])
    normal_angle = float(np.rad2deg(np.arctan2(normal_xz[1], normal_xz[0])))
    moment_angle = float(np.rad2deg(np.arctan2(moment_xz[1], moment_xz[0])))
    delta_angle = ((moment_angle - normal_angle + 180.0) % 360.0) - 180.0

    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=180)
    ax.set_aspect("equal", adjustable="box")
    ax.scatter(origin[0] * scale, origin[2] * scale, s=28, color="#111827", zorder=5)
    ax.plot(
        [origin[0] * scale, normal_end[0] * scale],
        [origin[2] * scale, normal_end[2] * scale],
        color="#15803d",
        linewidth=2.0,
        linestyle=(0, (4, 3)),
        label="нормаль датчика (+z)",
        zorder=4,
    )
    ax.scatter(normal_end[0] * scale, normal_end[2] * scale, marker="^", s=54, color="#15803d", zorder=5)
    ax.arrow(
        origin[0] * scale,
        origin[2] * scale,
        (moment_end[0] - origin[0]) * scale,
        (moment_end[2] - origin[2]) * scale,
        width=0.006 * vector_len * scale,
        head_width=0.06 * vector_len * scale,
        head_length=0.08 * vector_len * scale,
        color="#b21f2d",
        length_includes_head=True,
        label="вектор намагниченности",
        zorder=6,
    )

    arc_radius = 0.40 * vector_len * scale
    theta1 = normal_angle
    theta2 = normal_angle + delta_angle
    if theta2 < theta1:
        theta1, theta2 = theta2, theta1
    ax.add_patch(
        Arc(
            (origin[0] * scale, origin[2] * scale),
            2.0 * arc_radius,
            2.0 * arc_radius,
            angle=0.0,
            theta1=theta1,
            theta2=theta2,
            color="#15803d",
            linewidth=2.3,
            zorder=7,
        )
    )
    label_angle = normal_angle + 0.5 * delta_angle
    label_radius = 1.34 * arc_radius
    label_angle_rad = np.deg2rad(label_angle)
    ax.text(
        origin[0] * scale + np.cos(label_angle_rad) * label_radius,
        origin[2] * scale + np.sin(label_angle_rad) * label_radius,
        f"θμ к нормали = {angle_value:.2f}°",
        color="#15803d",
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.4},
    )

    model = str(result.get("magnetization_model", ""))
    mode = str(result.get("dipole_mode", ""))
    ax.set_title(f"Угол намагниченности к нормали датчика\nмодель: {model}, диполи: {mode}")
    ax.set_xlabel("x, мкм")
    ax.set_ylabel("z, мкм")
    ax.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.7)
    x_points = [origin[0], normal_end[0], moment_end[0]]
    z_points = [origin[2], normal_end[2], moment_end[2]]
    margin = 0.25 * vector_len
    ax.set_xlim((min(x_points) - margin) * scale, (max(x_points) + 2.3 * margin) * scale)
    ax.set_ylim((min(z_points) - margin) * scale, (max(z_points) + margin) * scale)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.76, 1.0))
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return png_path, svg_path
