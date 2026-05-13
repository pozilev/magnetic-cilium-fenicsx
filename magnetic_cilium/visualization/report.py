from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


log = logging.getLogger(__name__)

TARGET_FORCE_U_N = 60.0
FORCE_PASS_TOLERANCE_PERCENT = 2.0

FIGURE_FILENAMES = {
    "mechanics_card": "mechanics_summary_card",
    "deformed_overview": "deformed_shape_overview",
    "deformed_closeup": "deformed_shape_closeup",
    "von_mises_overview": "von_mises_overview",
    "von_mises_base_zoom": "von_mises_base_zoom",
    "jacobian_overview": "jacobian_J_overview",
    "jacobian_base_zoom": "jacobian_J_base_zoom",
    "panel": "mechanics_panel",
    "delta_b": "deltaB_components",
}

RU = {
    "report_title": "Визуальный отчёт по расчёту магнитной реснички",
    "mechanics_card_title": "Сводка механического расчёта",
    "mechanics_question": "Попадание в целевую реакцию порядка 60 мкН",
    "status_pass": "ПРОЙДЕНО",
    "status_check": "ПРОВЕРИТЬ",
    "delta_b_title": "Компоненты изменения магнитного поля в точке датчика Холла",
    "field_component_axis": "Компонента поля",
    "delta_b_axis": "Изменение магнитного поля, мкТл",
    "dominant_component": "Наиболее информативная компонента",
    "deformed_title": "Деформированная форма реснички",
    "von_mises_title": "Эквивалентное напряжение von Mises",
    "jacobian_title": "Якобиан деформации J",
    "mechanics_panel_title": "Механическая корректность расчёта",
    "x_axis": "x, мм",
    "y_axis": "y, мм",
    "z_axis": "z, мм",
    "ux_colorbar": "Перемещение по x, мкм",
    "vm_colorbar": "Напряжение von Mises, кПа",
    "j_colorbar": "J − 1, ×10⁻³",
    "lower_layer": "нижний PDMS-сегмент",
    "upper_layer": "верхний магнитный композит",
}

FIELD_ALIASES = {
    "delta_x_m": ("delta_x_m", "delta_x", "delta", "max_top_u_x_m"),
    "reaction_force_x_N": ("reaction_force_x_N",),
    "reaction_force_x_uN": ("reaction_force_x_uN", "reaction_force_x_u_n"),
    "reaction_error_percent": ("reaction_error_to_60uN_percent", "reaction_error_percent"),
    "J_min": ("J_min", "j_min", "detF_min", "jacobian_min"),
    "J_max": ("J_max", "j_max", "detF_max", "jacobian_max"),
    "von_mises_max_Pa": ("von_mises_max_Pa", "von_mises_DG0_max_Pa", "stress_max_Pa"),
    "element_degree": ("element_degree", "degree"),
    "h_cilium_m": ("h_cilium_m", "h_cilium"),
    "h_substrate_m": ("h_substrate_m", "h_substrate"),
    "nu_pdms": ("nu_pdms", "nu", "poisson_pdms"),
    "nu_magnetic": ("nu_magnetic", "poisson_magnetic"),
    "sensor_x_m": ("sensor_x_m", "sensor_x"),
    "sensor_y_m": ("sensor_y_m", "sensor_y"),
    "sensor_z_m": ("sensor_z_m", "sensor_z"),
    "Br_T": ("Br_magnetic_T", "Br_T", "Br"),
    "D_m": ("D", "diameter_m"),
    "R_m": ("R", "radius_m"),
    "L1_m": ("L1", "lower_length_m", "L1_m"),
    "L2_m": ("L2", "magnetic_length_m", "L2_m"),
}

DELTA_B_ALIASES = {
    "x": (
        ("dB_sensor_x_uT", 1.0),
        ("dBx_uT", 1.0),
        ("abs_dBx_uT", 1.0),
        ("dB_sensor_x_T", 1.0e6),
        ("delta_B_x_T", 1.0e6),
        ("dBx_T", 1.0e6),
    ),
    "y": (
        ("dB_sensor_y_uT", 1.0),
        ("dBy_uT", 1.0),
        ("abs_dBy_uT", 1.0),
        ("dB_sensor_y_T", 1.0e6),
        ("delta_B_y_T", 1.0e6),
        ("dBy_T", 1.0e6),
    ),
    "z": (
        ("dB_sensor_z_uT", 1.0),
        ("dBz_uT", 1.0),
        ("abs_dBz_uT", 1.0),
        ("dB_sensor_z_T", 1.0e6),
        ("delta_B_z_T", 1.0e6),
        ("dBz_T", 1.0e6),
    ),
}

VTK_FIELD_ALIASES = {
    "displacement": ("displacement", "u", "Displacement", "displacement vector"),
    "von_mises": ("von_mises_DG0", "von_mises", "von_Mises", "stress", "sigma_vm"),
    "J": ("J_detF", "J", "detF", "jacobian", "Jacobian"),
}


@dataclass(frozen=True)
class ResultFiles:
    result_dir: Path
    mechanics_json: Path | None = None
    params_json: Path | None = None
    summary_csv: Path | None = None
    magnetic_csv: Path | None = None
    pvd_file: Path | None = None
    restart_npz: Path | None = None
    vtu_files: tuple[Path, ...] = ()
    log_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class FigureRecord:
    key: str
    title: str
    png_path: Path
    pdf_path: Path | None = None


@dataclass
class VisualReportResult:
    outdir: Path
    figures: list[FigureRecord] = field(default_factory=list)
    report_md: Path | None = None
    report_html: Path | None = None
    files: ResultFiles | None = None
    diagnostics: list[str] = field(default_factory=list)
    missing_fields: dict[str, list[str]] = field(default_factory=dict)
    mechanics: dict[str, Any] = field(default_factory=dict)
    magnetic_row: dict[str, Any] | None = None


def setup_visual_logging(outdir: Path, verbose: bool = False) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "visual_report.log"
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    stream_handler.setLevel(level)

    for handler in list(root.handlers):
        if getattr(handler, "_magnetic_cilium_visual_report", False):
            root.removeHandler(handler)
    file_handler._magnetic_cilium_visual_report = True  # type: ignore[attr-defined]
    stream_handler._magnetic_cilium_visual_report = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    return log_path


def discover_result_files(result_dir: Path) -> ResultFiles:
    result_dir = Path(result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Директория результатов не найдена: {result_dir}")

    mechanics_json = _select_candidate(result_dir.rglob("mechanics_result.json"), result_dir, _score_mechanics_json)
    params_json = None
    restart_npz = None
    if mechanics_json is not None:
        params_json = _select_candidate([mechanics_json.parent / "params.json"], result_dir, _score_params_json)
        restart_npz = _select_candidate([mechanics_json.parent / "mechanics_restart.npz"], result_dir, _score_restart_npz)

    if params_json is None:
        params_json = _select_candidate(result_dir.rglob("params.json"), result_dir, _score_params_json)
    if restart_npz is None:
        restart_npz = _select_candidate(result_dir.rglob("mechanics_restart.npz"), result_dir, _score_restart_npz)

    csv_candidates = _unique_paths(
        list(result_dir.glob("*.csv"))
        + list(result_dir.rglob("*.csv"))
        + list(result_dir.parent.glob("summary.csv"))
        + list(result_dir.parent.glob("*summary*.csv"))
        + list(result_dir.parent.glob("magnetic*.csv"))
    )
    summary_csv = _select_candidate(csv_candidates, result_dir, _score_summary_csv)
    magnetic_csv = _select_candidate(
        [p for p in csv_candidates if _csv_has_any(p, ("dB", "dBx", "norm_dB", "delta_B"))],
        result_dir,
        _score_magnetic_csv,
    )
    if magnetic_csv is None:
        magnetic_csv = summary_csv if summary_csv and _csv_has_any(summary_csv, ("dB", "dBx", "norm_dB", "delta_B")) else None

    pvd_candidates: list[Path] = []
    if mechanics_json is not None:
        pvd_candidates.extend(mechanics_json.parent.glob("*.pvd"))
    pvd_candidates.extend(result_dir.rglob("*.pvd"))
    pvd_file = _select_candidate(_unique_paths(pvd_candidates), result_dir, _score_pvd)

    vtu_candidates: list[Path] = []
    if pvd_file is not None:
        vtu_candidates.extend(pvd_file.parent.glob("*.vtu"))
    if mechanics_json is not None:
        vtu_candidates.extend(mechanics_json.parent.glob("*.vtu"))
    vtu_candidates.extend(result_dir.rglob("*.vtu"))

    log_files = tuple(
        sorted(
            path
            for path in _unique_paths(list(result_dir.glob("*.log")) + list(result_dir.rglob("*.log")))
            if "visual_report" not in {part.lower() for part in path.parts}
        )
    )
    files = ResultFiles(
        result_dir=result_dir,
        mechanics_json=mechanics_json,
        params_json=params_json,
        summary_csv=summary_csv,
        magnetic_csv=magnetic_csv,
        pvd_file=pvd_file,
        restart_npz=restart_npz,
        vtu_files=tuple(sorted(_unique_paths(vtu_candidates))),
        log_files=log_files,
    )
    log.info("Выбран mechanics JSON: %s", mechanics_json)
    log.info("Выбран params JSON: %s", params_json)
    log.info("Выбран summary CSV: %s", summary_csv)
    log.info("Выбран магнитный CSV: %s", magnetic_csv)
    log.info("Выбран PVD: %s", pvd_file)
    log.info("Выбран restart NPZ: %s", restart_npz)
    return files


def load_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    if "mechanics" in payload and isinstance(payload["mechanics"], dict):
        mechanics = payload["mechanics"]
        values = mechanics.get("values")
        if isinstance(values, dict):
            return dict(values)
    if "values" in payload and isinstance(payload["values"], dict):
        return dict(payload["values"])
    return dict(payload)


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def select_baseline_magnetic_row(rows: Sequence[Mapping[str, Any]], mechanics: Mapping[str, Any]) -> dict[str, Any] | None:
    rows_with_delta = [dict(row) for row in rows if all(component_value_uT(row, c) is not None for c in ("x", "y", "z"))]
    if not rows_with_delta:
        return None
    if len(rows_with_delta) == 1:
        return rows_with_delta[0]

    mechanics_pvd = str(first_existing_value(mechanics, ("pvd_file",)) or "")
    mechanics_restart = str(first_existing_value(mechanics, ("restart_file",)) or "")

    def score(row: Mapping[str, Any], index: int) -> tuple[int, int]:
        text = " ".join(str(row.get(key, "")) for key in ("study", "mode", "experiment_id", "case_id")).lower()
        row_paths = " ".join(str(row.get(key, "")) for key in ("pvd_file", "restart_file", "restart_path", "mechanics_result_path"))
        value = 0
        if any(token in text for token in ("baseline", "final", "candidate", "single")):
            value += 40
        if _truthy(row.get("under_cilium_sensor_case_ok")):
            value += 35
        if _truthy(row.get("is_valid")) or _truthy(row.get("reliable")) or _truthy(row.get("solver_success")):
            value += 25
        if mechanics_pvd and mechanics_pvd in row_paths:
            value += 20
        if mechanics_restart and mechanics_restart in row_paths:
            value += 20
        if _as_float(row.get("rank_by_deltaB_norm")) == 1.0:
            value -= 20
        return value, index

    best_index, best = max(enumerate(rows_with_delta), key=lambda item: score(item[1], item[0]))
    log.info("Выбрана строка магнитного отклика #%d из %d без оптимизации sensor_x/sensor_y", best_index + 1, len(rows_with_delta))
    return best


def first_existing_value(mapping: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    for key in aliases:
        value = mapping.get(key)
        if value not in ("", None):
            return value
    return None


def numeric_value(mapping: Mapping[str, Any], aliases: Iterable[str]) -> float | None:
    return _as_float(first_existing_value(mapping, aliases))


def unit_value(mapping: Mapping[str, Any], canonical_name: str, scale: float = 1.0) -> float | None:
    value = numeric_value(mapping, FIELD_ALIASES[canonical_name])
    return None if value is None else value * scale


def force_error_percent(mechanics: Mapping[str, Any]) -> float | None:
    explicit = numeric_value(mechanics, FIELD_ALIASES["reaction_error_percent"])
    if explicit is not None:
        return explicit
    reaction_uN = reaction_force_uN(mechanics)
    if reaction_uN is None:
        return None
    return 100.0 * (reaction_uN - TARGET_FORCE_U_N) / TARGET_FORCE_U_N


def reaction_force_uN(mechanics: Mapping[str, Any]) -> float | None:
    reaction_uN = numeric_value(mechanics, FIELD_ALIASES["reaction_force_x_uN"])
    if reaction_uN is not None:
        return reaction_uN
    reaction_N = numeric_value(mechanics, FIELD_ALIASES["reaction_force_x_N"])
    return None if reaction_N is None else reaction_N * 1.0e6


def component_value_uT(row: Mapping[str, Any], component: str) -> float | None:
    for key, scale in DELTA_B_ALIASES[component]:
        value = _as_float(row.get(key))
        if value is not None:
            if key.startswith("abs_"):
                source = _signed_delta_alias_for_abs(component, row)
                return source if source is not None else value
            return value * scale
    return None


def magnetic_components_uT(row: Mapping[str, Any]) -> dict[str, float] | None:
    components = {component: component_value_uT(row, component) for component in ("x", "y", "z")}
    if any(value is None for value in components.values()):
        return None
    return {key: float(value) for key, value in components.items() if value is not None}


def magnetic_norm_uT(row: Mapping[str, Any], components: Mapping[str, float]) -> float:
    explicit = numeric_value(
        row,
        (
            "dB_sensor_norm_uT",
            "norm_dB_uT",
            "deltaB_norm_uT",
            "dB_norm_uT",
        ),
    )
    if explicit is not None:
        return explicit
    explicit_T = numeric_value(row, ("dB_sensor_norm_T", "norm_dB_T", "deltaB_norm_T", "delta_B_norm_T"))
    if explicit_T is not None:
        return explicit_T * 1.0e6
    return math.sqrt(sum(value * value for value in components.values()))


def dominant_component(components: Mapping[str, float]) -> str:
    key = max(components, key=lambda component: abs(components[component]))
    return {"x": "ΔBx", "y": "ΔBy", "z": "ΔBz"}[key]


def plot_mechanics_summary_card(mechanics: Mapping[str, Any], outdir: Path) -> FigureRecord:
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_axis_off()
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#f7f8fa")

    reaction_uN = reaction_force_uN(mechanics)
    error_percent = force_error_percent(mechanics)
    passed = error_percent is not None and abs(error_percent) <= FORCE_PASS_TOLERANCE_PERCENT
    status = RU["status_pass"] if passed else RU["status_check"]
    status_color = "#2f7d57" if passed else "#9a6a1f"
    status_bg = "#dfeee6" if passed else "#f3ead7"

    ax.text(0.05, 0.92, RU["mechanics_card_title"], fontsize=22, weight="bold", color="#1f2933", transform=ax.transAxes)
    ax.text(
        0.78,
        0.92,
        status,
        fontsize=15,
        weight="bold",
        color=status_color,
        ha="center",
        va="center",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.45,rounding_size=0.08", "facecolor": status_bg, "edgecolor": status_color, "linewidth": 1.2},
    )

    rows = [
        ("Заданное смещение вершины/верхней границы", _format_unit(unit_value(mechanics, "delta_x_m", 1.0e3), "мм", 3)),
        ("Реакция по x", _format_unit(reaction_uN, "мкН", 2)),
        ("Целевая сила", "60 мкН"),
        ("Относительная ошибка относительно 60 мкН", _format_unit(error_percent, "%", 2)),
        ("Минимальное и максимальное значение J", _format_range(numeric_value(mechanics, FIELD_ALIASES["J_min"]), numeric_value(mechanics, FIELD_ALIASES["J_max"]), "", 6)),
        ("Максимальное напряжение von Mises", _format_unit(unit_value(mechanics, "von_mises_max_Pa", 1.0e-3), "кПа", 2)),
        ("Степень конечного элемента", _format_plain(numeric_value(mechanics, FIELD_ALIASES["element_degree"]), 0)),
        ("Размеры сетки h", _format_mesh_sizes(mechanics)),
        ("Коэффициенты Пуассона материалов", _format_poisson(mechanics)),
    ]

    y = 0.78
    for label, value in rows:
        ax.text(0.07, y, label, fontsize=12.5, color="#52616f", transform=ax.transAxes)
        ax.text(0.58, y, value, fontsize=13.5, weight="bold", color="#1f2933", transform=ax.transAxes)
        ax.plot([0.07, 0.93], [y - 0.035, y - 0.035], color="#d8dee6", linewidth=0.8, transform=ax.transAxes)
        y -= 0.075

    conclusion = _force_conclusion(reaction_uN, error_percent, passed)
    ax.text(0.07, 0.08, RU["mechanics_question"], fontsize=12.0, weight="bold", color="#334e68", transform=ax.transAxes)
    ax.text(0.07, 0.035, conclusion, fontsize=12.0, color="#334e68", transform=ax.transAxes)
    return save_figure(fig, outdir, FIGURE_FILENAMES["mechanics_card"], RU["mechanics_card_title"], "mechanics_card")


def plot_delta_b_components(row: Mapping[str, Any], outdir: Path) -> FigureRecord:
    plt = _require_matplotlib()
    components = magnetic_components_uT(row)
    if components is None:
        raise ValueError("Не найдены компоненты ΔB в CSV-строке.")

    labels = ["ΔBx", "ΔBy", "ΔBz"]
    values = [components["x"], components["y"], components["z"]]
    norm = magnetic_norm_uT(row, components)
    dominant = dominant_component(components)

    colors = ["#3b82a0", "#5d8f61", "#b87739"]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bars = ax.bar(labels, values, color=colors, width=0.58, edgecolor="#26323d", linewidth=0.7)
    ax.axhline(0.0, color="#26323d", linewidth=1.0)
    ax.set_title(RU["delta_b_title"], fontsize=16, weight="bold", pad=16)
    ax.set_xlabel(RU["field_component_axis"], fontsize=12)
    ax.set_ylabel(RU["delta_b_axis"], fontsize=12)
    ax.grid(axis="y", color="#d9dee6", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)

    span = max(abs(v) for v in values) if values else 1.0
    if span <= 0.0:
        span = 1.0
    ax.set_ylim(min(values + [0.0]) - 0.28 * span, max(values + [0.0]) + 0.38 * span)

    for bar, value in zip(bars, values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.04 * span if value >= 0 else -0.04 * span
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            f"{value:.3g} мкТл",
            ha="center",
            va=va,
            fontsize=11,
            color="#1f2933",
        )

    info = f"|ΔB| = {norm:.3g} мкТл\n{RU['dominant_component']}: {dominant}"
    ax.text(
        0.98,
        0.95,
        info,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=11.5,
        color="#1f2933",
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.08", "facecolor": "#f7f8fa", "edgecolor": "#cbd5df"},
    )
    fig.tight_layout()
    return save_figure(fig, outdir, FIGURE_FILENAMES["delta_b"], RU["delta_b_title"], "delta_b")


def plot_mechanics_3d(
    files: ResultFiles,
    mechanics: Mapping[str, Any],
    outdir: Path,
) -> tuple[list[FigureRecord], dict[str, list[str]], list[str]]:
    diagnostics: list[str] = []
    missing: dict[str, list[str]] = {}
    try:
        import pyvista  # noqa: F401  # type: ignore
    except ModuleNotFoundError:
        diagnostics.append("PyVista недоступна; использован fallback-режим рендеринга matplotlib.")

    try:
        np = _require_numpy()
        plt = _require_matplotlib()
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
    except ModuleNotFoundError as exc:
        diagnostics.append(f"3D-визуализация пропущена: не установлена зависимость {exc.name}.")
        return [], missing, diagnostics

    restart = _resolve_path_from_values(files, mechanics, "restart_file", files.restart_npz)
    if restart is None or not restart.exists():
        diagnostics.append("3D-визуализация пропущена: не найден mechanics_restart.npz.")
        return [], missing, diagnostics

    try:
        mesh = _load_restart_mesh(restart, np)
    except Exception as exc:
        diagnostics.append(f"3D-визуализация пропущена: не удалось прочитать restart ({exc}).")
        return [], missing, diagnostics

    vtu_path, available_fields = _select_vtu_with_fields(files.vtu_files)
    if vtu_path is None:
        diagnostics.append("Для 3D-полей не найден VTU-файл с массивами результатов.")
        available_fields = []
    log.info("3D VTU для полей: %s", vtu_path)

    points_m = mesh["points"]
    cells = mesh["cells"]
    u_vertices_m = mesh["u_vertices"]
    material_cell_ids = mesh.get("material_cell_ids")
    if material_cell_ids is None or len(material_cell_ids) != len(cells):
        material_cell_ids = _infer_material_cell_ids(points_m, cells, mechanics, np)

    faces, owner_cells = _boundary_faces(cells, np)
    if faces.size == 0:
        diagnostics.append("3D-визуализация пропущена: не удалось выделить внешнюю поверхность сетки.")
        return [], missing, diagnostics

    figures: list[FigureRecord] = []
    face_material = np.asarray(material_cell_ids, dtype=int)[owner_cells]
    cilium_face_mask = face_material > 0
    substrate_face_mask = face_material <= 0
    lower_face_mask = face_material == 1
    upper_face_mask = face_material == 2
    displacement_face_uT = np.mean(u_vertices_m[faces, 0], axis=1) * 1.0e6
    reaction_uN = reaction_force_uN(mechanics)
    delta_mm = unit_value(mechanics, "delta_x_m", 1.0e3)
    deformed_points_mm = (points_m + u_vertices_m) * 1.0e3
    initial_points_mm = points_m * 1.0e3
    render_context = {
        "plt": plt,
        "poly_collection": Poly3DCollection,
        "np": np,
        "points_m": points_m,
        "cells": cells,
        "initial_points_mm": initial_points_mm,
        "deformed_points_mm": deformed_points_mm,
        "faces": faces,
        "owner_cells": owner_cells,
        "cell_material": np.asarray(material_cell_ids, dtype=int),
        "face_material": face_material,
        "cilium_face_mask": cilium_face_mask,
        "substrate_face_mask": substrate_face_mask,
        "lower_face_mask": lower_face_mask,
        "upper_face_mask": upper_face_mask,
        "displacement_face_uT": displacement_face_uT,
        "mechanics": mechanics,
        "delta_mm": delta_mm,
        "reaction_uN": reaction_uN,
    }

    for view_name, key in (("overview", "deformed_overview"), ("closeup", "deformed_closeup")):
        figures.append(_plot_deformed_view(render_context, outdir, view_name, key))

    vm_kpa = None
    jac = None
    if vtu_path is not None:
        vm = _load_vtu_cell_field(vtu_path, VTK_FIELD_ALIASES["von_mises"], np)
        if vm is None:
            missing["von_mises"] = available_fields
            diagnostics.append("Поле von Mises не найдено; рисунок напряжений пропущен.")
        else:
            vm_kpa = np.asarray(vm, dtype=float) * 1.0e-3
            vm_face = vm_kpa[owner_cells]
            vmax_real = float(np.nanmax(vm_kpa))
            field = {
                "kind": "von_mises",
                "face_values": vm_face,
                "cell_values": vm_kpa,
                "cmap": "magma",
                "colorbar": RU["vm_colorbar"],
                "title": RU["von_mises_title"],
                "subtitle": f"цветовая шкала ограничена по 99-му перцентилю, реальный максимум = {vmax_real:.2f} кПа",
                "vmin_vmax": _percentile_limits(vm_face[cilium_face_mask], np, 1.0, 99.0),
                "max_marker": True,
            }
            figures.append(_plot_scalar_view(render_context, field, outdir, "overview", "von_mises_overview"))
            figures.append(_plot_scalar_view(render_context, field, outdir, "base_zoom", "von_mises_base_zoom"))

        jac = _load_vtu_cell_field(vtu_path, VTK_FIELD_ALIASES["J"], np)
        if jac is None:
            missing["J"] = available_fields
            diagnostics.append("Поле J не найдено; рисунок якобиана пропущен.")
        else:
            jac = np.asarray(jac, dtype=float)
            jac_face = jac[owner_cells]
            j_min = float(np.nanmin(jac))
            j_max = float(np.nanmax(jac))
            j_deviation = (jac_face - 1.0) * 1.0e3
            j_dev_abs = max(abs(float(np.nanmin(j_deviation[cilium_face_mask]))), abs(float(np.nanmax(j_deviation[cilium_face_mask]))), 1.0e-6)
            field = {
                "kind": "jacobian",
                "face_values": j_deviation,
                "cell_values": jac,
                "cmap": "coolwarm",
                "colorbar": RU["j_colorbar"],
                "title": RU["jacobian_title"],
                "subtitle": f"J ∈ [{j_min:.6f}; {j_max:.6f}]",
                "vmin_vmax": (-j_dev_abs, j_dev_abs),
                "center": 0.0,
                "max_marker": False,
            }
            figures.append(_plot_scalar_view(render_context, field, outdir, "overview", "jacobian_overview"))
            figures.append(_plot_scalar_view(render_context, field, outdir, "base_zoom", "jacobian_base_zoom"))

    panel = _try_plot_mechanics_panel(render_context, outdir, vm_kpa=vm_kpa, jac=jac)
    if panel is not None:
        figures.append(panel)
    return figures, missing, diagnostics


def generate_visual_report(result_dir: Path, outdir: Path, enable_3d: bool = True, verbose: bool = False) -> VisualReportResult:
    result_dir = Path(result_dir)
    outdir = Path(outdir)
    setup_visual_logging(outdir, verbose=verbose)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    files = discover_result_files(result_dir)
    mechanics = load_json_mapping(files.mechanics_json)
    params = load_json_mapping(files.params_json)
    mechanics_with_params = {**params, **mechanics}
    magnetic_rows = read_csv_rows(files.magnetic_csv)
    magnetic_row = select_baseline_magnetic_row(magnetic_rows, mechanics_with_params)

    result = VisualReportResult(outdir=outdir, files=files, mechanics=mechanics_with_params, magnetic_row=magnetic_row)

    if mechanics_with_params:
        try:
            result.figures.append(plot_mechanics_summary_card(mechanics_with_params, figures_dir))
        except ModuleNotFoundError as exc:
            message = f"Карточка механики не построена: не установлена зависимость {exc.name}."
            log.warning(message)
            result.diagnostics.append(message)
        except Exception as exc:
            log.exception("Не удалось построить карточку механики")
            result.diagnostics.append(f"Карточка механики не построена: {exc}")
    else:
        result.diagnostics.append("Механический JSON не найден или пуст.")

    if magnetic_row is not None:
        try:
            result.figures.append(plot_delta_b_components(magnetic_row, figures_dir))
        except ModuleNotFoundError as exc:
            message = f"График ΔB не построен: не установлена зависимость {exc.name}."
            log.warning(message)
            result.diagnostics.append(message)
        except Exception as exc:
            log.exception("Не удалось построить график ΔB")
            result.diagnostics.append(f"График ΔB не построен: {exc}")
    else:
        result.diagnostics.append("Строка магнитного отклика с ΔBx/ΔBy/ΔBz не найдена.")

    if enable_3d:
        figures, missing, diagnostics = plot_mechanics_3d(files, mechanics_with_params, figures_dir)
        result.figures.extend(figures)
        result.missing_fields.update(missing)
        result.diagnostics.extend(diagnostics)
    else:
        result.diagnostics.append("3D-визуализация отключена флагом --no-3d.")

    result.report_md = write_markdown_report(result)
    result.report_html = write_html_report(result)
    log.info("Отчёт сохранён: %s", result.report_md)
    return result


def write_markdown_report(result: VisualReportResult) -> Path:
    outdir = result.outdir
    report_path = outdir / "report.md"
    files = result.files
    mechanics = result.mechanics
    magnetic_row = result.magnetic_row
    figures_by_key = {figure.key: figure for figure in result.figures}
    components = magnetic_components_uT(magnetic_row) if magnetic_row is not None else None

    lines = [
        f"# {RU['report_title']}",
        "",
        f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Использованные файлы",
        "",
    ]
    if files is not None:
        for label, path in (
            ("Директория результата", files.result_dir),
            ("JSON механики", files.mechanics_json),
            ("JSON параметров", files.params_json),
            ("CSV сводки", files.summary_csv),
            ("CSV магнитного отклика", files.magnetic_csv),
            ("PVD/VTK механики", files.pvd_file),
            ("Restart механики", files.restart_npz),
        ):
            lines.append(f"- {label}: `{_display_path(path)}`")
        if files.log_files:
            lines.append(f"- Логи: {', '.join(f'`{_display_path(path)}`' for path in files.log_files[:5])}")
    lines.extend(["", "## Механика", ""])
    for label, value in _mechanics_report_rows(mechanics):
        lines.append(f"- {label}: {value}")

    lines.extend(["", "## Магнитный отклик", ""])
    if magnetic_row is None or components is None:
        lines.append("- Компоненты ΔB: нет данных")
    else:
        norm = magnetic_norm_uT(magnetic_row, components)
        lines.append(f"- ΔBx: {components['x']:.6g} мкТл")
        lines.append(f"- ΔBy: {components['y']:.6g} мкТл")
        lines.append(f"- ΔBz: {components['z']:.6g} мкТл")
        lines.append(f"- |ΔB|: {norm:.6g} мкТл")
        lines.append(f"- Наиболее информативная компонента: {dominant_component(components)}")
        sensor = _format_sensor_position(magnetic_row)
        if sensor:
            lines.append(f"- Положение датчика Холла: {sensor}")

    lines.extend(["", "## Рисунки", ""])
    figure_order = (
        "mechanics_card",
        "deformed_overview",
        "deformed_closeup",
        "von_mises_overview",
        "von_mises_base_zoom",
        "jacobian_overview",
        "jacobian_base_zoom",
        "panel",
        "delta_b",
    )
    for key in figure_order:
        figure = figures_by_key.get(key)
        if figure is None:
            continue
        rel = figure.png_path.relative_to(outdir)
        lines.append(f"![{figure.title}]({rel.as_posix()})")
        lines.append("")

    lines.extend(["## Автоматический вывод", ""])
    for conclusion in automatic_conclusions(result):
        lines.append(f"- {conclusion}")

    if result.diagnostics:
        lines.extend(["", "## Диагностика", ""])
        for message in result.diagnostics:
            lines.append(f"- {message}")

    if result.missing_fields:
        lines.extend(["", "## Не найденные поля 3D", ""])
        for field, available in result.missing_fields.items():
            fields = ", ".join(available) if available else "список недоступен"
            lines.append(f"- {field}: доступные поля: {fields}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_html_report(result: VisualReportResult) -> Path:
    outdir = result.outdir
    report_path = outdir / "report.html"
    figures = "\n".join(
        f'<figure><img src="{html.escape(figure.png_path.relative_to(outdir).as_posix())}" alt="{html.escape(figure.title)}">'
        f"<figcaption>{html.escape(figure.title)}</figcaption></figure>"
        for figure in result.figures
    )
    mechanics_rows = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in _mechanics_report_rows(result.mechanics)
    )
    components = magnetic_components_uT(result.magnetic_row) if result.magnetic_row is not None else None
    if components:
        magnetic_rows = "\n".join(
            [
                f"<tr><th>ΔBx</th><td>{components['x']:.6g} мкТл</td></tr>",
                f"<tr><th>ΔBy</th><td>{components['y']:.6g} мкТл</td></tr>",
                f"<tr><th>ΔBz</th><td>{components['z']:.6g} мкТл</td></tr>",
                f"<tr><th>|ΔB|</th><td>{magnetic_norm_uT(result.magnetic_row or {}, components):.6g} мкТл</td></tr>",
                f"<tr><th>Доминирующая компонента</th><td>{dominant_component(components)}</td></tr>",
            ]
        )
    else:
        magnetic_rows = "<tr><td colspan='2'>Компоненты ΔB не найдены</td></tr>"
    conclusions = "\n".join(f"<li>{html.escape(item)}</li>" for item in automatic_conclusions(result))
    diagnostics = "\n".join(f"<li>{html.escape(item)}</li>" for item in result.diagnostics)
    page = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>{html.escape(RU['report_title'])}</title>
  <style>
    body {{ font-family: DejaVu Sans, Arial, sans-serif; margin: 36px; color: #1f2933; background: #ffffff; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px 0; min-width: 520px; }}
    th, td {{ border-bottom: 1px solid #d9dee6; padding: 8px 12px; text-align: left; }}
    th {{ color: #52616f; font-weight: 600; }}
    figure {{ margin: 24px 0; }}
    img {{ max-width: 980px; width: 100%; border: 1px solid #d9dee6; }}
    figcaption {{ margin-top: 6px; color: #52616f; }}
  </style>
</head>
<body>
  <h1>{html.escape(RU['report_title'])}</h1>
  <p>Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <h2>Механика</h2>
  <table>{mechanics_rows}</table>
  <h2>Магнитный отклик</h2>
  <table>{magnetic_rows}</table>
  <h2>Рисунки</h2>
  {figures}
  <h2>Автоматический вывод</h2>
  <ul>{conclusions}</ul>
  <h2>Диагностика</h2>
  <ul>{diagnostics}</ul>
</body>
</html>
"""
    report_path.write_text(page, encoding="utf-8")
    return report_path


def automatic_conclusions(result: VisualReportResult) -> list[str]:
    mechanics = result.mechanics
    reaction_uN = reaction_force_uN(mechanics)
    error = force_error_percent(mechanics)
    passed = error is not None and abs(error) <= FORCE_PASS_TOLERANCE_PERCENT
    conclusions = [_force_conclusion(reaction_uN, error, passed)]

    j_min = numeric_value(mechanics, FIELD_ALIASES["J_min"])
    j_max = numeric_value(mechanics, FIELD_ALIASES["J_max"])
    if j_min is None or j_max is None:
        conclusions.append("Диапазон J не удалось оценить: поля J_min/J_max не найдены.")
    elif j_min > 0.0 and 0.95 <= j_min <= j_max <= 1.05:
        conclusions.append(f"Диапазон J [{j_min:.6f}; {j_max:.6f}] физически корректен и близок к 1.")
    elif j_min > 0.0:
        conclusions.append(f"Диапазон J положителен [{j_min:.6f}; {j_max:.6f}], но требует отдельной проверки.")
    else:
        conclusions.append(f"Диапазон J некорректен: минимум J = {j_min:.6f}.")

    components = magnetic_components_uT(result.magnetic_row) if result.magnetic_row is not None else None
    if components:
        conclusions.append(f"Доминирует компонента {dominant_component(components)} при текущем фиксированном положении датчика.")
    else:
        conclusions.append("Доминирующую компоненту ΔB определить не удалось: нет полной тройки ΔBx/ΔBy/ΔBz.")

    if result.figures:
        names = ", ".join(figure.title for figure in result.figures)
        conclusions.append(f"Успешно построены рисунки: {names}.")
    else:
        conclusions.append("Рисунки не построены; см. диагностику.")
    return conclusions


def save_figure(
    fig: Any,
    outdir: Path,
    stem: str,
    title: str,
    key: str,
    *,
    save_pdf: bool = True,
    dpi: int = 240,
    tight: bool = True,
) -> FigureRecord:
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf" if save_pdf else None
    bbox = "tight" if tight else None
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox, facecolor=fig.get_facecolor())
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    _close_figure(fig)
    return FigureRecord(key=key, title=title, png_path=png_path, pdf_path=pdf_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Построить визуальный отчёт по готовым результатам magnetic_cilium.")
    parser.add_argument("--result-dir", required=True, type=Path, help="Директория с готовыми результатами расчёта.")
    parser.add_argument("--outdir", required=True, type=Path, help="Директория для visual_report.")
    parser.add_argument("--no-3d", action="store_true", help="Не строить 3D-визуализации механики.")
    parser.add_argument("--verbose", action="store_true", help="Подробный лог.")
    args = parser.parse_args(argv)
    report = generate_visual_report(args.result_dir, args.outdir, enable_3d=not args.no_3d, verbose=args.verbose)
    print(f"Отчёт: {report.report_md}")
    print(f"HTML: {report.report_html}")
    print(f"Рисунки: {args.outdir / 'figures'}")
    if report.diagnostics:
        print("Диагностика:")
        for message in report.diagnostics:
            print(f"- {message}")
    return 0


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise exc
    return np


def _require_matplotlib() -> Any:
    mpl_config_dir = Path(tempfile.gettempdir()) / "magnetic_cilium_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib  # type: ignore
    except ModuleNotFoundError as exc:
        raise exc
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt  # type: ignore

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "figure.dpi": 120,
        }
    )
    return plt


def _close_figure(fig: Any) -> None:
    try:
        from matplotlib import pyplot as plt  # type: ignore

        plt.close(fig)
    except Exception:
        pass


def _load_restart_mesh(path: Path, np: Any) -> dict[str, Any]:
    data = np.load(path)
    points = np.asarray(data["points"], dtype=float)
    cells = np.asarray(data["cells"], dtype=int)
    u_vertices = np.asarray(data["u_vertices"], dtype=float)
    material_cell_ids = np.asarray(data["material_cell_ids"], dtype=int) if "material_cell_ids" in data else None
    if cells.ndim != 2 or cells.shape[1] < 4:
        raise ValueError("restart cells must have at least four vertex columns")
    if u_vertices.shape[0] != points.shape[0]:
        raise ValueError("u_vertices length does not match points length")
    return {
        "points": points[:, :3],
        "cells": cells[:, :4],
        "u_vertices": u_vertices[:, :3],
        "material_cell_ids": material_cell_ids,
    }


def _infer_material_cell_ids(points_m: Any, cells: Any, mechanics: Mapping[str, Any], np: Any) -> Any:
    l1 = _lower_layer_length_m(mechanics)
    cell_centers = np.mean(points_m[cells[:, :4]], axis=1)
    material = np.zeros(len(cells), dtype=int)
    material[cell_centers[:, 2] >= 0.0] = 1
    material[cell_centers[:, 2] >= l1] = 2
    return material


def _select_vtu_with_fields(vtu_files: Sequence[Path]) -> tuple[Path | None, list[str]]:
    best_path: Path | None = None
    best_fields: list[str] = []
    best_score = -1
    for path in vtu_files:
        fields = _list_vtu_fields(path)
        score = 0
        names = {name.lower() for name in fields}
        for aliases in VTK_FIELD_ALIASES.values():
            if any(alias.lower() in names for alias in aliases):
                score += 1
        if "000001" in path.name or "_p0_000001" in path.name:
            score += 1
        if score > best_score:
            best_path = path
            best_fields = fields
            best_score = score
    if best_score <= 0:
        return None, best_fields
    return best_path, best_fields


def _list_vtu_fields(path: Path) -> list[str]:
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return []
    fields: list[str] = []
    for parent_name in ("PointData", "CellData"):
        for data_array in root.findall(f".//{parent_name}/DataArray"):
            name = data_array.attrib.get("Name")
            if name:
                fields.append(name)
    return fields


def _load_vtu_cell_field(path: Path, aliases: Sequence[str], np: Any) -> Any | None:
    alias_set = {alias.lower() for alias in aliases}
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        log.warning("Не удалось прочитать VTU %s: %s", path, exc)
        return None
    for data_array in root.findall(".//CellData/DataArray"):
        name = data_array.attrib.get("Name", "")
        if name.lower() in alias_set:
            text = data_array.text or ""
            values = np.fromstring(text, sep=" ")
            ncomp = int(data_array.attrib.get("NumberOfComponents", "1"))
            if ncomp > 1:
                values = values.reshape((-1, ncomp))
            return values
    return None


def _boundary_faces(cells: Any, np: Any) -> tuple[Any, Any]:
    owners: dict[tuple[int, int, int], tuple[tuple[int, int, int], int] | None] = {}
    face_indices = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for cell_id, cell in enumerate(cells[:, :4]):
        tet_faces = (
            (int(cell[0]), int(cell[1]), int(cell[2])),
            (int(cell[0]), int(cell[1]), int(cell[3])),
            (int(cell[0]), int(cell[2]), int(cell[3])),
            (int(cell[1]), int(cell[2]), int(cell[3])),
        )
        for face in tet_faces:
            key = tuple(sorted(face))
            if key in owners:
                owners[key] = None
            else:
                owners[key] = (face, cell_id)
    boundary = [value for value in owners.values() if value is not None]
    if not boundary:
        return np.empty((0, 3), dtype=int), np.empty((0,), dtype=int)
    faces = np.asarray([item[0] for item in boundary], dtype=int)
    owner_cells = np.asarray([item[1] for item in boundary], dtype=int)
    return faces, owner_cells


def _plot_deformed_view(context: Mapping[str, Any], outdir: Path, view_name: str, key: str) -> FigureRecord:
    plt = context["plt"]
    fig = plt.figure(figsize=(9.8, 7.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    bounds = _view_bounds(context, view_name)
    face_mask = _faces_in_bounds(context["deformed_points_mm"], context["faces"], bounds, context["np"])
    cilium_mask = face_mask & context["cilium_face_mask"]
    substrate_mask = face_mask & context["substrate_face_mask"]

    _plot_surface(
        ax,
        context,
        context["initial_points_mm"],
        face_mask,
        color="#aeb7c2",
        alpha=0.17,
        linewidth=0.0,
    )
    _plot_surface(
        ax,
        context,
        context["deformed_points_mm"],
        substrate_mask,
        color="#d4d8dd",
        alpha=0.28,
        linewidth=0.0,
    )
    mappable = _plot_surface(
        ax,
        context,
        context["deformed_points_mm"],
        cilium_mask,
        face_values=context["displacement_face_uT"][cilium_mask],
        cmap="viridis",
        alpha=0.98,
        linewidth=0.0,
    )
    _add_layer_ring_and_labels(ax, context, bounds, closeup=view_name != "overview")
    _add_displacement_arrow(ax, bounds, context["delta_mm"], context["reaction_uN"])
    _format_3d_axis(ax, RU["deformed_title"], _delta_force_label(context["delta_mm"], context["reaction_uN"]), bounds, closeup=view_name != "overview")
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.66, pad=0.02)
    cbar.set_label(RU["ux_colorbar"], fontsize=10)
    return save_figure(
        fig,
        outdir,
        FIGURE_FILENAMES[key],
        f"{RU['deformed_title']} ({_view_label(view_name)})",
        key,
        save_pdf=False,
        dpi=220,
        tight=False,
    )


def _plot_scalar_view(context: Mapping[str, Any], field: Mapping[str, Any], outdir: Path, view_name: str, key: str) -> FigureRecord:
    plt = context["plt"]
    fig = plt.figure(figsize=(9.8, 7.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    bounds = _view_bounds(context, view_name)
    face_mask = _faces_in_bounds(context["deformed_points_mm"], context["faces"], bounds, context["np"])
    cilium_mask = face_mask & context["cilium_face_mask"]
    substrate_mask = face_mask & context["substrate_face_mask"]

    _plot_surface(
        ax,
        context,
        context["deformed_points_mm"],
        substrate_mask,
        color="#d3d6db",
        alpha=0.24 if view_name == "overview" else 0.32,
        linewidth=0.0,
    )
    vmin, vmax = field["vmin_vmax"]
    if field.get("kind") == "von_mises" and cilium_mask.any():
        vmin, vmax = _percentile_limits(field["face_values"][cilium_mask], context["np"], 1.0, 99.0)
    elif field.get("kind") == "jacobian" and cilium_mask.any():
        local = field["face_values"][cilium_mask]
        dev = max(abs(float(context["np"].nanmin(local))), abs(float(context["np"].nanmax(local))), 1.0e-6)
        vmin, vmax = -dev, dev
    mappable = _plot_surface(
        ax,
        context,
        context["deformed_points_mm"],
        cilium_mask,
        face_values=field["face_values"][cilium_mask],
        cmap=str(field["cmap"]),
        alpha=0.98,
        linewidth=0.0,
        vmin=vmin,
        vmax=vmax,
        center=field.get("center"),
    )
    _add_layer_ring_and_labels(ax, context, bounds, closeup=view_name != "overview")
    if field.get("max_marker"):
        _add_face_maximum_marker(ax, context, field["face_values"], cilium_mask)
    _format_3d_axis(ax, str(field["title"]), str(field["subtitle"]), bounds, closeup=view_name != "overview")
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.66, pad=0.02)
    cbar.set_label(str(field["colorbar"]), fontsize=10)
    return save_figure(
        fig,
        outdir,
        FIGURE_FILENAMES[key],
        f"{field['title']} ({_view_label(view_name)})",
        key,
        save_pdf=False,
        dpi=220,
        tight=False,
    )


def _plot_surface(
    ax: Any,
    context: Mapping[str, Any],
    points_mm: Any,
    face_mask: Any,
    *,
    face_values: Any | None = None,
    cmap: str | None = None,
    alpha: float,
    color: str | None = None,
    linewidth: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
) -> Any:
    np = context["np"]
    faces = context["faces"][face_mask]
    if len(faces) == 0:
        return _empty_scalar_mappable(cmap or "viridis", vmin, vmax, center)
    vertices = points_mm[faces]
    facecolors, mappable = _surface_facecolors(vertices, face_values, cmap, color, alpha, vmin, vmax, center, np)
    collection = context["poly_collection"](
        vertices,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=linewidth,
        antialiaseds=True,
        zsort="average",
    )
    ax.add_collection3d(collection)
    return mappable


def _surface_facecolors(
    vertices: Any,
    face_values: Any | None,
    cmap: str | None,
    color: str | None,
    alpha: float,
    vmin: float | None,
    vmax: float | None,
    center: float | None,
    np: Any,
) -> tuple[Any, Any]:
    from matplotlib import cm, colors  # type: ignore

    normals = _face_normals(vertices, np)
    light_dir = np.asarray([0.35, -0.48, 0.80], dtype=float)
    light_dir = light_dir / np.linalg.norm(light_dir)
    diffuse = np.clip(np.abs(normals @ light_dir), 0.0, 1.0)
    specular = np.clip(normals @ light_dir, 0.0, 1.0) ** 18
    intensity = np.clip(0.58 + 0.36 * diffuse + 0.12 * specular, 0.0, 1.18)

    if face_values is None:
        base_rgb = np.asarray(colors.to_rgb(color or "#c7ced6"), dtype=float)
        rgba = np.tile(np.r_[base_rgb, alpha], (len(vertices), 1))
        mappable = _empty_scalar_mappable(cmap or "viridis", vmin, vmax, center)
    else:
        norm = _color_norm(face_values, vmin, vmax, center)
        cmap_obj = cm.get_cmap(cmap or "viridis")
        rgba = cmap_obj(norm(face_values))
        rgba[:, 3] = alpha
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(face_values)

    rgba[:, :3] = np.clip(rgba[:, :3] * intensity[:, None] + 0.035 * specular[:, None], 0.0, 1.0)
    return rgba, mappable


def _empty_scalar_mappable(cmap: str, vmin: float | None, vmax: float | None, center: float | None) -> Any:
    from matplotlib import cm  # type: ignore

    norm = _color_norm([0.0, 1.0], vmin, vmax, center)
    return cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap))


def _color_norm(values: Any, vmin: float | None, vmax: float | None, center: float | None) -> Any:
    from matplotlib import colors  # type: ignore

    if vmin is None:
        vmin = float(min(values))
    if vmax is None:
        vmax = float(max(values))
    if abs(float(vmax) - float(vmin)) < 1.0e-15:
        vmax = float(vmin) + 1.0
    if center is not None and vmin < center < vmax:
        return colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _face_normals(vertices: Any, np: Any) -> Any:
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0.0] = 1.0
    return normals / lengths[:, None]


def _format_3d_axis(ax: Any, title: str, subtitle: str, bounds: tuple[float, float, float, float, float, float], *, closeup: bool) -> None:
    ax.set_title(title if not subtitle else f"{title}\n{subtitle}", fontsize=10.6, weight="bold", pad=8)
    ax.set_xlabel(RU["x_axis"], fontsize=9.5, labelpad=4)
    ax.set_ylabel(RU["y_axis"], fontsize=9.5, labelpad=4)
    ax.set_zlabel(RU["z_axis"], fontsize=9.5, labelpad=4)
    ax.tick_params(labelsize=8.5, pad=1)
    ax.view_init(elev=18 if closeup else 20, azim=-70)
    ax.grid(not closeup)
    _set_axis_bounds(ax, bounds, compress_z=closeup)
    pane_alpha = 0.0 if closeup else 0.06
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.96, 0.97, 0.98, pane_alpha))
        axis.pane.set_edgecolor((0.82, 0.86, 0.90, 0.28 if not closeup else 0.0))


def _set_axis_bounds(ax: Any, bounds: tuple[float, float, float, float, float, float], *, compress_z: bool) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    xspan = max(xmax - xmin, 1.0e-6)
    yspan = max(ymax - ymin, 1.0e-6)
    zspan = max(zmax - zmin, 1.0e-6)
    lateral = max(xspan, yspan)
    aspect_z = min(zspan, 2.8 * lateral) if compress_z else min(zspan, 3.6 * lateral)
    try:
        ax.set_box_aspect((xspan, yspan, aspect_z))
    except Exception:
        pass


def _view_bounds(context: Mapping[str, Any], view_name: str) -> tuple[float, float, float, float, float, float]:
    np = context["np"]
    deformed = context["deformed_points_mm"]
    initial = context["initial_points_mm"]
    faces = context["faces"]
    cilium_faces = faces[context["cilium_face_mask"]]
    substrate_faces = faces[context["substrate_face_mask"]]

    if view_name == "overview" or cilium_faces.size == 0:
        points = np.vstack([initial, deformed])
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        margins = np.maximum((maxs - mins) * np.asarray([0.08, 0.08, 0.035]), np.asarray([0.06, 0.06, 0.05]))
        return (
            float(mins[0] - margins[0]),
            float(maxs[0] + margins[0]),
            float(mins[1] - margins[1]),
            float(maxs[1] + margins[1]),
            float(mins[2] - margins[2]),
            float(maxs[2] + margins[2]),
        )

    cilium_points = deformed[np.unique(cilium_faces.reshape(-1))]
    cmins = cilium_points.min(axis=0)
    cmaxs = cilium_points.max(axis=0)
    radius_mm = _cilium_radius_m(context["mechanics"]) * 1.0e3
    substrate_margin = max(4.2 * radius_mm, 0.24)
    x_margin = max(2.6 * radius_mm, 0.16)
    y_margin = max(2.6 * radius_mm, 0.16)

    if view_name == "base_zoom":
        l1_mm = _lower_layer_length_m(context["mechanics"]) * 1.0e3
        zmax = min(float(cmaxs[2] + 0.04), max(0.85, 0.55 * l1_mm))
        zmin = min(-0.18, float(cmins[2] - 0.04))
        xcenter = float(np.mean(cilium_points[cilium_points[:, 2] <= max(0.25, zmax), 0])) if cilium_points.size else 0.0
        ycenter = float(np.mean(cilium_points[:, 1])) if cilium_points.size else 0.0
        return (
            xcenter - substrate_margin,
            xcenter + substrate_margin,
            ycenter - substrate_margin,
            ycenter + substrate_margin,
            zmin,
            zmax,
        )

    if substrate_faces.size:
        substrate_points = deformed[np.unique(substrate_faces.reshape(-1))]
        top_context = substrate_points[substrate_points[:, 2] > -0.22]
        if top_context.size:
            context_x = top_context[:, 0]
            context_y = top_context[:, 1]
            xmin = min(float(cmins[0] - x_margin), float(np.percentile(context_x, 12)))
            xmax = max(float(cmaxs[0] + x_margin), float(np.percentile(context_x, 88)))
            ymin = min(float(cmins[1] - y_margin), float(np.percentile(context_y, 12)))
            ymax = max(float(cmaxs[1] + y_margin), float(np.percentile(context_y, 88)))
        else:
            xmin, xmax = float(cmins[0] - x_margin), float(cmaxs[0] + x_margin)
            ymin, ymax = float(cmins[1] - y_margin), float(cmaxs[1] + y_margin)
    else:
        xmin, xmax = float(cmins[0] - x_margin), float(cmaxs[0] + x_margin)
        ymin, ymax = float(cmins[1] - y_margin), float(cmaxs[1] + y_margin)

    return (xmin, xmax, ymin, ymax, min(-0.16, float(cmins[2] - 0.04)), float(cmaxs[2] + 0.08))


def _faces_in_bounds(points_mm: Any, faces: Any, bounds: tuple[float, float, float, float, float, float], np: Any) -> Any:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    centers = np.mean(points_mm[faces], axis=1)
    return (
        (centers[:, 0] >= xmin)
        & (centers[:, 0] <= xmax)
        & (centers[:, 1] >= ymin)
        & (centers[:, 1] <= ymax)
        & (centers[:, 2] >= zmin)
        & (centers[:, 2] <= zmax)
    )


def _add_displacement_arrow(ax: Any, bounds: tuple[float, float, float, float, float, float], delta_mm: float | None, reaction_uN: float | None) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xspan = xmax - xmin
    yspan = ymax - ymin
    zspan = zmax - zmin
    length = min(max(abs(delta_mm or 0.0) * 0.45, 0.16 * xspan), 0.42 * xspan)
    x0 = xmin + 0.10 * xspan
    y0 = ymax - 0.13 * yspan
    z0 = zmax - 0.10 * zspan
    ax.quiver(x0, y0, z0, length, 0.0, 0.0, color="#1f5f99", linewidth=2.2, arrow_length_ratio=0.24)
    ax.text2D(0.58, 0.80, "направление δx", transform=ax.transAxes, color="#1f5f99", fontsize=8.8)


def _delta_force_label(delta_mm: float | None, reaction_uN: float | None) -> str:
    parts = []
    if delta_mm is not None:
        parts.append(f"δx = {delta_mm:.3f} мм")
    if reaction_uN is not None:
        parts.append(f"Fx = {reaction_uN:.2f} мкН")
    return ", ".join(parts)


def _add_layer_ring_and_labels(
    ax: Any,
    context: Mapping[str, Any],
    bounds: tuple[float, float, float, float, float, float],
    *,
    closeup: bool,
    labels: bool = True,
) -> None:
    np = context["np"]
    l1_m = _lower_layer_length_m(context["mechanics"])
    l1_mm = l1_m * 1.0e3
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if not (zmin <= l1_mm <= zmax):
        return

    points_m = context["points_m"]
    deformed_mm = context["deformed_points_mm"]
    radius = _cilium_radius_m(context["mechanics"])
    tol = max(0.015e-3, 0.35 * radius)
    radial = np.linalg.norm(points_m[:, :2], axis=1)
    idx = np.where((np.abs(points_m[:, 2] - l1_m) <= tol) & (radial <= 1.35 * radius))[0]
    if len(idx) >= 8:
        ring = deformed_mm[idx]
        angles = np.arctan2(ring[:, 1] - np.mean(ring[:, 1]), ring[:, 0] - np.mean(ring[:, 0]))
        ring = ring[np.argsort(angles)]
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, 96)
        ring_radius_mm = radius * 1.0e3
        x_shift = (context["delta_mm"] or 0.0) * (l1_m / max(_total_length_m(context["mechanics"]), 1.0e-12))
        ring = np.column_stack([x_shift + ring_radius_mm * np.cos(theta), ring_radius_mm * np.sin(theta), np.full_like(theta, l1_mm)])
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#1f2933", linewidth=1.4, alpha=0.88)

    if labels and closeup:
        ax.text2D(
            0.06,
            0.30,
            RU["lower_layer"],
            transform=ax.transAxes,
            color="#335c77",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.22,rounding_size=0.04", "facecolor": "#eef5fb", "edgecolor": "#b7cadd", "alpha": 0.90},
        )
        ax.text2D(
            0.06,
            0.58,
            RU["upper_layer"],
            transform=ax.transAxes,
            color="#9a5b1d",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.22,rounding_size=0.04", "facecolor": "#fff4e5", "edgecolor": "#e1c39b", "alpha": 0.90},
        )


def _add_face_maximum_marker(ax: Any, context: Mapping[str, Any], face_values: Any, face_mask: Any) -> None:
    np = context["np"]
    visible_ids = np.where(face_mask)[0]
    if visible_ids.size == 0:
        return
    values = np.asarray(face_values, dtype=float)[visible_ids]
    if values.size == 0 or not np.isfinite(values).any():
        return
    face_id = int(visible_ids[int(np.nanargmax(values))])
    point = np.mean(context["deformed_points_mm"][context["faces"][face_id]], axis=0)
    ax.scatter([point[0]], [point[1]], [point[2]], color="#d62728", s=42, depthshade=True)
    ax.text(point[0], point[1], point[2], "  максимум", color="#a51d2d", fontsize=8.3)


def _add_maximum_marker(ax: Any, context: Mapping[str, Any], cell_values: Any, bounds: tuple[float, float, float, float, float, float]) -> None:
    np = context["np"]
    cell_material = context["cell_material"]
    candidate = np.where(cell_material > 0)[0]
    if candidate.size == 0:
        candidate = np.arange(len(cell_values))
    max_cell = int(candidate[np.nanargmax(np.asarray(cell_values)[candidate])])
    if max_cell >= len(context["cells"]):
        return
    point_ids = context["cells"][max_cell, :4]
    point = np.mean(context["deformed_points_mm"][point_ids], axis=0)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax and zmin <= point[2] <= zmax):
        return
    ax.scatter([point[0]], [point[1]], [point[2]], color="#d62728", s=36, depthshade=True)
    ax.text(point[0], point[1], point[2], "  максимум", color="#a51d2d", fontsize=8.3)


def _view_label(view_name: str) -> str:
    return {
        "overview": "общий вид",
        "closeup": "крупный вид",
        "base_zoom": "zoom основания",
    }.get(view_name, view_name)


def _lower_layer_length_m(mechanics: Mapping[str, Any]) -> float:
    return numeric_value(mechanics, FIELD_ALIASES["L1_m"]) or 2.0e-3


def _total_length_m(mechanics: Mapping[str, Any]) -> float:
    l1 = _lower_layer_length_m(mechanics)
    l2 = numeric_value(mechanics, FIELD_ALIASES["L2_m"]) or 2.0e-3
    return l1 + l2


def _cilium_radius_m(mechanics: Mapping[str, Any]) -> float:
    radius = numeric_value(mechanics, FIELD_ALIASES["R_m"])
    if radius is not None:
        return radius
    diameter = numeric_value(mechanics, FIELD_ALIASES["D_m"])
    return 0.5 * diameter if diameter is not None else 60.0e-6


def _percentile_limits(values: Any, np: Any, low: float, high: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(float(vmax) - float(vmin)) < 1.0e-15:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    if abs(float(vmax) - float(vmin)) < 1.0e-15:
        vmax = float(vmin) + 1.0
    return float(vmin), float(vmax)


def _try_plot_mechanics_panel(context: Mapping[str, Any], outdir: Path, *, vm_kpa: Any | None, jac: Any | None) -> FigureRecord | None:
    if vm_kpa is None or jac is None:
        return None
    plt = context["plt"]
    np = context["np"]
    owner_cells = context["owner_cells"]
    vm_face = np.asarray(vm_kpa, dtype=float)[owner_cells]
    jac_face = (np.asarray(jac, dtype=float)[owner_cells] - 1.0) * 1.0e3
    cilium_mask = context["cilium_face_mask"]
    j_dev_abs = max(abs(float(np.nanmin(jac_face[cilium_mask]))), abs(float(np.nanmax(jac_face[cilium_mask]))), 1.0e-6)
    fig = plt.figure(figsize=(16.8, 6.2), facecolor="white")
    fig.suptitle(RU["mechanics_panel_title"], fontsize=15.5, weight="bold", y=0.98)
    panel_data = [
        ("а", RU["deformed_title"], "overview", context["displacement_face_uT"], "viridis", RU["ux_colorbar"], None, None, None),
        ("б", "von Mises: zoom основания", "base_zoom", vm_face, "magma", RU["vm_colorbar"], *_percentile_limits(vm_face[cilium_mask], np, 1.0, 99.0), None),
        ("в", "Отклонение J − 1", "base_zoom", jac_face, "coolwarm", RU["j_colorbar"], -j_dev_abs, j_dev_abs, 0.0),
    ]
    for index, (letter, title, view_name, values, cmap, cbar_label, vmin, vmax, center) in enumerate(panel_data, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        bounds = _view_bounds(context, view_name)
        face_mask = _faces_in_bounds(context["deformed_points_mm"], context["faces"], bounds, np)
        cilium_faces = face_mask & context["cilium_face_mask"]
        substrate_faces = face_mask & context["substrate_face_mask"]
        _plot_surface(
            ax,
            context,
            context["deformed_points_mm"],
            substrate_faces,
            color="#d5d8dd",
            alpha=0.22 if index == 1 else 0.30,
            linewidth=0.0,
        )
        mappable = _plot_surface(
            ax,
            context,
            context["deformed_points_mm"],
            cilium_faces,
            face_values=values[cilium_faces],
            cmap=cmap,
            alpha=0.96,
            linewidth=0.0,
            vmin=vmin,
            vmax=vmax,
            center=center,
        )
        _add_layer_ring_and_labels(ax, context, bounds, closeup=True, labels=False)
        _format_3d_axis(ax, title, "", bounds, closeup=True)
        ax.text2D(0.02, 0.95, letter, transform=ax.transAxes, fontsize=15, weight="bold", color="#1f2933")
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.54, pad=0.01)
        cbar.set_label(cbar_label, fontsize=9)
    return save_figure(fig, outdir, FIGURE_FILENAMES["panel"], RU["mechanics_panel_title"], "panel", save_pdf=False, dpi=220, tight=False)


def _resolve_path_from_values(files: ResultFiles, values: Mapping[str, Any], key: str, fallback: Path | None) -> Path | None:
    raw = first_existing_value(values, (key,))
    candidates: list[Path] = []
    if raw:
        raw_path = Path(str(raw))
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            bases = [files.result_dir]
            if files.mechanics_json is not None:
                bases.append(files.mechanics_json.parent)
            bases.extend([Path.cwd(), Path.cwd() / "magnetic_cilium_pipeline"])
            candidates.extend(base / raw_path for base in bases)
    if fallback is not None:
        candidates.append(fallback)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return fallback


def _mechanics_report_rows(mechanics: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Заданное смещение", _format_unit(unit_value(mechanics, "delta_x_m", 1.0e3), "мм", 3)),
        ("Реакция по x", _format_unit(reaction_force_uN(mechanics), "мкН", 2)),
        ("Ошибка относительно 60 мкН", _format_unit(force_error_percent(mechanics), "%", 2)),
        ("J_min", _format_plain(numeric_value(mechanics, FIELD_ALIASES["J_min"]), 6)),
        ("J_max", _format_plain(numeric_value(mechanics, FIELD_ALIASES["J_max"]), 6)),
        ("Максимальное von Mises", _format_unit(unit_value(mechanics, "von_mises_max_Pa", 1.0e-3), "кПа", 2)),
        ("Степень элемента", _format_plain(numeric_value(mechanics, FIELD_ALIASES["element_degree"]), 0)),
        ("Размеры сетки h", _format_mesh_sizes(mechanics)),
        ("Коэффициенты Пуассона", _format_poisson(mechanics)),
    ]


def _format_sensor_position(row: Mapping[str, Any]) -> str:
    x = numeric_value(row, FIELD_ALIASES["sensor_x_m"])
    y = numeric_value(row, FIELD_ALIASES["sensor_y_m"])
    z = numeric_value(row, FIELD_ALIASES["sensor_z_m"])
    if x is None and y is None and z is None:
        return ""
    return f"x={_format_unit(None if x is None else x * 1e6, 'мкм', 1)}, y={_format_unit(None if y is None else y * 1e6, 'мкм', 1)}, z={_format_unit(None if z is None else z * 1e6, 'мкм', 1)}"


def _force_conclusion(reaction_uN: float | None, error_percent: float | None, passed: bool) -> str:
    if reaction_uN is None or error_percent is None:
        return "Попадание в 60 мкН не удалось оценить: нет реакции или ошибки."
    if passed:
        return f"Механический расчёт попал в целевую силу: реакция {reaction_uN:.2f} мкН, ошибка {error_percent:.2f}%."
    return f"Механический расчёт требует проверки: реакция {reaction_uN:.2f} мкН, ошибка {error_percent:.2f}%."


def _format_mesh_sizes(mechanics: Mapping[str, Any]) -> str:
    h_cilium = unit_value(mechanics, "h_cilium_m", 1.0e6)
    h_substrate = unit_value(mechanics, "h_substrate_m", 1.0e6)
    parts = []
    if h_cilium is not None:
        parts.append(f"h реснички = {h_cilium:.1f} мкм")
    if h_substrate is not None:
        parts.append(f"h подложки = {h_substrate:.1f} мкм")
    return ", ".join(parts) if parts else "нет данных"


def _format_poisson(mechanics: Mapping[str, Any]) -> str:
    nu_pdms = numeric_value(mechanics, FIELD_ALIASES["nu_pdms"])
    nu_magnetic = numeric_value(mechanics, FIELD_ALIASES["nu_magnetic"])
    parts = []
    if nu_pdms is not None:
        parts.append(f"PDMS = {nu_pdms:.3f}")
    if nu_magnetic is not None:
        parts.append(f"магнитный слой = {nu_magnetic:.3f}")
    return ", ".join(parts) if parts else "нет данных"


def _format_unit(value: float | None, unit: str, digits: int) -> str:
    if value is None or not math.isfinite(value):
        return "нет данных"
    return f"{value:.{digits}f} {unit}"


def _format_plain(value: float | None, digits: int) -> str:
    if value is None or not math.isfinite(value):
        return "нет данных"
    if digits <= 0:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def _format_range(vmin: float | None, vmax: float | None, unit: str, digits: int) -> str:
    if vmin is None or vmax is None:
        return "нет данных"
    suffix = f" {unit}" if unit else ""
    return f"{vmin:.{digits}f} ... {vmax:.{digits}f}{suffix}"


def _as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "ok"}
    return bool(value)


def _signed_delta_alias_for_abs(component: str, row: Mapping[str, Any]) -> float | None:
    for key, scale in DELTA_B_ALIASES[component]:
        if key.startswith("abs_"):
            continue
        value = _as_float(row.get(key))
        if value is not None:
            return value * scale
    return None


def _display_path(path: Path | None) -> str:
    return "не найдено" if path is None else str(path)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = Path(path)
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            result.append(path)
    return result


def _select_candidate(paths: Iterable[Path], result_dir: Path, scorer: Any) -> Path | None:
    candidates = _unique_paths(paths)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (scorer(path, result_dir), path.stat().st_mtime))


def _score_mechanics_json(path: Path, result_dir: Path) -> int:
    return _name_score(path, result_dir, preferred=("mechanics_result", "final", "baseline"), avoided=("visual_report",))


def _score_params_json(path: Path, result_dir: Path) -> int:
    return _name_score(path, result_dir, preferred=("params", "final", "baseline"), avoided=("visual_report",))


def _score_restart_npz(path: Path, result_dir: Path) -> int:
    return _name_score(path, result_dir, preferred=("mechanics_restart", "final", "baseline"), avoided=("visual_report",))


def _score_summary_csv(path: Path, result_dir: Path) -> int:
    return _name_score(path, result_dir, preferred=("summary", "final", "baseline"), avoided=("visual_report", "interpolation"))


def _score_magnetic_csv(path: Path, result_dir: Path) -> int:
    return _name_score(path, result_dir, preferred=("summary", "magnetic", "master", "baseline"), avoided=("visual_report", "interpolation"))


def _score_pvd(path: Path, result_dir: Path) -> int:
    return _name_score(path, result_dir, preferred=("magnetic_cilium_3d", "mechanics", "final", "baseline"), avoided=("magnetostatic", "visual_report"))


def _name_score(path: Path, result_dir: Path, *, preferred: Sequence[str], avoided: Sequence[str]) -> int:
    text = path.as_posix().lower()
    score = 0
    if path.parent == result_dir:
        score += 15
    for index, token in enumerate(preferred):
        if token in text:
            score += 20 - index
    for token in avoided:
        if token in text:
            score -= 50
    return score


def _csv_has_any(path: Path, tokens: Sequence[str]) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            header = f.readline()
    except OSError:
        return False
    header_lower = header.lower()
    return any(token.lower() in header_lower for token in tokens)


if __name__ == "__main__":
    raise SystemExit(main())
