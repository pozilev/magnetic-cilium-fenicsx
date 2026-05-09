import csv
import gc
import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np

from params import ModelParams, TARGET_REACTION_U_N, make_params_from_args
from mechanics_model import (
    build_gmsh_mesh_3d,
    compute_J_field,
    compute_material_volumes,
    compute_von_mises_field,
    create_dolfinx_mesh_3d,
    evaluate_displacement_at_geometry_vertices,
    extract_cells_from_domain,
    extract_material_cell_ids,
    log_mesh_resolution_warning,
    make_material_field,
    make_material_field_from_cell_ids,
    save_results,
    solve_hyperelasticity_displacement_control_3d,
    validate_result_quality,
)
from magnetics_dipoles import compute_magnetic_dipole_diagnostics, log_magnetic_diagnostics


log = logging.getLogger("magnetic_cilium_3d")

DEFAULT_MAGNETIC_ONLY_RESTART_DIR = (
    "magnetic_cilium_3d_results_final/"
    "final_P2_hcil_20um_hsub_100um_delta_0p550mm"
)

MAGNETIC_ONLY_BR_VALUES = [0.02, 0.05, 0.10, 0.15]
MAGNETIC_ONLY_SENSOR_GAPS = [25e-6, 50e-6, 100e-6]
MAGNETIC_ONLY_SENSOR_X_FACTORS = [0.0, 0.5, 1.0]
MAX_STABLE_MAGNETIC_FIELD_T = 10.0


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def save_mechanics_restart(domain, material, u_vertices: np.ndarray, params: ModelParams, mechanics_result: Dict[str, Any]) -> str:
    os.makedirs(params.outdir, exist_ok=True)
    points = domain.geometry.x[:, :3].copy()
    cells = extract_cells_from_domain(domain)
    material_cell_ids = extract_material_cell_ids(material)

    restart_npz = os.path.join(params.outdir, "mechanics_restart.npz")
    np.savez_compressed(
        restart_npz,
        points=points,
        cells=cells,
        material_cell_ids=material_cell_ids,
        u_vertices=u_vertices,
    )

    with open(os.path.join(params.outdir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2)

    with open(os.path.join(params.outdir, "mechanics_result.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(mechanics_result), f, indent=2)

    log.info("restart: saved %s", restart_npz)
    return restart_npz


def load_mechanics_restart(restart_dir: str) -> Tuple[Any, Any, np.ndarray, ModelParams, Dict[str, Any]]:
    restart_npz = os.path.join(restart_dir, "mechanics_restart.npz")
    params_json = os.path.join(restart_dir, "params.json")
    mechanics_json = os.path.join(restart_dir, "mechanics_result.json")

    if not os.path.exists(restart_npz):
        raise FileNotFoundError(f"Restart file not found: {restart_npz}")
    if not os.path.exists(params_json):
        raise FileNotFoundError(f"Params file not found: {params_json}")

    with open(params_json, "r", encoding="utf-8") as f:
        params = ModelParams(**json.load(f))

    data = np.load(restart_npz)
    domain = create_dolfinx_mesh_3d(data["points"], data["cells"])
    material = make_material_field_from_cell_ids(domain, data["material_cell_ids"])
    u_vertices = data["u_vertices"]

    if os.path.exists(mechanics_json):
        with open(mechanics_json, "r", encoding="utf-8") as f:
            mechanics_result = json.load(f)
    else:
        mechanics_result = {}

    log.info("restart: loaded %s", restart_npz)
    return domain, material, u_vertices, params, mechanics_result


def override_magnetic_params(params: ModelParams, args) -> ModelParams:
    d = asdict(params)
    d["Br_magnetic"] = args.Br
    d["sensor_x"] = args.sensor_x
    d["sensor_y"] = args.sensor_y
    d["sensor_z"] = args.sensor_z
    d["rotate_magnetization"] = not args.no_rotate_magnetization
    return ModelParams(**d)


def summary_columns() -> List[str]:
    return [
        "run", "study", "delta_x_m", "h_cilium_m", "h_substrate_m", "element_degree",
        "nu_pdms", "nu_magnetic", "max_top_u_x_m", "reaction_force_x_N", "reaction_force_x_uN",
        "reaction_error_to_60uN_percent", "J_min", "J_max", "von_mises_min_Pa", "von_mises_max_Pa",
        "volume_substrate_m3", "volume_lower_m3", "volume_upper_m3", "expected_layer_volume_m3",
        "lower_volume_rel_error_percent", "upper_volume_rel_error_percent", "restart_file",
        "Br_magnetic_T", "M_magnetic_A_per_m", "sensor_x_m", "sensor_y_m", "sensor_z_m", "rotate_magnetization",
        "magnetic_dipole_cells", "magnetic_dipole_volume_m3", "magnetic_initial_volume_m3", "magnetic_deformed_volume_m3",
        "magnetic_min_distance_to_sensor_m", "magnetic_skipped_near_cells",
        "B0_sensor_x_T", "B0_sensor_y_T", "B0_sensor_z_T", "B0_sensor_norm_T",
        "B1_sensor_x_T", "B1_sensor_y_T", "B1_sensor_z_T", "B1_sensor_norm_T",
        "dB_sensor_x_T", "dB_sensor_y_T", "dB_sensor_z_T", "dB_sensor_norm_T",
        "B0_sensor_x_uT", "B0_sensor_y_uT", "B0_sensor_z_uT", "B0_sensor_norm_uT",
        "B1_sensor_x_uT", "B1_sensor_y_uT", "B1_sensor_z_uT", "B1_sensor_norm_uT",
        "dB_sensor_x_uT", "dB_sensor_y_uT", "dB_sensor_z_uT", "dB_sensor_norm_uT",
        "B_before_x_T", "B_before_y_T", "B_before_z_T", "B_before_norm_T",
        "B_after_x_T", "B_after_y_T", "B_after_z_T", "B_after_norm_T",
        "delta_B_x_T", "delta_B_y_T", "delta_B_z_T", "delta_B_norm_T",
        "relative_delta_B_norm", "sensor_gap_m", "sensor_x_over_R",
        "Br_T", "gap_m",
        "Bx_before_T", "By_before_T", "Bz_before_T",
        "Bx_after_T", "By_after_T", "Bz_after_T",
        "dBx_T", "dBy_T", "dBz_T",
        "deltaB_norm_T", "relative_deltaB",
        "is_valid", "rank_by_deltaB_norm", "rank_by_abs_dBz",
        "run_params_json", "pvd_file",
    ]


def write_summary(results: List[Dict[str, Any]], summary_path: str) -> None:
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    columns = summary_columns()
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow({key: r.get(key, "") for key in columns})


def print_summary(results: List[Dict[str, Any]], summary_path: str) -> None:
    print("\n=== MECHANICAL AND MAGNETIC VALIDATION SUMMARY ===")
    print(f"summary_csv = {summary_path}")
    for r in results:
        s = (
            f"run={r.get('run', 0):03d}, study={r.get('study', '')}, "
            f"delta={r.get('delta_x_m', 0.0) * 1e3:.3f} mm, "
            f"h_cil={r.get('h_cilium_m', 0.0) * 1e6:.0f} um, "
            f"h_sub={r.get('h_substrate_m', 0.0) * 1e6:.0f} um, "
            f"P{r.get('element_degree', '')}, nu={r.get('nu_pdms', 0.0):.2f}"
        )
        if "reaction_force_x_uN" in r:
            s += f", reaction={r['reaction_force_x_uN']:.3f} uN, err60={r['reaction_error_to_60uN_percent']:.1f}%, J=[{r['J_min']:.4f}, {r['J_max']:.4f}], vm_max={r['von_mises_max_Pa']:.3e} Pa"
        if "dB_sensor_norm_uT" in r:
            s += f", dB=[{r['dB_sensor_x_uT']:.2f}, {r['dB_sensor_y_uT']:.2f}, {r['dB_sensor_z_uT']:.2f}] uT, |dB|={r['dB_sensor_norm_uT']:.2f} uT"
        if "restart_file" in r:
            s += f", restart={r['restart_file']}"
        if "pvd_file" in r:
            s += f", file={r['pvd_file']}"
        print(s)


def run_mechanics_case(params: ModelParams, run_id: int, study: str) -> Dict[str, Any]:
    log.info("=" * 80)
    log.info("RUN %03d | %s | MECHANICS", run_id, study)
    log.info("delta_x = %.6e m", params.delta_x)
    log.info("h_cilium = %.6e m", params.h_cilium)
    log.info("h_substrate = %.6e m", params.h_substrate)
    log.info("element_degree = %d", params.element_degree)
    log.info("nu_pdms = %.4f, nu_magnetic = %.4f", params.nu_pdms, params.nu_magnetic)
    log.info("outdir = %s", params.outdir)
    log.info("=" * 80)

    log_mesh_resolution_warning(params)
    points, cells = build_gmsh_mesh_3d(params)
    domain = create_dolfinx_mesh_3d(points, cells)
    material, E, nu = make_material_field(domain, params)

    material_volumes = compute_material_volumes(domain, material)
    expected_layer_volume = np.pi * params.R**2 * params.L1
    log.info("volume: substrate = %.9e m^3", material_volumes[0])
    log.info("volume: lower PDMS cilium = %.9e m^3", material_volumes[1])
    log.info("volume: upper magnetic cilium = %.9e m^3", material_volumes[2])
    log.info("volume: expected lower/upper layer volume = %.9e m^3", expected_layer_volume)

    uh, max_ux, reaction_force_x, _, _ = solve_hyperelasticity_displacement_control_3d(domain, E, nu, params)
    J_field, J_min, J_max = compute_J_field(domain, uh)
    log.info("validation: min J = %.9e", J_min)
    log.info("validation: max J = %.9e", J_max)

    von_mises, vm_min, vm_max = compute_von_mises_field(domain, uh, E, nu)
    log.info("validation: min von Mises = %.9e Pa", vm_min)
    log.info("validation: max von Mises = %.9e Pa", vm_max)

    vtk_path = save_results(domain, uh, material, E, nu, J_field, von_mises, params)
    u_vertices = evaluate_displacement_at_geometry_vertices(domain, uh)

    reaction_uN = reaction_force_x * 1e6
    reaction_error_percent = 100.0 * (reaction_uN - TARGET_REACTION_U_N) / TARGET_REACTION_U_N
    lower_volume_rel_error_percent = 100.0 * (material_volumes[1] - expected_layer_volume) / expected_layer_volume
    upper_volume_rel_error_percent = 100.0 * (material_volumes[2] - expected_layer_volume) / expected_layer_volume

    result = {
        "run": run_id,
        "study": study,
        "delta_x_m": params.delta_x,
        "h_cilium_m": params.h_cilium,
        "h_substrate_m": params.h_substrate,
        "element_degree": params.element_degree,
        "nu_pdms": params.nu_pdms,
        "nu_magnetic": params.nu_magnetic,
        "max_top_u_x_m": max_ux,
        "reaction_force_x_N": reaction_force_x,
        "reaction_force_x_uN": reaction_uN,
        "reaction_error_to_60uN_percent": reaction_error_percent,
        "J_min": J_min,
        "J_max": J_max,
        "von_mises_min_Pa": vm_min,
        "von_mises_max_Pa": vm_max,
        "volume_substrate_m3": material_volumes[0],
        "volume_lower_m3": material_volumes[1],
        "volume_upper_m3": material_volumes[2],
        "expected_layer_volume_m3": expected_layer_volume,
        "lower_volume_rel_error_percent": lower_volume_rel_error_percent,
        "upper_volume_rel_error_percent": upper_volume_rel_error_percent,
        "pvd_file": vtk_path,
    }

    result["restart_file"] = save_mechanics_restart(domain, material, u_vertices, params, result)
    validate_result_quality(params, result)

    result["_domain"] = domain
    result["_material"] = material
    result["_u_vertices"] = u_vertices
    del uh, E, nu, J_field, von_mises
    gc.collect()
    return result


def run_magnetics_case_from_objects(domain, material, u_vertices: np.ndarray, params: ModelParams, base_result: Dict[str, Any]) -> Dict[str, Any]:
    log.info("=" * 80)
    log.info("MAGNETICS POSTPROCESSING")
    log.info("Br_magnetic = %.6e T", params.Br_magnetic)
    log.info("sensor point = [%.6e, %.6e, %.6e] m", params.sensor_x, params.sensor_y, params.sensor_z)
    log.info("rotate_magnetization = %s", params.rotate_magnetization)
    log.info("=" * 80)

    magnetic_diag = compute_magnetic_dipole_diagnostics(domain, u_vertices, material, params)
    log_magnetic_diagnostics(magnetic_diag)
    result = {k: v for k, v in base_result.items() if not k.startswith("_")}
    result.update(magnetic_diag)
    return result


def run_full_case(params: ModelParams, run_id: int, study: str) -> Dict[str, Any]:
    mechanics_result = run_mechanics_case(params, run_id, study)
    result = run_magnetics_case_from_objects(
        mechanics_result["_domain"], mechanics_result["_material"], mechanics_result["_u_vertices"], params, mechanics_result
    )
    del mechanics_result
    gc.collect()
    return result


def run_magnetics_from_restart(args) -> List[Dict[str, Any]]:
    if args.restart_dir is None:
        raise RuntimeError("--restart-dir is required for --mode magnetics")
    domain, material, u_vertices, saved_params, mechanics_result = load_mechanics_restart(args.restart_dir)
    params = override_magnetic_params(saved_params, args)
    result = run_magnetics_case_from_objects(domain, material, u_vertices, params, mechanics_result)
    result["study"] = "magnetics_from_restart"
    outdir = args.outdir if args.outdir is not None else args.restart_dir
    os.makedirs(outdir, exist_ok=True)
    summary_path = os.path.join(outdir, "magnetic_summary.csv")
    write_summary([result], summary_path)
    print_summary([result], summary_path)
    return [result]


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (float, int, np.floating, np.integer)) and np.isfinite(float(value))


def is_stable_magnetic_result(result: Dict[str, Any], sensor_gap: float) -> bool:
    if not (0.02 <= float(result["Br_magnetic_T"]) <= 0.15):
        return False
    if not (sensor_gap >= 25e-6 and sensor_gap > 0.0):
        return False

    fields = [
        "B0_sensor_x_T", "B0_sensor_y_T", "B0_sensor_z_T", "B0_sensor_norm_T",
        "B1_sensor_x_T", "B1_sensor_y_T", "B1_sensor_z_T", "B1_sensor_norm_T",
        "dB_sensor_x_T", "dB_sensor_y_T", "dB_sensor_z_T", "dB_sensor_norm_T",
    ]
    if any(not is_finite_number(result.get(key)) for key in fields):
        return False
    if any(abs(float(result[key])) > MAX_STABLE_MAGNETIC_FIELD_T for key in fields):
        return False
    return int(result.get("magnetic_skipped_near_cells", 0)) == 0


def add_magnetic_only_aliases(result: Dict[str, Any], sensor_gap: float, sensor_x_over_R: float) -> Dict[str, Any]:
    b0_norm = float(result["B0_sensor_norm_T"])
    dB_norm = float(result["dB_sensor_norm_T"])
    relative_change = dB_norm / b0_norm if abs(b0_norm) > 1e-30 else None
    result.update(
        {
            "B_before_x_T": result["B0_sensor_x_T"],
            "B_before_y_T": result["B0_sensor_y_T"],
            "B_before_z_T": result["B0_sensor_z_T"],
            "B_before_norm_T": result["B0_sensor_norm_T"],
            "B_after_x_T": result["B1_sensor_x_T"],
            "B_after_y_T": result["B1_sensor_y_T"],
            "B_after_z_T": result["B1_sensor_z_T"],
            "B_after_norm_T": result["B1_sensor_norm_T"],
            "delta_B_x_T": result["dB_sensor_x_T"],
            "delta_B_y_T": result["dB_sensor_y_T"],
            "delta_B_z_T": result["dB_sensor_z_T"],
            "delta_B_norm_T": result["dB_sensor_norm_T"],
            "relative_delta_B_norm": relative_change,
            "sensor_gap_m": sensor_gap,
            "sensor_x_over_R": sensor_x_over_R,
            "Br_T": result["Br_magnetic_T"],
            "gap_m": sensor_gap,
            "Bx_before_T": result["B0_sensor_x_T"],
            "By_before_T": result["B0_sensor_y_T"],
            "Bz_before_T": result["B0_sensor_z_T"],
            "Bx_after_T": result["B1_sensor_x_T"],
            "By_after_T": result["B1_sensor_y_T"],
            "Bz_after_T": result["B1_sensor_z_T"],
            "dBx_T": result["dB_sensor_x_T"],
            "dBy_T": result["dB_sensor_y_T"],
            "dBz_T": result["dB_sensor_z_T"],
            "deltaB_norm_T": result["dB_sensor_norm_T"],
            "relative_deltaB": relative_change,
            "is_valid": is_stable_magnetic_result(result, sensor_gap),
        }
    )
    return result


def rank_magnetic_only_results(results: List[Dict[str, Any]]) -> None:
    valid_results = [r for r in results if r.get("is_valid")]

    for rank, result in enumerate(
        sorted(valid_results, key=lambda r: float(r["deltaB_norm_T"]), reverse=True),
        start=1,
    ):
        result["rank_by_deltaB_norm"] = rank

    for rank, result in enumerate(
        sorted(valid_results, key=lambda r: abs(float(r["dBz_T"])), reverse=True),
        start=1,
    ):
        result["rank_by_abs_dBz"] = rank

    for result in results:
        result.setdefault("rank_by_deltaB_norm", "")
        result.setdefault("rank_by_abs_dBz", "")


def write_magnetic_only_run_jsons(
    results: List[Dict[str, Any]],
    base_outdir: str,
    restart_dir: str,
    mechanics_result: Dict[str, Any],
) -> None:
    for result in results:
        run_id = int(result["run"])
        params_json = os.path.join(base_outdir, f"magnetic_only_run_{run_id:03d}.json")
        run_payload = {
            "run": run_id,
            "study": result["study"],
            "restart_dir": restart_dir,
            "restart_file": result["restart_file"],
            "mechanics_result": mechanics_result,
            "params": result.get("_params", {}),
            "magnetic_result": {k: v for k, v in result.items() if k != "run_params_json" and not k.startswith("_")},
        }
        with open(params_json, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(run_payload), f, indent=2)
        result["run_params_json"] = params_json


def write_best_magnetic_configurations(results: List[Dict[str, Any]], base_outdir: str) -> None:
    valid_results = [r for r in results if r.get("is_valid")]
    best_by_norm = next((r for r in valid_results if r.get("rank_by_deltaB_norm") == 1), None)
    best_by_abs_dBz = next((r for r in valid_results if r.get("rank_by_abs_dBz") == 1), None)
    public_best_by_norm = {k: v for k, v in best_by_norm.items() if not k.startswith("_")} if best_by_norm else None
    public_best_by_abs_dBz = {k: v for k, v in best_by_abs_dBz.items() if not k.startswith("_")} if best_by_abs_dBz else None
    output = {
        "criterion": "maximize deltaB_norm_T among valid magnetic-only configurations",
        "validity": {
            "Br_T_range": [0.02, 0.15],
            "min_gap_m": 25e-6,
            "max_abs_field_T": MAX_STABLE_MAGNETIC_FIELD_T,
            "requires_finite_values": True,
            "requires_no_skipped_near_cells": True,
        },
        "best_by_deltaB_norm": public_best_by_norm,
        "best_by_abs_dBz": public_best_by_abs_dBz,
    }
    best_path = os.path.join(base_outdir, "magnetic_best_configurations.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(output), f, indent=2)
    if best_by_norm:
        log.info(
            "best magnetic configuration by |delta_B|: run=%03d, Br=%.3e T, gap=%.3e m, sensor_x=%.3e m, |delta_B|=%.6e T",
            best_by_norm["run"], best_by_norm["Br_T"], best_by_norm["gap_m"],
            best_by_norm["sensor_x_m"], best_by_norm["deltaB_norm_T"],
        )
    if best_by_abs_dBz:
        log.info(
            "best Hall Bz configuration by |dBz|: run=%03d, Br=%.3e T, gap=%.3e m, sensor_x=%.3e m, |dBz|=%.6e T",
            best_by_abs_dBz["run"], best_by_abs_dBz["Br_T"], best_by_abs_dBz["gap_m"],
            best_by_abs_dBz["sensor_x_m"], abs(best_by_abs_dBz["dBz_T"]),
        )


def run_magnetic_only_validation(args) -> List[Dict[str, Any]]:
    restart_dir = args.restart_dir or DEFAULT_MAGNETIC_ONLY_RESTART_DIR
    if not os.path.exists(os.path.join(restart_dir, "mechanics_restart.npz")):
        parent_restart_dir = os.path.join("..", restart_dir)
        if os.path.exists(os.path.join(parent_restart_dir, "mechanics_restart.npz")):
            restart_dir = parent_restart_dir

    domain, material, u_vertices, saved_params, mechanics_result = load_mechanics_restart(restart_dir)
    log.info("magnetic-only: restart_dir = %s", restart_dir)
    log.info("magnetic-only: saved params = %s", asdict(saved_params))
    if mechanics_result:
        log.info(
            "mechanical validation: max_top_u_x=%.9e m, reaction=%.9e N (%.6f uN), J=[%.9e, %.9e], von_mises_max=%.9e Pa",
            mechanics_result.get("max_top_u_x_m", float("nan")),
            mechanics_result.get("reaction_force_x_N", float("nan")),
            mechanics_result.get("reaction_force_x_uN", float("nan")),
            mechanics_result.get("J_min", float("nan")),
            mechanics_result.get("J_max", float("nan")),
            mechanics_result.get("von_mises_max_Pa", float("nan")),
        )
    base_outdir = args.outdir or "magnetic_cilium_3d_results_final"
    if args.outdir == "magnetic_cilium_3d_results_final":
        restart_parent = os.path.dirname(os.path.normpath(restart_dir))
        if os.path.basename(restart_parent) == "magnetic_cilium_3d_results_final":
            base_outdir = restart_parent
    os.makedirs(base_outdir, exist_ok=True)
    log.info(
        "magnetic sweep parameters: Br=%s T, gap=%s m, sensor_x/R=%s",
        MAGNETIC_ONLY_BR_VALUES,
        MAGNETIC_ONLY_SENSOR_GAPS,
        MAGNETIC_ONLY_SENSOR_X_FACTORS,
    )

    results: List[Dict[str, Any]] = []
    run_id = 1
    for Br in MAGNETIC_ONLY_BR_VALUES:
        for sensor_gap in MAGNETIC_ONLY_SENSOR_GAPS:
            for sensor_x_factor in MAGNETIC_ONLY_SENSOR_X_FACTORS:
                params_dict = asdict(saved_params)
                params_dict["Br_magnetic"] = Br
                params_dict["sensor_x"] = sensor_x_factor * saved_params.R
                params_dict["sensor_y"] = 0.0
                params_dict["sensor_z"] = -sensor_gap
                params_dict["rotate_magnetization"] = saved_params.rotate_magnetization
                params_dict["outdir"] = base_outdir
                params = ModelParams(**params_dict)

                result = run_magnetics_case_from_objects(domain, material, u_vertices, params, mechanics_result)
                result["run"] = run_id
                result["study"] = "magnetic_only_validation"
                result["restart_file"] = os.path.join(restart_dir, "mechanics_restart.npz")
                result["_params"] = asdict(params)
                add_magnetic_only_aliases(result, sensor_gap, sensor_x_factor)
                log.info(
                    "magnetic run %03d: Br=%.3e T, gap=%.3e m, sensor=[%.3e, %.3e, %.3e] m",
                    run_id,
                    result["Br_T"],
                    result["gap_m"],
                    result["sensor_x_m"],
                    result["sensor_y_m"],
                    result["sensor_z_m"],
                )
                log.info(
                    "magnetic run %03d: B_before=[%.6e, %.6e, %.6e] T, B_after=[%.6e, %.6e, %.6e] T",
                    run_id,
                    result["Bx_before_T"],
                    result["By_before_T"],
                    result["Bz_before_T"],
                    result["Bx_after_T"],
                    result["By_after_T"],
                    result["Bz_after_T"],
                )
                log.info(
                    "magnetic run %03d: delta_B=[%.6e, %.6e, %.6e] T, |delta_B|=%.6e T, relative=%s, valid=%s",
                    run_id,
                    result["dBx_T"],
                    result["dBy_T"],
                    result["dBz_T"],
                    result["deltaB_norm_T"],
                    result["relative_deltaB"],
                    result["is_valid"],
                )

                results.append(result)
                run_id += 1

    rank_magnetic_only_results(results)
    write_magnetic_only_run_jsons(results, base_outdir, restart_dir, mechanics_result)
    write_best_magnetic_configurations(results, base_outdir)
    summary_path = os.path.join(base_outdir, "magnetic_validation_summary.csv")
    write_summary(results, summary_path)
    print_summary(results, summary_path)
    return results


def run_single_pipeline_case(args) -> List[Dict[str, Any]]:
    if args.mode == "magnetic-only":
        raise RuntimeError("Internal guard: magnetic-only mode must not enter the mechanics/full pipeline.")
    base_outdir = args.outdir
    os.makedirs(base_outdir, exist_ok=True)
    run_name = (
        f"final_P2_hcil_{args.h_cilium * 1e6:.0f}um_"
        f"hsub_{args.h_substrate * 1e6:.0f}um_"
        f"delta_{args.delta * 1e3:.3f}mm"
    ).replace(".", "p")
    params = make_params_from_args(args, os.path.join(base_outdir, run_name))
    if args.mode == "mechanics":
        result = run_mechanics_case(params, 1, "final_candidate_run_mechanics_only")
        result = {k: v for k, v in result.items() if not k.startswith("_")}
    else:
        result = run_full_case(params, 1, "final_candidate_run")
    return [result]


def run_validation_study(args) -> List[Dict[str, Any]]:
    base_outdir = args.outdir
    os.makedirs(base_outdir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    run_id = 1

    delta_values = [0.20e-3, 0.25e-3, 0.30e-3, 0.32e-3, 0.35e-3, 0.40e-3]
    for delta_x in delta_values:
        run_name = f"calib_P2_hcil_30um_hsub_100um_delta_{delta_x * 1e3:.3f}mm".replace(".", "p")
        local_args = args
        local_args.delta = delta_x
        params = make_params_from_args(local_args, os.path.join(base_outdir, run_name))
        result = run_full_case(params, run_id, "calib_P2_hcil_30um_hsub_100um_delta_")
        results.append(result)
        run_id += 1

    delta_for_mesh_check = args.mesh_delta
    h_cilium_values = [60e-6, 45e-6, 30e-6]
    if args.include_h20:
        h_cilium_values.append(20e-6)
    for h_cilium in h_cilium_values:
        run_name = f"mesh_P2_hcil_{h_cilium * 1e6:.0f}um_hsub_{args.h_substrate * 1e6:.0f}um_delta_{delta_for_mesh_check * 1e3:.3f}mm".replace(".", "p")
        local_args = args
        local_args.delta = delta_for_mesh_check
        local_args.h_cilium = h_cilium
        params = make_params_from_args(local_args, os.path.join(base_outdir, run_name))
        result = run_full_case(params, run_id, "mesh_convergence")
        results.append(result)
        run_id += 1

    for nu_value in [0.45, 0.47, 0.49]:
        run_name = f"nu_check_P2_h60um_nu_{nu_value:.2f}_delta_{delta_for_mesh_check * 1e3:.3f}mm".replace(".", "p")
        local_args = args
        local_args.delta = delta_for_mesh_check
        local_args.nu = nu_value
        params = make_params_from_args(local_args, os.path.join(base_outdir, run_name))
        result = run_full_case(params, run_id, "poisson_ratio_sensitivity")
        results.append(result)
        run_id += 1

    return results
