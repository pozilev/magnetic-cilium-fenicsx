import logging
from dataclasses import asdict
from typing import Any, Dict, Tuple

import basix.ufl
import gmsh
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, geometry, mesh, default_scalar_type
from dolfinx.fem import petsc as fem_petsc

from magnetics_dipoles import compute_cell_deformation_gradient_P1, rotation_from_deformation_gradient
from mechanics_model import create_dolfinx_mesh_3d
from params import ModelParams
from sensor_sampling import make_sensor_sample_points


log = logging.getLogger("magnetic_cilium_3d")


def build_air_box_mesh(
        params: ModelParams,
        air_radius: float,
        air_below: float,
        air_above: float,
        h_air: float,
        h_air_near: float,
        h_air_far: float,
        near_radius: float,
        max_air_cells: int,
        allow_large_air_mesh: bool,
    ):
    """Build a tetrahedral air box for the magnetostatic FEM postprocessor."""
    gmsh.initialize()
    gmsh.model.add("magnetic_cilium_air_box")
    try:
        gmsh.model.occ.addBox(
            -air_radius,
            -air_radius,
            -air_below,
            2.0 * air_radius,
            2.0 * air_radius,
            params.L + air_above + air_below,
        )
        gmsh.model.occ.synchronize()
        h_near = float(h_air_near)
        h_far = max(float(h_air_far), h_near)
        near_radius = float(near_radius)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_near)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_far)
        box_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(box_field, "VIn", h_near)
        gmsh.model.mesh.field.setNumber(box_field, "VOut", h_far)
        gmsh.model.mesh.field.setNumber(box_field, "XMin", -near_radius)
        gmsh.model.mesh.field.setNumber(box_field, "XMax", near_radius)
        gmsh.model.mesh.field.setNumber(box_field, "YMin", -near_radius)
        gmsh.model.mesh.field.setNumber(box_field, "YMax", near_radius)
        gmsh.model.mesh.field.setNumber(box_field, "ZMin", -air_below)
        gmsh.model.mesh.field.setNumber(box_field, "ZMax", params.L + air_above)
        gmsh.model.mesh.field.setAsBackgroundMesh(box_field)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        points = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
        tet_nodes = None
        for etype, enodes in zip(elem_types, elem_node_tags):
            _, dim, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
            if dim == 3 and num_nodes == 4:
                tet_nodes = np.asarray(enodes, dtype=np.int64).reshape(-1, 4)
                break
        if tet_nodes is None:
            raise RuntimeError("No linear tetrahedral cells were generated for magnetic FEM.")

        tag_to_local = {int(tag): i for i, tag in enumerate(node_tags)}
        cells = np.array([[tag_to_local[int(t)] for t in tet] for tet in tet_nodes], dtype=np.int64)
        log.info("magnetics-fem: air box nodes=%d, tetrahedra=%d", points.shape[0], cells.shape[0])
        if cells.shape[0] > max_air_cells and not allow_large_air_mesh:
            raise RuntimeError(
                "Air mesh too large for default laptop-safe mode: "
                f"{cells.shape[0]} tetrahedra > max_air_cells={max_air_cells}. "
                "Increase --h-air, reduce air box factors, or pass --allow-large-air-mesh."
            )
        return create_dolfinx_mesh_3d(points, cells)
    finally:
        gmsh.finalize()


def mechanical_magnetic_tets(mechanics_domain, material, u_vertices: np.ndarray, params: ModelParams, deformed: bool):
    mu0 = 4.0 * np.pi * 1e-7
    M0_ref = np.array([0.0, 0.0, params.Br_magnetic / mu0], dtype=np.float64)

    tdim = mechanics_domain.topology.dim
    mechanics_domain.topology.create_connectivity(tdim, 0)
    c_to_v = mechanics_domain.topology.connectivity(tdim, 0)
    X_vertices = mechanics_domain.geometry.x[:, :3].copy()
    Q = material.function_space

    tet_points = []
    tet_M = []
    num_cells = mechanics_domain.topology.index_map(tdim).size_local
    for c in range(num_cells):
        dof = Q.dofmap.cell_dofs(int(c))[0]
        if int(round(float(material.x.array[dof]))) != 2:
            continue

        vertex_ids = c_to_v.links(int(c))
        X = X_vertices[vertex_ids]
        uX = u_vertices[vertex_ids]
        if X.shape[0] != 4:
            continue

        if deformed:
            tet = X + uX
            if params.rotate_magnetization:
                F = compute_cell_deformation_gradient_P1(X, uX)
                M = rotation_from_deformation_gradient(F) @ M0_ref
            else:
                M = M0_ref.copy()
        else:
            tet = X
            M = M0_ref.copy()

        tet_points.append(tet)
        tet_M.append(M)

    if not tet_points:
        raise RuntimeError("No magnetic mechanical tetrahedra found in restart.")

    tets = np.asarray(tet_points, dtype=np.float64)
    M_values = np.asarray(tet_M, dtype=np.float64)
    return tets, M_values


def point_in_tet(point: np.ndarray, tet: np.ndarray, tol: float = 1e-10) -> bool:
    A = np.vstack([tet[1] - tet[0], tet[2] - tet[0], tet[3] - tet[0]]).T
    try:
        lambdas = np.linalg.solve(A, point - tet[0])
    except np.linalg.LinAlgError:
        return False
    l0 = 1.0 - float(np.sum(lambdas))
    bary = np.array([l0, lambdas[0], lambdas[1], lambdas[2]], dtype=np.float64)
    return bool(np.all(bary >= -tol) and np.all(bary <= 1.0 + tol))


def cell_volume_from_vertices(vertices: np.ndarray) -> float:
    return abs(float(np.linalg.det(np.vstack([
        vertices[1] - vertices[0],
        vertices[2] - vertices[0],
        vertices[3] - vertices[0],
    ])))) / 6.0


def tetrahedral_cell_volumes(domain) -> np.ndarray:
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    c_to_v = domain.topology.connectivity(tdim, 0)
    points = domain.geometry.x[:, :3]
    num_cells = domain.topology.index_map(tdim).size_local
    volumes = np.zeros(num_cells, dtype=np.float64)
    for c in range(num_cells):
        vertices = points[c_to_v.links(int(c))]
        volumes[c] = cell_volume_from_vertices(vertices)
    return volumes


def build_source_fields(magnetic_domain, source_tets: np.ndarray, source_M: np.ndarray):
    """Tag magnetic FEM cells by center-in-deformed-tetra membership.

    This is a deliberately simple first baseline. It uses bounding boxes before
    barycentric checks; accuracy depends on h_air and source discretization.
    """
    tdim = magnetic_domain.topology.dim
    cell_ids = np.arange(magnetic_domain.topology.index_map(tdim).size_local, dtype=np.int32)
    centers = mesh.compute_midpoints(magnetic_domain, tdim, cell_ids)

    bmin = np.min(source_tets, axis=1)
    bmax = np.max(source_tets, axis=1)

    Q = fem.functionspace(magnetic_domain, ("DG", 0))
    V0 = fem.functionspace(magnetic_domain, ("DG", 0, (3,)))
    indicator = fem.Function(Q)
    indicator.name = "magnetic_region"
    M_field = fem.Function(V0)
    M_field.name = "magnetization"

    M_array = M_field.x.array.reshape((-1, 3))
    cell_volumes = tetrahedral_cell_volumes(magnetic_domain)
    tagged = 0
    tagged_volume = 0.0
    for c, point in enumerate(centers):
        candidates = np.where(np.all((point >= bmin) & (point <= bmax), axis=1))[0]
        for idx in candidates:
            if point_in_tet(point, source_tets[idx]):
                dof = Q.dofmap.cell_dofs(int(c))[0]
                indicator.x.array[dof] = 1.0
                M_array[c, :] = source_M[idx]
                tagged += 1
                tagged_volume += cell_volumes[c]
                break

    indicator.x.scatter_forward()
    M_field.x.scatter_forward()
    log.info("magnetics-fem: tagged magnetic source cells = %d, volume=%.6e m^3", tagged, tagged_volume)
    return M_field, indicator, tagged, tagged_volume


def solve_magnetostatic_scalar_potential(
        magnetic_domain,
        M_field,
        magnetic_indicator,
        sensor_points,
        params: ModelParams,
        magnetic_boundary: str,
        petsc_options_prefix: str = "magnetostatic_",
    ):
    """Solve div(-grad(phi) + M)=0 using scalar potential.

    Weak form sign convention:
        int grad(phi).grad(v) dx = int M.grad(v) dx
    so B_air(sensor) = -mu0 grad(phi)(sensor).
    """
    V = fem.functionspace(magnetic_domain, ("Lagrange", 1))
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L = ufl.inner(M_field, ufl.grad(v)) * ufl.dx

    bcs = []
    if magnetic_boundary == "dirichlet_zero":
        fdim = magnetic_domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            magnetic_domain,
            fdim,
            lambda x: np.full(x.shape[1], True, dtype=bool),
        )
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bcs = [fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)]
    elif magnetic_boundary == "natural":
        gauge_dof = np.array([0], dtype=np.int32)
        bcs = [fem.dirichletbc(default_scalar_type(0.0), gauge_dof, V)]
    else:
        raise RuntimeError(f"Unknown magnetic boundary mode: {magnetic_boundary}")

    problem = fem_petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=petsc_options_prefix,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1e-8,
            "ksp_atol": 1e-12,
            "ksp_max_it": 2000,
        }
    )
    phi_h = problem.solve()
    phi_h.name = "magnetic_scalar_potential"
    phi_h.x.scatter_forward()

    gradients = []
    for point in np.asarray(sensor_points, dtype=np.float64):
        gradients.append(gradient_at_point(magnetic_domain, V, phi_h, point))
    grad_phi = np.mean(np.asarray(gradients, dtype=np.float64), axis=0)
    B_sensor = -(4.0 * np.pi * 1e-7) * grad_phi
    return phi_h, B_sensor


def find_cell_containing_point(domain, point: np.ndarray) -> int:
    points = np.asarray([point], dtype=np.float64)
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)
    cells = colliding.links(0)
    if len(cells) == 0:
        raise RuntimeError(f"Sensor point is outside magnetic FEM mesh: {point.tolist()}")
    return int(cells[0])


def gradient_at_point(domain, V, phi_h, point: np.ndarray) -> np.ndarray:
    cell = find_cell_containing_point(domain, point)
    dofs = V.dofmap.cell_dofs(cell)
    coords = V.tabulate_dof_coordinates()[dofs, :3]
    values = np.real(phi_h.x.array[dofs])
    A = np.vstack([coords[1] - coords[0], coords[2] - coords[0], coords[3] - coords[0]]).T
    b = np.array([values[1] - values[0], values[2] - values[0], values[3] - values[0]], dtype=np.float64)
    return np.linalg.solve(A.T, b)


def compute_air_box_dimensions(params: ModelParams, initial_tets: np.ndarray, deformed_tets: np.ndarray, args) -> Tuple[float, float, float]:
    requested_radius = args.air_radius_factor * params.R
    requested_below = args.air_below_factor * params.R
    requested_above = args.air_above_factor * params.R

    all_tets = np.vstack([initial_tets.reshape((-1, 3)), deformed_tets.reshape((-1, 3))])
    source_xy_radius = float(np.max(np.linalg.norm(all_tets[:, :2], axis=1)))
    source_z_min = float(np.min(all_tets[:, 2]))
    source_z_max = float(np.max(all_tets[:, 2]))

    air_radius = max(requested_radius, source_xy_radius + params.R)
    air_below = max(requested_below, -source_z_min + params.R, -float(args.sensor_z) + params.R)
    air_above = max(requested_above, source_z_max - params.L + params.R)
    return air_radius, air_below, air_above


def compute_near_field_radius(params: ModelParams, initial_tets: np.ndarray, deformed_tets: np.ndarray, sensor_point: np.ndarray, args) -> float:
    all_source_points = np.vstack([initial_tets.reshape((-1, 3)), deformed_tets.reshape((-1, 3))])
    source_xy_radius = float(np.max(np.linalg.norm(all_source_points[:, :2], axis=1)))
    sensor_xy_radius = float(np.linalg.norm(sensor_point[:2])) + (
        float(args.sensor_average_radius) if bool(args.sensor_average) else 0.0
    )
    return max(
        float(args.near_radius_factor) * params.R,
        source_xy_radius + float(args.near_source_padding_factor) * params.R,
        sensor_xy_radius + float(args.near_sensor_padding_factor) * params.R,
    )


def compute_magnetostatic_fem_diagnostics(mechanics_domain, material, u_vertices, saved_params: ModelParams, mechanics_result: Dict[str, Any], args) -> Dict[str, Any]:
    params_dict = asdict(saved_params)
    params_dict["Br_magnetic"] = args.Br
    params_dict["sensor_x"] = args.sensor_x
    params_dict["sensor_y"] = args.sensor_y
    params_dict["sensor_z"] = args.sensor_z
    params_dict["rotate_magnetization"] = not args.no_rotate_magnetization
    params = ModelParams(**params_dict)

    sensor_point = np.array([params.sensor_x, params.sensor_y, params.sensor_z], dtype=np.float64)
    sensor_points = make_sensor_sample_points(
        sensor_point,
        bool(args.sensor_average),
        float(args.sensor_average_radius),
        int(args.sensor_average_n),
    )
    initial_tets, initial_M = mechanical_magnetic_tets(mechanics_domain, material, u_vertices, params, deformed=False)
    deformed_tets, deformed_M = mechanical_magnetic_tets(mechanics_domain, material, u_vertices, params, deformed=True)
    air_radius, air_below, air_above = compute_air_box_dimensions(params, initial_tets, deformed_tets, args)
    near_radius = compute_near_field_radius(params, initial_tets, deformed_tets, sensor_point, args)
    near_radius = min(near_radius, air_radius)

    log.info(
        "magnetics-fem: air_radius=%.6e m, air_below=%.6e m, air_above=%.6e m, h_air=%.6e m, h_near=%.6e m, h_far=%.6e m, near_radius=%.6e m, boundary=%s",
        air_radius, air_below, air_above, args.h_air, args.h_air_near, args.h_air_far, near_radius, args.magnetic_boundary,
    )
    log.info(
        "magnetics-fem: sensor_average=%s, sample_points=%d, max_air_cells=%d",
        bool(args.sensor_average), sensor_points.shape[0], int(args.max_air_cells),
    )

    magnetic_domain = build_air_box_mesh(
        params,
        air_radius,
        air_below,
        air_above,
        args.h_air,
        args.h_air_near,
        args.h_air_far,
        near_radius,
        int(args.max_air_cells),
        bool(args.allow_large_air_mesh),
    )
    air_cells = int(magnetic_domain.topology.index_map(magnetic_domain.topology.dim).size_local)
    air_vertices = int(magnetic_domain.topology.index_map(0).size_local)

    M0, indicator0, tagged0, source_volume0 = build_source_fields(magnetic_domain, initial_tets, initial_M)
    _, B0 = solve_magnetostatic_scalar_potential(
        magnetic_domain, M0, indicator0, sensor_points, params, args.magnetic_boundary,
        petsc_options_prefix="magnetostatic_initial_"
    
    )

    M1, indicator1, tagged1, source_volume1 = build_source_fields(magnetic_domain, deformed_tets, deformed_M)
    _, B1 = solve_magnetostatic_scalar_potential(
        magnetic_domain, M1, indicator1, sensor_points, params, args.magnetic_boundary,
        petsc_options_prefix="magnetostatic_deformed_"
    )

    dB = B1 - B0
    reaction_uN = float(mechanics_result.get("reaction_force_x_uN", 0.0) or 0.0)
    reference_volume = float(mechanics_result.get("volume_upper_m3", 0.0) or 0.0)
    if reference_volume <= 0.0:
        reference_volume = float(np.pi * params.R**2 * params.L2)
    source_volume_error0 = 100.0 * abs(source_volume0 - reference_volume) / reference_volume
    source_volume_error1 = 100.0 * abs(source_volume1 - reference_volume) / reference_volume
    source_projection_ok = max(source_volume_error0, source_volume_error1) <= 5.0
    source_projection_warning = max(source_volume_error0, source_volume_error1) > 20.0
    if source_volume_error0 > 20.0 or source_volume_error1 > 20.0:
        log.warning(
            "Magnetic source volume projection error is large: initial=%.3f%%, deformed=%.3f%%",
            source_volume_error0, source_volume_error1,
        )

    diagnostics = {
        "Br_magnetic_T": params.Br_magnetic,
        "magnetic_boundary": args.magnetic_boundary,
        "sensor_average": bool(args.sensor_average),
        "sensor_average_radius_m": float(args.sensor_average_radius),
        "sensor_average_n": int(args.sensor_average_n),
        "sensor_average_points_used": int(sensor_points.shape[0]),
        "sensor_x_m": params.sensor_x,
        "sensor_y_m": params.sensor_y,
        "sensor_z_m": params.sensor_z,
        "sensor_x_over_R": params.sensor_x / params.R if params.R > 0.0 else "",
        "air_radius_factor": args.air_radius_factor,
        "air_below_factor": args.air_below_factor,
        "air_above_factor": args.air_above_factor,
        "air_radius_m": air_radius,
        "air_below_m": air_below,
        "air_above_m": air_above,
        "h_air_m": args.h_air,
        "h_air_near_m": args.h_air_near,
        "h_air_far_m": args.h_air_far,
        "near_radius_factor": args.near_radius_factor,
        "near_source_padding_factor": args.near_source_padding_factor,
        "near_sensor_padding_factor": args.near_sensor_padding_factor,
        "near_radius_m": near_radius,
        "near_radius_over_R": near_radius / params.R if params.R > 0.0 else "",
        "air_cells": air_cells,
        "air_vertices": air_vertices,
        "initial_source_cells": tagged0,
        "deformed_source_cells": tagged1,
        "source_cells_initial": tagged0,
        "source_cells_deformed": tagged1,
        "source_volume_initial_m3": source_volume0,
        "source_volume_deformed_m3": source_volume1,
        "reference_magnetic_volume_m3": reference_volume,
        "source_volume_error_initial_percent": source_volume_error0,
        "source_volume_error_deformed_percent": source_volume_error1,
        "source_projection_ok": source_projection_ok,
        "source_projection_warning": source_projection_warning,
        "B0_sensor_x_uT": B0[0] * 1e6,
        "B0_sensor_y_uT": B0[1] * 1e6,
        "B0_sensor_z_uT": B0[2] * 1e6,
        "B0_sensor_norm_uT": float(np.linalg.norm(B0)) * 1e6,
        "B1_sensor_x_uT": B1[0] * 1e6,
        "B1_sensor_y_uT": B1[1] * 1e6,
        "B1_sensor_z_uT": B1[2] * 1e6,
        "B1_sensor_norm_uT": float(np.linalg.norm(B1)) * 1e6,
        "dB_sensor_x_uT": dB[0] * 1e6,
        "dB_sensor_y_uT": dB[1] * 1e6,
        "dB_sensor_z_uT": dB[2] * 1e6,
        "dB_sensor_norm_uT": float(np.linalg.norm(dB)) * 1e6,
        "reaction_force_x_uN": reaction_uN,
        "sensitivity_x_uT_per_uN": (dB[0] * 1e6 / reaction_uN) if reaction_uN > 0.0 else "",
        "sensitivity_z_uT_per_uN": (dB[2] * 1e6 / reaction_uN) if reaction_uN > 0.0 else "",
        "sensitivity_norm_uT_per_uN": (float(np.linalg.norm(dB)) * 1e6 / reaction_uN) if reaction_uN > 0.0 else "",
    }
    return diagnostics
