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


log = logging.getLogger("magnetic_cilium_3d")


def build_air_box_mesh(params: ModelParams, air_radius: float, air_below: float, air_above: float, h_air: float):
    """Build the first magnetostatic FEM baseline air box.

    This uses phi=0 on the outer boundary as an approximation of an open
    magnetic domain. Accuracy depends on air box size and h_air.
    """
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
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_air)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_air)
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
    tagged = 0
    for c, point in enumerate(centers):
        candidates = np.where(np.all((point >= bmin) & (point <= bmax), axis=1))[0]
        for idx in candidates:
            if point_in_tet(point, source_tets[idx]):
                dof = Q.dofmap.cell_dofs(int(c))[0]
                indicator.x.array[dof] = 1.0
                M_array[c, :] = source_M[idx]
                tagged += 1
                break

    indicator.x.scatter_forward()
    M_field.x.scatter_forward()
    log.info("magnetics-fem: tagged magnetic source cells = %d", tagged)
    return M_field, indicator, tagged


def solve_magnetostatic_scalar_potential(
        magnetic_domain,
        M_field,
        magnetic_indicator,
        sensor_point,
        params: ModelParams,
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

    fdim = magnetic_domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        magnetic_domain,
        fdim,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)

    problem = fem_petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
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

    grad_phi = gradient_at_point(magnetic_domain, V, phi_h, np.asarray(sensor_point, dtype=np.float64))
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


def compute_magnetostatic_fem_diagnostics(mechanics_domain, material, u_vertices, saved_params: ModelParams, mechanics_result: Dict[str, Any], args) -> Dict[str, Any]:
    params_dict = asdict(saved_params)
    params_dict["Br_magnetic"] = args.Br
    params_dict["sensor_x"] = args.sensor_x
    params_dict["sensor_y"] = args.sensor_y
    params_dict["sensor_z"] = args.sensor_z
    params_dict["rotate_magnetization"] = not args.no_rotate_magnetization
    params = ModelParams(**params_dict)

    sensor_point = np.array([params.sensor_x, params.sensor_y, params.sensor_z], dtype=np.float64)
    initial_tets, initial_M = mechanical_magnetic_tets(mechanics_domain, material, u_vertices, params, deformed=False)
    deformed_tets, deformed_M = mechanical_magnetic_tets(mechanics_domain, material, u_vertices, params, deformed=True)
    air_radius, air_below, air_above = compute_air_box_dimensions(params, initial_tets, deformed_tets, args)

    log.info(
        "magnetics-fem: air_radius=%.6e m, air_below=%.6e m, air_above=%.6e m, h_air=%.6e m",
        air_radius, air_below, air_above, args.h_air,
    )
    log.info(
        "magnetics-fem: first FEM baseline; phi=0 outer boundary approximates open magnetic space. Accuracy depends on air box size and h_air."
    )

    magnetic_domain = build_air_box_mesh(params, air_radius, air_below, air_above, args.h_air)

    M0, indicator0, tagged0 = build_source_fields(magnetic_domain, initial_tets, initial_M)
    _, B0 = solve_magnetostatic_scalar_potential(
        magnetic_domain, M0, indicator0, sensor_point, params,
        petsc_options_prefix="magnetostatic_initial_"
    
    )

    M1, indicator1, tagged1 = build_source_fields(magnetic_domain, deformed_tets, deformed_M)
    _, B1 = solve_magnetostatic_scalar_potential(
        magnetic_domain, M1, indicator1, sensor_point, params,
        petsc_options_prefix="magnetostatic_deformed_"
    )

    dB = B1 - B0
    reaction_uN = float(mechanics_result.get("reaction_force_x_uN", 0.0) or 0.0)

    diagnostics = {
        "Br_magnetic_T": params.Br_magnetic,
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
        "initial_source_cells": tagged0,
        "deformed_source_cells": tagged1,
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
