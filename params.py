import argparse
import logging
import sys
from dataclasses import dataclass

import dolfinx
from mpi4py import MPI


TARGET_REACTION_U_N = 60.0


@dataclass(frozen=True)
class ModelParams:
    # Geometry
    D: float = 120e-6
    L1: float = 2e-3
    L2: float = 2e-3
    substrate_radius: float = 0.60e-3
    substrate_thickness: float = 0.50e-3

    # Materials
    E_pdms: float = 1.5e6
    nu_pdms: float = 0.49
    E_magnetic: float = 16.6e6
    nu_magnetic: float = 0.49

    # Mesh
    h_cilium: float = 30e-6
    h_substrate: float = 100e-6
    element_degree: int = 2

    # Mechanics
    delta_x: float = 0.32e-3
    n_steps: int = 12

    # Magnetics
    Br_magnetic: float = 0.10
    sensor_x: float = 0.0
    sensor_y: float = 0.0
    sensor_z: float = -50e-6
    rotate_magnetization: bool = True

    # Output
    outdir: str = "magnetic_cilium_3d_results"

    @property
    def R(self) -> float:
        return self.D / 2.0

    @property
    def L(self) -> float:
        return self.L1 + self.L2


def check_runtime(log: logging.Logger) -> None:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run serially only: python main.py")

    log.info("Python: %s", sys.version.split()[0])
    log.info("DOLFINx: %s", dolfinx.__version__)

    if not dolfinx.__version__.startswith("0.10."):
        log.warning("Expected DOLFINx 0.10.x, got %s", dolfinx.__version__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline 3D hyperelastic FEM + dipole magnetic postprocessor for a double-layer magnetic cilium."
    )

    parser.add_argument(
        "--mode",
        choices=["validation", "final", "full", "mechanics", "magnetics"],
        default="validation",
        help="validation: sweeps; full/final: mechanics+magnetics; mechanics: save restart; magnetics: read restart.",
    )
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--restart-dir", default=None)

    # Geometry
    parser.add_argument("--D", type=float, default=120e-6, help="Cilium diameter, m.")
    parser.add_argument("--L1", type=float, default=2e-3, help="Lower PDMS layer length, m.")
    parser.add_argument("--L2", type=float, default=2e-3, help="Upper magnetic layer length, m.")
    parser.add_argument("--substrate-radius", type=float, default=0.60e-3, help="Substrate radius, m.")
    parser.add_argument("--substrate-thickness", type=float, default=0.50e-3, help="Substrate thickness, m.")

    # Mechanics and mesh
    parser.add_argument("--delta", type=float, default=0.32e-3)
    parser.add_argument("--mesh-delta", type=float, default=0.32e-3)
    parser.add_argument("--h-cilium", type=float, default=30e-6)
    parser.add_argument("--h-substrate", type=float, default=100e-6)
    parser.add_argument("--nu", type=float, default=0.49)
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--include-h20", action="store_true")

    # Magnetics
    parser.add_argument("--Br", type=float, default=0.10, help="Effective remanence, T.")
    parser.add_argument("--sensor-x", type=float, default=0.0)
    parser.add_argument("--sensor-y", type=float, default=0.0)
    parser.add_argument("--sensor-z", type=float, default=-50e-6, help="Use --sensor-z=-50e-6 for negative values.")
    parser.add_argument("--no-rotate-magnetization", action="store_true")

    args = parser.parse_args()

    if args.mode == "final":
        args.mode = "full"

    if args.outdir is None:
        if args.mode == "validation":
            args.outdir = "magnetic_cilium_3d_results_validation"
        elif args.mode == "magnetics":
            args.outdir = args.restart_dir
        else:
            args.outdir = "magnetic_cilium_3d_results_final"

    return args


def make_params_from_args(args, outdir: str) -> ModelParams:
    return ModelParams(
        D=args.D,
        L1=args.L1,
        L2=args.L2,
        substrate_radius=args.substrate_radius,
        substrate_thickness=args.substrate_thickness,
        E_pdms=1.5e6,
        nu_pdms=args.nu,
        E_magnetic=16.6e6,
        nu_magnetic=args.nu,
        h_cilium=args.h_cilium,
        h_substrate=args.h_substrate,
        element_degree=2,
        delta_x=args.delta,
        n_steps=args.n_steps,
        Br_magnetic=args.Br,
        sensor_x=args.sensor_x,
        sensor_y=args.sensor_y,
        sensor_z=args.sensor_z,
        rotate_magnetization=not args.no_rotate_magnetization,
        outdir=outdir,
    )
