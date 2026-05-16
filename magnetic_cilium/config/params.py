import argparse
import logging
import os
import sys
from dataclasses import dataclass

try:
    import dolfinx
    from mpi4py import MPI
except ModuleNotFoundError:
    dolfinx = None
    MPI = None


TARGET_REACTION_U_N = 60.0


def default_results_path(*parts: str) -> str:
    """Return a results path outside the source tree when run from this repo."""
    base = "../results" if os.path.basename(os.getcwd()) == "magnetic_cilium_pipeline" else "results"
    return os.path.join(base, *parts)


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
    magnetization_model: str | None = None
    theta_mu_rad: float = 0.0
    follow_factor_alpha: float = 1.0
    dipole_mode: str = "tetrahedral"
    n_point_dipoles: int = 8
    compare_magnetization_models: bool = False
    comparison_magnetization_models: tuple[str, ...] = ()
    comparison_dipole_modes: tuple[str, ...] = ()
    plot_magnetization_hall: bool = False

    # Output
    outdir: str = default_results_path("magnetic_cilium_3d_results")
    save_mechanics_frames: bool = False
    mechanics_frames_every: int = 1
    mechanics_animation: bool = False

    @property
    def R(self) -> float:
        return self.D / 2.0

    @property
    def L(self) -> float:
        return self.L1 + self.L2


def check_runtime(log: logging.Logger) -> None:
    if dolfinx is None or MPI is None:
        raise RuntimeError("DOLFINx/MPI runtime is not available. Activate the FEniCSx environment before running solves.")

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run serially only: python main.py")

    log.info("Python: %s", sys.version.split()[0])
    log.info("DOLFINx: %s", dolfinx.__version__)

    if not dolfinx.__version__.startswith("0.10."):
        log.warning("Expected DOLFINx 0.10.x, got %s", dolfinx.__version__)


def load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required for --config. Install the pyyaml package.") from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Config must be a YAML mapping: {path}")

    config = {str(key).replace("-", "_"): value for key, value in data.items()}
    if {"global", "fixed_physical_parameters", "sweep"}.issubset(config):
        global_config = config.get("global") or {}
        return {
            "mode": "magnetics-fem-validation",
            "restart_dir": global_config.get("restart_dir"),
            "outdir": global_config.get("output_dir"),
        }
    if "Br_magnetic" in config and "Br" not in config:
        config["Br"] = config["Br_magnetic"]
    if "delta_x" in config and "delta" not in config:
        config["delta"] = config["delta_x"]
    if "rotate_magnetization" in config and "no_rotate_magnetization" not in config:
        config["no_rotate_magnetization"] = not bool(config["rotate_magnetization"])
    if "theta_mu_deg" in config and "theta_mu_rad" not in config:
        config["theta_mu_rad"] = float(config["theta_mu_deg"]) * 3.141592653589793 / 180.0
    return config


def load_config_defaults(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_yaml_config(config_path)


def resolve_config_path(config_path: str | None) -> str | None:
    if config_path is None or os.path.exists(config_path):
        return config_path
    configs_path = os.path.join("configs", config_path)
    if os.path.exists(configs_path):
        return configs_path
    return config_path


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="YAML config file with CLI defaults.")
    pre_args, _ = pre_parser.parse_known_args()
    pre_args.config = resolve_config_path(pre_args.config)
    config_defaults = load_config_defaults(pre_args.config)

    def cfg(name, default):
        return config_defaults.get(name, default)

    def cli_has(option: str) -> bool:
        return any(arg == option or arg.startswith(option + "=") for arg in sys.argv[1:])

    def cfg_bool(name, default):
        value = cfg(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def str_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    parser = argparse.ArgumentParser(
        description="Pipeline 3D hyperelastic FEM + dipole magnetic postprocessor for a double-layer magnetic cilium.",
        parents=[pre_parser],
    )

    parser.add_argument(
        "--mode",
        choices=[
            "validation", "final", "full", "mechanics", "magnetics", "magnetic-only",
            "magnetic-validation", "magnetics-fem", "magnetics-fem-validation",
            "magnetic-sensor-position-sweep", "interpolate-magnetic-results",
        ],
        default=cfg("mode", "validation"),
        help="validation: sweeps; full/final: mechanics+magnetics; mechanics: save restart; magnetics/magnetic-only: read restart.",
    )
    parser.add_argument("--outdir", default=cfg("outdir", None))
    parser.add_argument("--restart-dir", default=cfg("restart_dir", None))
    parser.add_argument("--results-write-mode", choices=["debug", "experiment"], default=cfg("results_write_mode", "debug"))
    parser.add_argument("--experiment-id", default=cfg("experiment_id", None))
    parser.add_argument("--master-csv-path", default=cfg("master_csv_path", default_results_path("magnetic_results_master.csv")))
    parser.add_argument("--local-summary-path", default=cfg("local_summary_path", None))
    parser.add_argument("--input-csv", default=cfg("input_csv", default_results_path("magnetic_results_master.csv")))
    parser.add_argument("--model-type", default=cfg("model_type", None))
    parser.add_argument("--mode-filter", default=cfg("mode_filter", None))
    parser.add_argument("--x-column", default=cfg("x_column", "sensor_x_over_R"))
    parser.add_argument("--z-column", default=cfg("z_column", "sensor_z_over_R"))
    parser.add_argument("--target-column", default=cfg("target_column", "abs_dBx_uT"))
    parser.add_argument("--reliable-only", type=str_bool, default=cfg_bool("reliable_only", True))
    parser.add_argument("--output-dir", dest="interpolation_output_dir", default=cfg("output_dir", cfg("interpolation_output_dir", None)))

    # Geometry
    parser.add_argument("--D", type=float, default=cfg("D", 120e-6), help="Cilium diameter, m.")
    parser.add_argument("--L1", type=float, default=cfg("L1", 2e-3), help="Lower PDMS layer length, m.")
    parser.add_argument("--L2", type=float, default=cfg("L2", 2e-3), help="Upper magnetic layer length, m.")
    parser.add_argument("--substrate-radius", type=float, default=cfg("substrate_radius", 0.60e-3), help="Substrate radius, m.")
    parser.add_argument("--substrate-thickness", type=float, default=cfg("substrate_thickness", 0.50e-3), help="Substrate thickness, m.")

    # Mechanics and mesh
    parser.add_argument("--delta", type=float, default=cfg("delta", 0.32e-3))
    parser.add_argument("--mesh-delta", type=float, default=cfg("mesh_delta", 0.32e-3))
    parser.add_argument("--h-cilium", type=float, default=cfg("h_cilium", 30e-6))
    parser.add_argument("--h-substrate", type=float, default=cfg("h_substrate", 100e-6))
    parser.add_argument("--nu", type=float, default=cfg("nu", 0.49))
    parser.add_argument("--n-steps", type=int, default=cfg("n_steps", 12))
    parser.add_argument("--include-h20", action="store_true", default=cfg("include_h20", False))

    # Magnetics
    parser.add_argument("--Br", type=float, default=cfg("Br", 0.10), help="Effective remanence, T.")
    parser.add_argument("--sensor-x", type=float, default=cfg("sensor_x", 0.0))
    parser.add_argument("--sensor-x-over-r", type=float, default=cfg("sensor_x_over_r", None))
    parser.add_argument("--sensor-y", type=float, default=cfg("sensor_y", 0.0))
    parser.add_argument("--sensor-y-over-r", type=float, default=cfg("sensor_y_over_r", None))
    parser.add_argument("--sensor-z", type=float, default=cfg("sensor_z", -50e-6), help="Use --sensor-z=-50e-6 for negative values.")
    parser.add_argument("--sensor-z-over-r", type=float, default=cfg("sensor_z_over_r", None))
    parser.add_argument("--no-rotate-magnetization", dest="no_rotate_magnetization", action="store_true", default=cfg("no_rotate_magnetization", False))
    parser.add_argument("--rotate-magnetization", dest="no_rotate_magnetization", action="store_false")
    parser.add_argument(
        "--magnetization-model",
        choices=["fixed_global", "rotate_with_material", "prescribed_theta_mu", "follow_factor"],
        default=cfg("magnetization_model", None),
        help="Optional explicit magnetization orientation law. Legacy rotate_magnetization is used when omitted.",
    )
    parser.add_argument("--theta-mu-rad", type=float, default=cfg("theta_mu_rad", cfg("theta_mu", 0.0)))
    parser.add_argument("--theta-mu-deg", type=float, default=cfg("theta_mu_deg", None))
    parser.add_argument("--follow-factor-alpha", type=float, default=cfg("follow_factor_alpha", cfg("alpha", 1.0)))
    parser.add_argument(
        "--dipole-mode",
        choices=["tetrahedral", "tip_single_dipole", "n_point_dipoles"],
        default=cfg("dipole_mode", "tetrahedral"),
    )
    parser.add_argument("--n-point-dipoles", type=int, default=cfg("n_point_dipoles", cfg("n_dipoles", 8)))
    parser.add_argument("--compare-magnetization-models", action="store_true", default=cfg("compare_magnetization_models", False))
    parser.add_argument("--comparison-magnetization-models", nargs="+", default=cfg("comparison_magnetization_models", ()))
    parser.add_argument("--comparison-dipole-modes", nargs="+", default=cfg("comparison_dipole_modes", ()))
    parser.add_argument("--plot-magnetization-hall", action="store_true", default=cfg("plot_magnetization_hall", False))
    parser.add_argument("--target-sensitivity-uT-per-uN", type=float, default=cfg("target_sensitivity_uT_per_uN", 1.0))
    parser.add_argument("--air-radius-factor", type=float, default=cfg("air_radius_factor", 8.0))
    parser.add_argument("--air-below-factor", type=float, default=cfg("air_below_factor", 4.0))
    parser.add_argument("--air-above-factor", type=float, default=cfg("air_above_factor", 4.0))
    parser.add_argument("--h-air", type=float, default=cfg("h_air", 100e-6))
    parser.add_argument("--h-air-near", type=float, default=cfg("h_air_near", None))
    parser.add_argument("--h-air-far", type=float, default=cfg("h_air_far", None))
    parser.add_argument("--near-radius-factor", type=float, default=cfg("near_radius_factor", 3.0))
    parser.add_argument("--near-source-padding-factor", type=float, default=cfg("near_source_padding_factor", 2.0))
    parser.add_argument("--near-sensor-padding-factor", type=float, default=cfg("near_sensor_padding_factor", 1.0))
    parser.add_argument("--magnetic-boundary", choices=["dirichlet_zero", "natural"], default=cfg("magnetic_boundary", "natural"))
    parser.add_argument("--sensor-average", action="store_true", default=cfg("sensor_average", False))
    parser.add_argument("--sensor-average-radius", type=float, default=cfg("sensor_average_radius", 25e-6))
    parser.add_argument("--sensor-average-n", type=int, default=cfg("sensor_average_n", 5))
    parser.add_argument("--projection-mode", choices=["cell_center"], default=cfg("projection_mode", "cell_center"))
    parser.add_argument("--max-air-cells", type=int, default=cfg("max_air_cells", 700000))
    parser.add_argument("--allow-large-air-mesh", action="store_true", default=cfg("allow_large_air_mesh", False))
    parser.add_argument("--save-mechanics-frames", action="store_true", default=cfg("save_mechanics_frames", False))
    parser.add_argument("--mechanics-frames-every", type=int, default=cfg("mechanics_frames_every", 1))
    parser.add_argument("--mechanics-animation", action="store_true", default=cfg("mechanics_animation", False))

    args = parser.parse_args()
    args.config = pre_args.config
    if args.theta_mu_deg is not None:
        args.theta_mu_rad = float(args.theta_mu_deg) * np_pi() / 180.0
    if isinstance(args.comparison_magnetization_models, str):
        args.comparison_magnetization_models = tuple(
            part.strip() for part in args.comparison_magnetization_models.replace(",", " ").split() if part.strip()
        )
    elif args.comparison_magnetization_models is None:
        args.comparison_magnetization_models = ()
    else:
        args.comparison_magnetization_models = tuple(args.comparison_magnetization_models)
    if isinstance(args.comparison_dipole_modes, str):
        args.comparison_dipole_modes = tuple(
            part.strip() for part in args.comparison_dipole_modes.replace(",", " ").split() if part.strip()
        )
    elif args.comparison_dipole_modes is None:
        args.comparison_dipole_modes = ()
    else:
        args.comparison_dipole_modes = tuple(args.comparison_dipole_modes)
    if args.h_air_near is None:
        args.h_air_near = args.h_air
    if args.h_air_far is None:
        args.h_air_far = 3.0 * args.h_air

    if args.mode == "final":
        args.mode = "full"
    if args.mode == "magnetic-validation":
        args.mode = "magnetic-only"
    if args.mode == "magnetic-sensor-position-sweep":
        args.mode = "magnetics-fem-validation"

    magnetic_modes = {"magnetics", "magnetic-only", "magnetics-fem"}
    if args.mode in magnetic_modes:
        sensor_x_from_config = "sensor_x" in config_defaults
        sensor_x_over_r_from_config = "sensor_x_over_r" in config_defaults
        sensor_y_over_r_from_config = "sensor_y_over_r" in config_defaults
        sensor_z_from_config = "sensor_z" in config_defaults
        sensor_z_over_r_from_config = "sensor_z_over_r" in config_defaults
        if args.sensor_x_over_r is None and not cli_has("--sensor-x") and not sensor_x_from_config and not sensor_x_over_r_from_config:
            args.sensor_x_over_r = 1.0
        if args.sensor_x_over_r is not None and not cli_has("--sensor-x") and not sensor_x_from_config:
            args.sensor_x = args.sensor_x_over_r * (args.D / 2.0)
        if args.sensor_y_over_r is not None and not cli_has("--sensor-y"):
            args.sensor_y = args.sensor_y_over_r * (args.D / 2.0)
        elif sensor_y_over_r_from_config and not cli_has("--sensor-y"):
            args.sensor_y = args.sensor_y_over_r * (args.D / 2.0)
        if args.sensor_z_over_r is not None and not cli_has("--sensor-z"):
            args.sensor_z = args.sensor_z_over_r * (args.D / 2.0)
        if not cli_has("--sensor-z") and not sensor_z_from_config and args.sensor_z_over_r is None:
            args.sensor_z = 0.0

    if args.outdir is None:
        if args.mode == "validation":
            args.outdir = default_results_path("magnetic_cilium_3d_results_validation")
        elif args.mode == "magnetics":
            args.outdir = args.restart_dir
        else:
            args.outdir = default_results_path("magnetic_cilium_3d_results_final")
    if args.mode == "interpolate-magnetic-results" and args.interpolation_output_dir is None:
        exp = args.experiment_id or "default"
        args.interpolation_output_dir = default_results_path("interpolation", exp)

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
        magnetization_model=args.magnetization_model,
        theta_mu_rad=args.theta_mu_rad,
        follow_factor_alpha=args.follow_factor_alpha,
        dipole_mode=args.dipole_mode,
        n_point_dipoles=args.n_point_dipoles,
        compare_magnetization_models=bool(args.compare_magnetization_models),
        comparison_magnetization_models=tuple(args.comparison_magnetization_models or ()),
        comparison_dipole_modes=tuple(args.comparison_dipole_modes or ()),
        plot_magnetization_hall=bool(args.plot_magnetization_hall),
        outdir=outdir,
        save_mechanics_frames=bool(args.save_mechanics_frames),
        mechanics_frames_every=max(1, int(args.mechanics_frames_every)),
        mechanics_animation=bool(args.mechanics_animation),
    )


def np_pi() -> float:
    return 3.141592653589793
