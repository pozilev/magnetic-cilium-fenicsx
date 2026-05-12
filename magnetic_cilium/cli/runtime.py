import logging
import os

from magnetic_cilium.io.logging_utils import setup_logging
from magnetic_cilium.config.params import check_runtime, parse_args

log = logging.getLogger("magnetic_cilium_3d")


def main():
    args = parse_args()
    log_outdir = args.interpolation_output_dir if args.mode == "interpolate-magnetic-results" and args.interpolation_output_dir else args.outdir
    log_file = setup_logging(log_outdir)
    log.info("log file = %s", log_file)
    log.info("configuration = %s", vars(args))

    try:
        if args.mode == "interpolate-magnetic-results":
            from magnetic_cilium.postprocess.interpolation import run_magnetic_interpolation
            report = run_magnetic_interpolation(args)
            print("\n=== MAGNETIC INTERPOLATION SUMMARY ===")
            print(f"input_csv = {report['input_csv']}")
            print(f"filtered_rows = {report.get('filtered_rows', 0)}")
            print(f"output_dir = {args.interpolation_output_dir}")
            print(f"report = {os.path.join(args.interpolation_output_dir, 'interpolation_report.json')}")
            print(f"next_points = {os.path.join(args.interpolation_output_dir, 'next_points.csv')}")
            return

        check_runtime(log)
        from magnetic_cilium.pipeline.execution import (
            print_summary,
            run_magnetic_only_validation,
            run_magnetics_fem_validation_from_config,
            run_magnetics_fem_from_restart,
            run_magnetics_from_restart,
            run_single_pipeline_case,
            run_validation_study,
            write_summary,
        )

        log.info("mode = %s", args.mode)
        log.info("base outdir = %s", args.outdir)

        if args.mode == "validation":
            results = run_validation_study(args)
            summary_path = os.path.join(args.outdir, "summary.csv")
            write_summary(results, summary_path)
            print_summary(results, summary_path)
        elif args.mode in ("full", "mechanics"):
            results = run_single_pipeline_case(args)
            summary_name = "summary.csv" if args.mode == "full" else "mechanics_summary.csv"
            summary_path = os.path.join(args.outdir, summary_name)
            write_summary(results, summary_path)
            print_summary(results, summary_path)
        elif args.mode == "magnetics":
            run_magnetics_from_restart(args)
        elif args.mode == "magnetics-fem":
            run_magnetics_fem_from_restart(args)
        elif args.mode == "magnetics-fem-validation":
            run_magnetics_fem_validation_from_config(args.config)
        elif args.mode == "magnetic-only":
            run_magnetic_only_validation(args)
        else:
            raise RuntimeError(f"Unknown mode: {args.mode}")
    except Exception:
        log.exception("Fatal error")
        raise


if __name__ == "__main__":
    main()
