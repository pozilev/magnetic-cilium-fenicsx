import logging
import os

from logging_utils import setup_logging
from params import check_runtime, parse_args

log = logging.getLogger("magnetic_cilium_3d")


def main():
    args = parse_args()
    log_file = setup_logging(args.outdir)
    log.info("log file = %s", log_file)
    log.info("configuration = %s", vars(args))

    try:
        check_runtime(log)
        from pipeline import (
            print_summary,
            run_magnetic_only_validation,
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
        elif args.mode == "magnetic-only":
            run_magnetic_only_validation(args)
        else:
            raise RuntimeError(f"Unknown mode: {args.mode}")
    except Exception:
        log.exception("Fatal error")
        raise


if __name__ == "__main__":
    main()
