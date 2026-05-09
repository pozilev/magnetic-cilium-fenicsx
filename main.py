import logging
import os

from params import check_runtime, parse_args
from pipeline import (
    print_summary,
    run_magnetics_from_restart,
    run_single_pipeline_case,
    run_validation_study,
    write_summary,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("magnetic_cilium_3d")


def main():
    check_runtime(log)
    args = parse_args()

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
    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
