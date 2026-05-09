import logging
import os
import sys
from datetime import datetime


def setup_logging(outdir: str | None, prefix: str = "magnetic_validation") -> str:
    log_dir = outdir or "."
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    return log_path
