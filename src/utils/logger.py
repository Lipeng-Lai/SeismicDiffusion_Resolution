from __future__ import annotations

import logging
from pathlib import Path


def build_logger(log_dir: str, name: str = "md-diffusion") -> logging.Logger:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(path / "run.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger

