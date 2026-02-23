from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf


def load_config(path: str, cli_overrides: bool = True):
    cfg_path = Path(path)
    config = OmegaConf.load(cfg_path)
    if cli_overrides:
        overrides = OmegaConf.from_cli()
        config = OmegaConf.merge(config, overrides)

    config.runtime.gpu_num = torch.cuda.device_count()
    return config


def resolve_device(config) -> torch.device:
    req = str(config.runtime.device).lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)

