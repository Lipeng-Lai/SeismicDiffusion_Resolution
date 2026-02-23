from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "step": step,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    torch.save(state, out_path)


def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)

