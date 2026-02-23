from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


def _as_4d(x_2d: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_2d).float().unsqueeze(0).unsqueeze(0)


def _to_np(x_4d: torch.Tensor) -> np.ndarray:
    return x_4d.squeeze(0).squeeze(0).cpu().numpy()


def _interp_mode_2d(mode: str) -> str:
    mode = mode.lower()
    if mode == "trilinear":
        return "bilinear"
    return mode


def downsample_2d(x_hr: np.ndarray, scale: int, mode: str, antialias: bool) -> np.ndarray:
    t = _as_4d(x_hr)
    out_h = x_hr.shape[0] // scale
    out_w = x_hr.shape[1] // scale
    interp_mode = _interp_mode_2d(mode)
    kwargs = {"mode": interp_mode}
    if interp_mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
        kwargs["antialias"] = antialias
    x_lr = F.interpolate(t, size=(out_h, out_w), **kwargs)
    return _to_np(x_lr)


def add_gaussian_noise(x: np.ndarray, std_min: float, std_max: float) -> np.ndarray:
    std = random.uniform(std_min, std_max)
    if std <= 0.0:
        return x
    noise = np.random.normal(loc=0.0, scale=std, size=x.shape).astype(x.dtype)
    return x + noise


def random_upsample_2d(
    x_lr: np.ndarray,
    target_hw: tuple[int, int],
    modes: list[str],
    random_choice: bool,
) -> np.ndarray:
    if not modes:
        raise ValueError("degradation.upsample_to_hr.modes cannot be empty")
    mode = random.choice(modes) if random_choice else modes[0]
    interp_mode = _interp_mode_2d(mode)

    t = _as_4d(x_lr)
    kwargs = {"mode": interp_mode}
    if interp_mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    out = F.interpolate(t, size=target_hw, **kwargs)
    return _to_np(out)

