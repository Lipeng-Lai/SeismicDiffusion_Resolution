from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_groups(channels: int, requested: int) -> int:
    g = min(requested, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class ResBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float,
        num_groups: int,
    ):
        super().__init__()
        g1 = _valid_groups(in_channels, num_groups)
        g2 = _valid_groups(out_channels, num_groups)

        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int, mode: str):
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kwargs = {"scale_factor": 2.0, "mode": self.mode}
        if self.mode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
        x = F.interpolate(x, **kwargs)
        return self.conv(x)
