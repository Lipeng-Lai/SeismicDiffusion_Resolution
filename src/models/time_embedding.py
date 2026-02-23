from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("time embedding dimension must be even")
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=x.device, dtype=x.dtype) * -(math.log(10000.0) / max(half - 1, 1))
    )
    args = x[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        gamma = torch.clamp(gamma, min=1.0e-8, max=1.0)
        log_gamma = torch.log(gamma)
        emb = sinusoidal_embedding(log_gamma, self.embed_dim)
        return self.mlp(emb)

