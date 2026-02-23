from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks_2d import Downsample2D, ResBlock2D, Upsample2D, _valid_groups
from .time_embedding import TimeEmbedding


class UNet2DNoisePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        mcfg = config.model
        if str(mcfg.cond_injection) != "concat":
            raise ValueError("Only model.cond_injection=concat is supported.")

        in_ch = int(mcfg.in_channels_xt) + int(mcfg.in_channels_cond)
        out_ch = int(mcfg.out_channels)
        base = int(mcfg.base_channels)
        mults = list(mcfg.channel_multipliers)
        num_res_blocks = int(mcfg.num_res_blocks)
        if num_res_blocks != 2:
            raise ValueError("This concise implementation expects model.num_res_blocks = 2.")

        time_dim = int(mcfg.time_embedding_dim)
        dropout = float(mcfg.dropout)
        num_groups = int(mcfg.num_groups)
        up_mode = str(mcfg.upsample_mode)

        self.time_embed = TimeEmbedding(time_dim)
        self.input_conv = nn.Conv2d(in_ch, base, kernel_size=3, padding=1)

        self.down_stages = nn.ModuleList()
        current_ch = base
        for i, m in enumerate(mults):
            stage_ch = base * int(m)
            stage = nn.ModuleDict(
                {
                    "res1": ResBlock2D(current_ch, stage_ch, time_dim, dropout, num_groups),
                    "res2": ResBlock2D(stage_ch, stage_ch, time_dim, dropout, num_groups),
                    "down": Downsample2D(stage_ch) if i < len(mults) - 1 else nn.Identity(),
                }
            )
            self.down_stages.append(stage)
            current_ch = stage_ch

        self.mid_1 = ResBlock2D(current_ch, current_ch, time_dim, dropout, num_groups)
        self.mid_2 = ResBlock2D(current_ch, current_ch, time_dim, dropout, num_groups)

        self.up_stages = nn.ModuleList()
        rev_mults = list(reversed(mults))
        for i, m in enumerate(rev_mults):
            skip_ch = base * int(m)
            stage = nn.ModuleDict(
                {
                    "res1": ResBlock2D(current_ch + skip_ch, skip_ch, time_dim, dropout, num_groups),
                    "res2": ResBlock2D(skip_ch, skip_ch, time_dim, dropout, num_groups),
                    "up": Upsample2D(skip_ch, mode=up_mode) if i < len(rev_mults) - 1 else nn.Identity(),
                }
            )
            self.up_stages.append(stage)
            current_ch = skip_ch

        self.out_norm = nn.GroupNorm(
            num_groups=_valid_groups(current_ch, num_groups),
            num_channels=current_ch,
        )
        self.out_conv = nn.Conv2d(current_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, cond: torch.Tensor, x_t: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(gamma)
        x = torch.cat([cond, x_t], dim=1)
        x = self.input_conv(x)

        skips = []
        for stage in self.down_stages:
            x = stage["res1"](x, t_emb)
            x = stage["res2"](x, t_emb)
            skips.append(x)
            x = stage["down"](x)

        x = self.mid_1(x, t_emb)
        x = self.mid_2(x, t_emb)

        for stage in self.up_stages:
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = stage["res1"](x, t_emb)
            x = stage["res2"](x, t_emb)
            x = stage["up"](x)

        return self.out_conv(F.silu(self.out_norm(x)))
