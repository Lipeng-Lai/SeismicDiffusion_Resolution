from __future__ import annotations

import torch


class DiffusionSchedule:
    def __init__(self, betas: torch.Tensor):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.timesteps = betas.shape[0]

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self


def make_beta_schedule(schedule: str, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if schedule != "linear":
        raise ValueError("Only diffusion.beta_schedule=linear is supported in this concise version.")
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def build_schedule(config, device: torch.device) -> DiffusionSchedule:
    betas = make_beta_schedule(
        schedule=str(config.diffusion.beta_schedule),
        timesteps=int(config.diffusion.timesteps),
        beta_start=float(config.diffusion.beta_start),
        beta_end=float(config.diffusion.beta_end),
    )
    return DiffusionSchedule(betas).to(device)


def sample_timesteps_and_gamma(batch_size: int, schedule: DiffusionSchedule, device: torch.device):
    t = torch.randint(low=0, high=schedule.timesteps, size=(batch_size,), device=device)
    alpha_t = schedule.alpha_bar[t]
    alpha_prev = torch.where(
        t > 0,
        schedule.alpha_bar[t - 1],
        torch.ones_like(alpha_t),
    )
    low = torch.minimum(alpha_prev, alpha_t)
    high = torch.maximum(alpha_prev, alpha_t)
    gamma = low + torch.rand(batch_size, device=device) * (high - low)
    return t, gamma


def q_sample(x0: torch.Tensor, gamma: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    gamma = gamma[:, None, None, None]
    return torch.sqrt(gamma) * x0 + torch.sqrt(1.0 - gamma) * noise

