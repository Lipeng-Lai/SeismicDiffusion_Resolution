from __future__ import annotations

import numpy as np
import torch


def build_sampling_steps(timesteps: int, sample_steps: int, mode: str) -> np.ndarray:
    if sample_steps < 2:
        raise ValueError("sampling.sample_steps must be >= 2")
    if sample_steps > timesteps:
        sample_steps = timesteps

    if mode == "linear":
        steps = np.linspace(0, timesteps - 1, sample_steps)
    elif mode == "quadratic":
        x = np.linspace(0, 1, sample_steps)
        steps = (x**2) * (timesteps - 1)
    elif mode == "logarithmic":
        x = np.linspace(0, 1, sample_steps)
        steps = (np.exp(x * np.log(timesteps)) - 1.0)
    else:
        raise ValueError(f"Unsupported sampling.step_schedule: {mode}")

    idx = np.clip(np.round(steps).astype(np.int64), 0, timesteps - 1)
    idx = np.unique(idx)
    if idx[0] != 0:
        idx = np.insert(idx, 0, 0)
    if idx[-1] != timesteps - 1:
        idx = np.append(idx, timesteps - 1)
    return idx


@torch.no_grad()
def ddim_sample(model, cond: torch.Tensor, schedule, config, device: torch.device, generator=None):
    eta = float(config.sampling.eta)
    steps = build_sampling_steps(
        timesteps=int(config.diffusion.timesteps),
        sample_steps=int(config.sampling.sample_steps),
        mode=str(config.sampling.step_schedule),
    )
    alpha_bar = schedule.alpha_bar
    batch = cond.shape[0]
    x_t = torch.randn(cond.shape, device=device, generator=generator)

    for i in range(len(steps) - 1, -1, -1):
        t_idx = int(steps[i])
        a_t = alpha_bar[t_idx].clamp(min=1.0e-8, max=1.0)
        gamma_t = torch.full((batch,), a_t.item(), device=device, dtype=cond.dtype)

        eps_hat = model(cond, x_t, gamma_t)
        x0_hat = (x_t - torch.sqrt(1.0 - a_t) * eps_hat) / torch.sqrt(a_t)
        if bool(config.sampling.clip_denoised):
            r0, r1 = config.data.normalize.range
            x0_hat = torch.clamp(x0_hat, min=float(r0), max=float(r1))

        if i == 0:
            x_t = x0_hat
            break

        t_prev = int(steps[i - 1])
        a_prev = alpha_bar[t_prev].clamp(min=1.0e-8, max=1.0)
        sigma = eta * torch.sqrt(((1.0 - a_prev) / (1.0 - a_t)) * (1.0 - a_t / a_prev))
        sigma = torch.clamp(sigma, min=0.0)

        noise = torch.randn_like(x_t, generator=generator) if sigma.item() > 0 else 0.0
        dir_xt = torch.sqrt(torch.clamp(1.0 - a_prev - sigma**2, min=0.0)) * eps_hat
        x_t = torch.sqrt(a_prev) * x0_hat + dir_xt + sigma * noise

    return x_t

