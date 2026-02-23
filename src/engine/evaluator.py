from __future__ import annotations

import torch


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=(-2, -1))
    data_range = (target.amax(dim=(-2, -1)) - target.amin(dim=(-2, -1))).clamp(min=eps)
    return 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse.clamp(min=eps))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, c1: float, c2: float, eps: float = 1.0e-8):
    mu_x = pred.mean(dim=(-2, -1), keepdim=True)
    mu_y = target.mean(dim=(-2, -1), keepdim=True)
    sigma_x = ((pred - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_y = ((target - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(-2, -1), keepdim=True)

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = numerator / (denominator + eps)
    return ssim.squeeze(-1).squeeze(-1)


def batch_metrics(pred: torch.Tensor, target: torch.Tensor, config):
    metrics = {}
    metric_names = set(config.evaluation.metrics)
    if "psnr" in metric_names:
        metrics["psnr"] = compute_psnr(pred, target).mean().item()
    if "ssim" in metric_names:
        metrics["ssim"] = compute_ssim(
            pred,
            target,
            c1=float(config.evaluation.ssim_c1),
            c2=float(config.evaluation.ssim_c2),
        ).mean().item()
    return metrics

