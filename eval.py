from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.dat_io import load_dat_2d
from src.engine.evaluator import compute_psnr, compute_ssim
from src.utils.config import load_config
from src.utils.logger import build_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predicted .dat files against HR labels.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def _center_like_dataset_start(size: int, patch: int, stride: int) -> int:
    max_start = size - patch
    if max_start <= 0:
        return 0
    starts = list(range(0, max_start + 1, max(1, stride)))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts[len(starts) // 2]


def _load_pred_with_mode(path: Path, config):
    raw = np.fromfile(path, dtype=config.data.dtype)
    full_shape = tuple(config.data.hr_shape)
    patch_shape = tuple(config.data.patch_size)
    full_n = int(np.prod(full_shape))
    patch_n = int(np.prod(patch_shape))

    if raw.size == full_n:
        return raw.reshape(full_shape), "full"
    if raw.size == patch_n:
        return raw.reshape(patch_shape), "patch"

    raise ValueError(
        f"Unexpected size for {path}: got {raw.size}, expected {full_n} (hr_shape={full_shape}) "
        f"or {patch_n} (patch_size={patch_shape})"
    )


def _target_for_pred(hr: np.ndarray, pred_mode: str, config) -> np.ndarray:
    if pred_mode == "full":
        return hr
    if pred_mode == "patch":
        ph, pw = tuple(config.data.patch_size)
        sy = _center_like_dataset_start(hr.shape[0], ph, int(config.data.patch_stride[0]))
        sx = _center_like_dataset_start(hr.shape[1], pw, int(config.data.patch_stride[1]))
        return hr[sy : sy + ph, sx : sx + pw]
    raise ValueError(f"Unknown pred_mode: {pred_mode}")


def main():
    args = parse_args()
    config = load_config(args.config, cli_overrides=True)

    pred_dir = Path(str(config.evaluation.pred_dir))
    save_dir = Path(str(config.evaluation.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(str(save_dir), name="md-eval")

    suffix = str(config.evaluation.pred_suffix)
    pred_files = sorted(pred_dir.glob(f"*{suffix}"))
    max_items = int(config.evaluation.max_items)
    pred_files = pred_files[:max_items]

    results = []
    for pred_path in pred_files:
        stem = pred_path.name[: -len(suffix)]
        hr_path = Path(str(config.data.hr_dir)) / f"{stem}{config.data.file_ext}"
        if not hr_path.exists():
            logger.warning("skip=%s reason=missing_hr", pred_path.name)
            continue

        pred, pred_mode = _load_pred_with_mode(pred_path, config)
        hr = load_dat_2d(str(hr_path), tuple(config.data.hr_shape), dtype=config.data.dtype)
        hr_target = _target_for_pred(hr, pred_mode, config)

        pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        hr_t = torch.from_numpy(hr_target).float().unsqueeze(0).unsqueeze(0)

        psnr = compute_psnr(pred_t, hr_t).mean().item()
        ssim = compute_ssim(
            pred_t,
            hr_t,
            c1=float(config.evaluation.ssim_c1),
            c2=float(config.evaluation.ssim_c2),
        ).mean().item()

        item = {"name": stem, "pred_mode": pred_mode, "psnr": psnr, "ssim": ssim}
        results.append(item)
        logger.info("file=%s mode=%s psnr=%.4f ssim=%.4f", stem, pred_mode, psnr, ssim)

    if results:
        summary = {
            "psnr": float(np.mean([x["psnr"] for x in results])),
            "ssim": float(np.mean([x["ssim"] for x in results])),
        }
    else:
        summary = {}
    logger.info("summary=%s", summary)

    with open(save_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"items": results, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
