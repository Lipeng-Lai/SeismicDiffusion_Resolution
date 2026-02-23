from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dat_io import save_dat_2d
from src.data.dataset_2d import PairedSeismic2DDataset, denormalize_tensor
from src.diffusion.ddim_sampler import ddim_sample
from src.diffusion.schedules import build_schedule
from src.engine.evaluator import batch_metrics
from src.models.unet2d import UNet2DNoisePredictor
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, resolve_device
from src.utils.logger import build_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Sample with trained 2D MD Diffusion.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, cli_overrides=True)
    set_seed(int(config.sampling.seed))
    device = resolve_device(config)

    ckpt_path = args.checkpoint or config.sampling.checkpoint_path
    if ckpt_path is None:
        raise ValueError("Please provide --checkpoint or sampling.checkpoint_path in config.yaml.")

    out_dir = Path(str(config.sampling.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(str(out_dir), name="md-sample")

    schedule = build_schedule(config, device)
    model = UNet2DNoisePredictor(config).to(device)
    state = load_checkpoint(str(ckpt_path), map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    dataset = PairedSeismic2DDataset(config, split="val")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config.data.num_workers),
        pin_memory=bool(config.data.pin_memory),
    )
    num_files = int(config.sampling.num_files)

    all_metrics = []
    generator = torch.Generator(device="cuda") if device.type == "cuda" else torch.Generator()
    generator.manual_seed(int(config.sampling.seed))

    for i, batch in enumerate(loader):
        if i >= num_files:
            break

        name = batch["name"][0]
        cond = batch["c"].to(device)
        target = batch["x0"].to(device)

        pred_norm = ddim_sample(
            model=model,
            cond=cond,
            schedule=schedule,
            config=config,
            device=device,
            generator=generator,
        )

        metrics = batch_metrics(pred_norm, target, config)
        metrics["name"] = name
        all_metrics.append(metrics)
        logger.info("sample=%s metrics=%s", name, metrics)

        norm_min = batch["norm_min"].to(device)[:, None, None, None]
        norm_max = batch["norm_max"].to(device)[:, None, None, None]
        pred_real = denormalize_tensor(
            pred_norm,
            norm_min=norm_min,
            norm_max=norm_max,
            out_range=tuple(config.data.normalize.range),
        )

        pred_np = pred_real.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        save_path = out_dir / f"{name}_pred.dat"
        save_dat_2d(str(save_path), pred_np, dtype=config.data.dtype)

    if all_metrics:
        summary = {}
        keys = [k for k in all_metrics[0].keys() if k != "name"]
        for k in keys:
            summary[k] = float(np.mean([m[k] for m in all_metrics]))
        logger.info("summary=%s", summary)
    else:
        summary = {}

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"items": all_metrics, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
