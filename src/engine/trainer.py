from __future__ import annotations

from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.dataset_2d import PairedSeismic2DDataset
from src.diffusion.schedules import build_schedule, q_sample, sample_timesteps_and_gamma
from src.models.unet2d import UNet2DNoisePredictor
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import resolve_device
from src.utils.logger import build_logger


def _amp_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported runtime.amp_dtype: {dtype_name}")


class MDTrainer2D:
    def __init__(self, config):
        self.cfg = config
        self.device = resolve_device(config)
        self.schedule = build_schedule(config, self.device)

        out_root = Path(config.project.output_root) / str(config.project.experiment)
        out_root.mkdir(parents=True, exist_ok=True)
        self.output_root = out_root
        self.logger = build_logger(str(out_root), name="md-train")
        OmegaConf.save(config, out_root / "resolved_config.yaml")

        self.train_set = PairedSeismic2DDataset(config, split="train")
        self.val_set = PairedSeismic2DDataset(config, split="val")
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=int(config.train.batch_size),
            shuffle=True,
            num_workers=int(config.data.num_workers),
            pin_memory=bool(config.data.pin_memory),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=int(config.train.batch_size),
            shuffle=False,
            num_workers=int(config.data.num_workers),
            pin_memory=bool(config.data.pin_memory),
            drop_last=False,
        )

        self.model = UNet2DNoisePredictor(config).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        amp_enabled = bool(config.train.amp) and self.device.type == "cuda"
        self.amp_enabled = amp_enabled
        self.amp_dtype = _amp_dtype(str(config.runtime.amp_dtype))
        self.scaler = GradScaler(enabled=amp_enabled)

        self.global_step = 0
        self._maybe_resume()

    def _build_optimizer(self):
        optim_name = str(self.cfg.train.optimizer.name).lower()
        if optim_name != "adam":
            raise ValueError("Only train.optimizer.name=adam is supported.")
        return torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.train.optimizer.lr),
            betas=tuple(self.cfg.train.optimizer.betas),
            weight_decay=float(self.cfg.train.optimizer.weight_decay),
        )

    def _build_scheduler(self):
        sched_name = str(self.cfg.train.scheduler.name).lower()
        if sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(self.cfg.train.max_steps),
                eta_min=float(self.cfg.train.scheduler.min_lr),
            )
        if sched_name == "none":
            return None
        raise ValueError("Supported train.scheduler.name: cosine, none.")

    def _loss(self, pred_noise: torch.Tensor, true_noise: torch.Tensor) -> torch.Tensor:
        loss_type = str(self.cfg.diffusion.loss.type).lower()
        weight = float(self.cfg.diffusion.loss.weight)
        if loss_type == "l1":
            return weight * F.l1_loss(pred_noise, true_noise)
        if loss_type == "l2":
            return weight * F.mse_loss(pred_noise, true_noise)
        raise ValueError("Supported diffusion.loss.type: l1, l2.")

    def _maybe_resume(self):
        resume_path = self.cfg.train.resume
        if resume_path is None:
            return
        state = load_checkpoint(str(resume_path), map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state["step"])
        self.logger.info("Resumed from %s at step %d", resume_path, self.global_step)

    def _run_one_batch(self, batch):
        x0 = batch["x0"].to(self.device, non_blocking=True)
        cond = batch["c"].to(self.device, non_blocking=True)
        noise = torch.randn_like(x0)
        _, gamma = sample_timesteps_and_gamma(x0.size(0), self.schedule, self.device)
        x_t = q_sample(x0, gamma, noise)
        pred = self.model(cond, x_t, gamma)
        return self._loss(pred, noise)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        losses = []
        max_batches = int(self.cfg.train.val_batches)
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            losses.append(self._run_one_batch(batch).item())
        self.model.train()
        return float(sum(losses) / max(len(losses), 1))

    def train(self):
        max_steps = int(self.cfg.train.max_steps)
        accum_steps = int(self.cfg.train.grad_accum_steps)
        log_every = int(self.cfg.train.log_every)
        val_every = int(self.cfg.train.val_every)
        save_every = int(self.cfg.train.save_every)

        train_iter = cycle(self.train_loader)
        self.model.train()

        while self.global_step < max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0

            for _ in range(accum_steps):
                batch = next(train_iter)
                with autocast(
                    enabled=self.amp_enabled,
                    dtype=self.amp_dtype if self.amp_enabled else torch.float32,
                ):
                    loss = self._run_one_batch(batch) / accum_steps
                self.scaler.scale(loss).backward()
                total_loss += loss.item()

            if float(self.cfg.train.grad_clip) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.train.grad_clip))

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1

            if self.global_step % log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info("step=%d train_loss=%.6f lr=%.6e", self.global_step, total_loss, lr)

            if self.global_step % val_every == 0:
                val_loss = self._validate()
                self.logger.info("step=%d val_loss=%.6f", self.global_step, val_loss)

            if self.global_step % save_every == 0 or self.global_step == max_steps:
                ckpt_path = self.output_root / f"model_step_{self.global_step}.pth"
                save_checkpoint(
                    path=str(ckpt_path),
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.global_step,
                    config=self.cfg,
                )
                self.logger.info("Saved checkpoint to %s", ckpt_path)

