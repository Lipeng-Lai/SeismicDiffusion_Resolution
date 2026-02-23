from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .dat_io import list_paired_dat_files, load_dat_2d
from .degrade import add_gaussian_noise, downsample_2d, random_upsample_2d


@dataclass
class NormStats:
    x_min: float
    x_max: float


def denormalize_tensor(x_norm: torch.Tensor, norm_min: torch.Tensor, norm_max: torch.Tensor, out_range):
    r0, r1 = float(out_range[0]), float(out_range[1])
    x01 = (x_norm - r0) / (r1 - r0)
    return x01 * (norm_max - norm_min) + norm_min


class PairedSeismic2DDataset(Dataset):
    def __init__(self, config, split: str = "train"):
        if int(config.data.ndim) != 2:
            raise ValueError("This implementation only supports data.ndim = 2.")

        self.cfg = config
        self.split = split
        self.is_train = split == "train"

        pairs = list_paired_dat_files(
            lr_dir=config.data.lr_dir,
            hr_dir=config.data.hr_dir,
            file_ext=config.data.file_ext,
        )
        if not pairs:
            raise RuntimeError("No paired .dat files were found in configured lr_dir/hr_dir.")

        rng = random.Random(int(config.data.split_seed))
        rng.shuffle(pairs)
        split_idx = int(len(pairs) * float(config.data.split.train))
        if split == "train":
            self.samples = pairs[:split_idx]
        else:
            self.samples = pairs[split_idx:]
        if not self.samples:
            self.samples = pairs

        self.lr_shape = tuple(config.data.lr_shape)
        self.hr_shape = tuple(config.data.hr_shape)
        self.patch_size = tuple(config.data.patch_size)
        self.patch_stride = tuple(config.data.patch_stride)
        self.norm_range = tuple(config.data.normalize.range)
        self.norm_eps = float(config.data.normalize.eps)

    def __len__(self) -> int:
        return len(self.samples)

    def _pick_start(self, size: int, patch: int, stride: int) -> int:
        max_start = size - patch
        if max_start <= 0:
            return 0
        starts = list(range(0, max_start + 1, max(1, stride)))
        if starts[-1] != max_start:
            starts.append(max_start)
        if self.is_train:
            return random.choice(starts)
        return starts[len(starts) // 2]

    def _maybe_augment(self, hr: np.ndarray, cond: np.ndarray):
        aug_cfg = self.cfg.data.augment
        if not self.is_train:
            return hr, cond

        if bool(aug_cfg.random_flip) and random.random() < float(aug_cfg.flip_prob):
            hr = np.flip(hr, axis=1).copy()
            cond = np.flip(cond, axis=1).copy()
        if bool(aug_cfg.random_flip) and random.random() < float(aug_cfg.flip_prob):
            hr = np.flip(hr, axis=0).copy()
            cond = np.flip(cond, axis=0).copy()

        if bool(aug_cfg.random_rotate_90) and random.random() < float(aug_cfg.rotate_prob):
            k = random.randint(0, 3)
            hr = np.rot90(hr, k).copy()
            cond = np.rot90(cond, k).copy()

        return hr, cond

    def _normalize_pair(self, hr: np.ndarray, cond: np.ndarray):
        r0, r1 = self.norm_range
        x_min = float(np.min(hr))
        x_max = float(np.max(hr))
        scale = max(x_max - x_min, self.norm_eps)
        hr_01 = (hr - x_min) / scale
        cond_01 = (cond - x_min) / scale
        hr_norm = hr_01 * (r1 - r0) + r0
        cond_norm = cond_01 * (r1 - r0) + r0
        return hr_norm, cond_norm, NormStats(x_min=x_min, x_max=x_max)

    def _build_cond(self, lr: np.ndarray, hr: np.ndarray) -> np.ndarray:
        deg = self.cfg.degradation
        if bool(deg.simulate_from_hr):
            x_lr = downsample_2d(
                hr,
                scale=int(deg.downsample.scale),
                mode=str(deg.downsample.mode),
                antialias=bool(deg.downsample.antialias),
            )
            x_lr = add_gaussian_noise(
                x_lr,
                std_min=float(deg.noise.std_min),
                std_max=float(deg.noise.std_max),
            )
        else:
            x_lr = lr

        if bool(deg.use_paired_lr):
            x_lr = lr

        cond = random_upsample_2d(
            x_lr,
            target_hw=self.hr_shape,
            modes=list(deg.upsample_to_hr.modes),
            random_choice=bool(deg.upsample_to_hr.random_choice and self.is_train),
        )
        return cond

    def __getitem__(self, idx: int):
        name, lr_path, hr_path = self.samples[idx]
        lr = load_dat_2d(lr_path, self.lr_shape, dtype=self.cfg.data.dtype)
        hr = load_dat_2d(hr_path, self.hr_shape, dtype=self.cfg.data.dtype)

        cond_full = self._build_cond(lr=lr, hr=hr)

        ph, pw = self.patch_size
        h, w = hr.shape
        sy = self._pick_start(h, ph, self.patch_stride[0])
        sx = self._pick_start(w, pw, self.patch_stride[1])
        hr_patch = hr[sy : sy + ph, sx : sx + pw]
        cond_patch = cond_full[sy : sy + ph, sx : sx + pw]

        hr_patch, cond_patch = self._maybe_augment(hr_patch, cond_patch)
        hr_norm, cond_norm, stats = self._normalize_pair(hr_patch, cond_patch)

        return {
            "name": name,
            "x0": torch.from_numpy(hr_norm).float().unsqueeze(0),
            "c": torch.from_numpy(cond_norm).float().unsqueeze(0),
            "norm_min": torch.tensor(stats.x_min, dtype=torch.float32),
            "norm_max": torch.tensor(stats.x_max, dtype=torch.float32),
        }

