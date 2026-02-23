from __future__ import annotations

from pathlib import Path

import numpy as np


def list_paired_dat_files(lr_dir: str, hr_dir: str, file_ext: str = ".dat"):
    lr_paths = {p.stem: p for p in Path(lr_dir).glob(f"*{file_ext}") if p.is_file()}
    hr_paths = {p.stem: p for p in Path(hr_dir).glob(f"*{file_ext}") if p.is_file()}
    common = sorted(set(lr_paths.keys()) & set(hr_paths.keys()))
    return [(name, str(lr_paths[name]), str(hr_paths[name])) for name in common]


def load_dat_2d(path: str, shape: tuple[int, int], dtype: str = "float32") -> np.ndarray:
    arr = np.fromfile(path, dtype=dtype)
    expected = shape[0] * shape[1]
    if arr.size != expected:
        raise ValueError(
            f"Unexpected size for {path}: got {arr.size}, expected {expected} for shape {shape}"
        )
    return arr.reshape(shape)


def save_dat_2d(path: str, array: np.ndarray, dtype: str = "float32") -> None:
    arr = np.asarray(array, dtype=dtype)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)

