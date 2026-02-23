from __future__ import annotations

import argparse

from src.engine.trainer import MDTrainer2D
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2D MD Diffusion.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, cli_overrides=True)
    set_seed(int(config.project.seed))
    trainer = MDTrainer2D(config)
    trainer.train()


if __name__ == "__main__":
    main()

