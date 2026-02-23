# SeismicDiffusion_Resolution
Diffusion models for multidimensional seismic noise attenuation and superresolution(No Official and 2D)


# SeismicDiffusion_Resolution

A concise **2D reproduction** of MD Diffusion for seismic denoising + super-resolution, based on:
- Xiao et al., 2024: *Diffusion Models for Multidimensional Seismic Noise Attenuation and Super-Resolution*
- Data source: `JintaoLee-Roger/SeismicSuperResolution`

> Note: the original paper is 3D. This repo implements a clean **2D adaptation** (paired `nx2/sx` slices) while keeping the core conditional diffusion workflow.

## Highlights
- Conditional diffusion with inputs `(c, x_t, gamma_t)`
- Continuous noise-level sampling (`gamma ~ U(alpha_{t-1}, alpha_t)`)
- Attention-free U-Net (2D adaptation)
- DDIM accelerated sampling with logarithmic step schedule
- All runtime parameters are managed in `configs/config.yaml`

## Repository Layout
```text
.
├── configs/config.yaml
├── train.py
├── sample.py
├── eval.py
├── src/
│   ├── data/         # dat IO, paired dataset, degradation/upsampling
│   ├── models/       # 2D U-Net, time embedding, blocks
│   ├── diffusion/    # schedules, continuous q-sampling, DDIM sampler
│   ├── engine/       # trainer, metrics
│   └── utils/        # config, logger, checkpoint, seed
├── view_syn.ipynb
├── view_field.ipynb
└── codex.md          # reproduction notes and design rationale
```

## Data
Default config expects:
- LR: `/home/llp/data/Resolution/seismicSuperResolutionData/nx2`
- HR: `/home/llp/data/Resolution/seismicSuperResolutionData/sx`
- Field: `/home/llp/data/Resolution/seismicSuperResolutionData/field`

File format: `float32 .dat`, paired by filename stem (e.g., `0008.dat`).


## Quick Start
1. Train
```bash
python train.py --config configs/config.yaml
```

2. Sample (generate predictions)
```bash
python sample.py --config configs/config.yaml --checkpoint runs/md_diffusion_nx2_sx/model_step_400000.pth
```

3. Evaluate saved predictions
```bash
python eval.py --config configs/config.yaml
```

4. Visualize
```bash
jupyter lab view_syn.ipynb
jupyter lab view_field.ipynb
```

## Config-Driven Workflow
All key settings are controlled by `configs/config.yaml`, including:
- data paths / shapes / split / augmentation
- diffusion schedule and objective
- model width/depth and conditioning mode
- optimizer / training steps / checkpoint interval
- DDIM sampling steps and schedule
- evaluation metrics and prediction directory

You can override any parameter from CLI:
```bash
python train.py --config configs/config.yaml train.max_steps=1000 train.batch_size=4
```

## Current Scope
- Implemented: 2D paired training/inference/evaluation pipeline
- Not included: full 3D MD Diffusion training on volumetric patches



## Acknowledgements
- https://github.com/Dululu-xy/MD-Diffusion
- https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
- https://github.com/JintaoLee-Roger/SeismicSuperResolution

