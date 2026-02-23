# MD Diffusion 复现流程与代码框架（2D 适配版）

> 说明：论文原始方法面向 3D 数据体；本仓库按你的要求只复现 2D 版本（基于 `nx2/sx` 配对切片），并保持条件扩散训练与 DDIM 对数采样核心思想不变。

本文档基于以下信息整理：
- 论文：Xiao et al., 2024, *Diffusion Models for Multidimensional Seismic Noise Attenuation and Super-Resolution*。
- 项目说明：`github_README.md` 中给出的论文仓库、数据来源和实现线索。
- 本地数据：`/home/llp/data/Resolution/seismicSuperResolutionData/nx2`（LR）与 `/home/llp/data/Resolution/seismicSuperResolutionData/sx`（HR）。

---

## 0. 链接调研结论

1. `Dululu-xy/MD-Diffusion` 当前公开仓库内容以 README 为主，未提供完整可运行训练代码主干（需按论文与引用工程自行搭建）。
2. `Janspiry/Image-Super-Resolution-via-Iterative-Refinement` 提供了条件扩散训练/推理的成熟工程框架，可作为训练器与采样器组织方式参考。
3. `JintaoLee-Roger/SeismicSuperResolution` 提供与当前数据一致的 `sx/nx2` 数据来源背景，可直接复用其数据读取约定（同名配对、float32 dat）。

---

## 1. 复现目标与边界

目标是复现 MD Diffusion 的核心能力：
1. 条件扩散模型实现去噪 + 超分辨率联合恢复。
2. 3D U-Net 预测噪声（无 Attention）。
3. 训练使用连续噪声采样（`gamma ~ U(alpha_{t-1}, alpha_t)`）。
4. 推理使用 DDIM 对数采样子序列加速。

边界说明：
1. 当前 `nx2/sx` 是 `.dat` 切片数据（2D 样本对），而论文核心是 3D patch 训练。
2. 复现时需要先构造 3D 体数据，再做 `64x64x64` patch 采样。
3. 若后续获取真实 3D 体数据，可直接复用本框架的数据接口。

---

## 2. 建议代码目录

```text
SeismicDiffusion_Resolution/
  configs/
    config.yaml
  src/
    data/
      dat_io.py
      build_volumes.py
      dataset_3d.py
      degrade.py
    models/
      time_embedding.py
      blocks_3d.py
      unet3d.py
    diffusion/
      schedules.py
      q_sample.py
      ddim_sampler.py
    engine/
      trainer.py
      evaluator.py
    utils/
      seed.py
      logger.py
      checkpoint.py
  train.py
  sample.py
  eval.py
  codex.md
```

---

## 3. 配置驱动（已创建）

你要求的统一配置文件已建立：`configs/config.yaml`。

配置覆盖了以下关键项：
1. 数据路径与形状：`lr_dir/hr_dir`, `lr_shape/hr_shape`。
2. 3D 训练参数：`volume_depth`, `patch_size`, `patch_stride`。
3. 扩散过程：`timesteps=1000`, 连续 `gamma` 采样策略。
4. 模型结构：`channel_multipliers: [1,2,4,8,8]`, `use_attention: false`。
5. 训练超参：`batch_size=8`, `max_steps=400000`, `Adam lr=1e-4`。
6. 采样策略：`DDIM + logarithmic schedule + sample_steps=100`。

---

## 4. 数据流程（先做这个）

### 4.1 读入与配对
1. 从 `nx2` 和 `sx` 读取同名 `.dat` 文件并配对。
2. 用 `float32` 读取，按配置 reshape 为：
   - LR: `128x128`
   - HR: `256x256`
3. 做归一化到 `[-1, 1]`（与 config 对齐）。

### 4.2 2D 样本堆叠为 3D 体
1. 按文件索引顺序，将连续切片堆叠成体：`[D, H, W]`。
2. `D=64`（来自 `volume_depth`），滑窗步长可先设 `D/2`。
3. LR 与 HR 必须在同一索引窗口上对齐。

### 4.3 patch 提取
1. 从 3D 体中裁剪 `64x64x64` patch（可随机 + 滑窗混合）。
2. 训练时返回：
   - `x0`: HR patch（干净目标）
   - `c`: 条件输入（来自 LR 上采样到 HR 网格后的 patch）

### 4.4 退化策略
1. 当前数据已有配对 LR，可先设置 `use_paired_lr=true`。
2. 若做合成退化实验，再启用 `simulate_from_hr=true`：
   - 下采样 + 滤波 + 高斯噪声。
3. 上采样模式按论文细节随机选：
   - `trilinear` / `nearest`。

---

## 5. 模型流程（3D U-Net，无 Attention）

### 5.1 输入构成
每次前向输入三部分：
1. `c`：条件体（上采样后）。
2. `x_t`：当前扩散状态。
3. `gamma_t`：噪声水平嵌入（MLP/Sinusoidal embedding）。

推荐实现：
1. 将 `c` 与 `x_t` 在通道维 concat。
2. `gamma_t` 通过 time-embedding 注入 ResBlock。

### 5.2 U-Net 关键点
1. 全部使用 3D Conv/Downsample/Upsample。
2. `channel_multipliers=[1,2,4,8,8]`。
3. 去掉所有 Attention 模块，保证尺度泛化。
4. 输出为噪声预测 `eps_theta`，shape 与 `x_t` 相同。

---

## 6. 训练算法（连续噪声采样）

每个 iteration：
1. 采样 `(x0, c)`。
2. 采样离散步 `t ~ Uniform({1...T})`。
3. 从区间采样连续噪声水平：`gamma ~ U(alpha_{t-1}, alpha_t)`。
4. 采样噪声 `eps ~ N(0, I)`。
5. 构造 `x_t = sqrt(gamma)*x0 + sqrt(1-gamma)*eps`。
6. 预测 `eps_hat = model(c, x_t, gamma)`。
7. 损失 `L = ||eps_hat - eps||_1`（或 L2）。
8. Adam 更新参数。

伪代码：

```python
for step in range(max_steps):
    x0, c = next(loader)                       # [B,1,D,H,W]
    t = randint(1, T)
    gamma = uniform(alpha[t-1], alpha[t])     # continuous
    eps = randn_like(x0)
    x_t = (gamma**0.5) * x0 + ((1-gamma)**0.5) * eps

    eps_hat = model(c, x_t, gamma)
    loss = l1(eps_hat, eps)
    loss.backward()
    opt.step(); opt.zero_grad()
```

---

## 7. 推理算法（DDIM + 对数子序列）

### 7.1 子序列生成
1. 原始总步数 `T=1000`。
2. 设加速步数 `L=100`。
3. 用对数分布生成 `tau`（长度 L），再映射到 `[1, T]` 并去重排序。

### 7.2 反向采样
1. 初始化 `x_T ~ N(0, I)`。
2. 按 `tau` 从大到小迭代：
   - `eps_hat = model(c, x_t, gamma_t)`
   - 按 DDIM 更新到 `x_{t-1}`（最后一步 `z=0`）
3. 输出 `x_0`。

伪代码：

```python
x_t = randn_like_target()
for i, t in reversed(list(enumerate(tau))):
    gamma_t = alpha_bar[t]
    eps_hat = model(c, x_t, gamma_t)
    x0_hat = (x_t - (1-gamma_t).sqrt()*eps_hat) / gamma_t.sqrt()

    if i > 0:
        t_prev = tau[i-1]
        gamma_prev = alpha_bar[t_prev]
        sigma_t = eta * ddim_sigma(gamma_t, gamma_prev)
        z = randn_like(x_t)
        x_t = gamma_prev.sqrt()*x0_hat + (1-gamma_prev-sigma_t**2).sqrt()*eps_hat + sigma_t*z
    else:
        x_t = x0_hat
```

---

## 8. 分阶段落地顺序

### 阶段 A：最小可运行版本
1. 完成 `dat_io.py`：读取 `.dat`、reshape、归一化。
2. 完成 `dataset_3d.py`：2D->3D 堆叠 + patch 输出。
3. 完成 `unet3d.py`：无 Attention 版本。
4. 完成 `trainer.py`：连续 `gamma` 训练循环。
5. 完成 `sample.py`：DDIM 对数采样。

### 阶段 B：论文对齐增强
1. 加入随机上采样策略（`trilinear`/`nearest`）。
2. 对比线性/二次/对数采样子序列。
3. 增加 PSNR/SSIM/FID-like 评估与可视化输出。

### 阶段 C：泛化验证
1. 用合成数据训练。
2. 仅输入 field data 条件体进行推理测试。
3. 记录结构保持性与噪声抑制表现。

---

## 9. 你可以直接执行的开发清单

1. 先按 `configs/config.yaml` 搭建配置读取（OmegaConf 风格）。
2. 优先打通训练一个 epoch 的端到端链路，不先追求指标。
3. 再实现 DDIM 对数采样并做可视化对比。
4. 最后再补齐指标、日志、checkpoint、推理脚本参数化。

---

## 10. 与论文关键点的一一对应

1. 条件输入 `c` + 扩散状态 `x_t` + 噪声水平 `gamma_t`：已在模型接口定义。
2. 连续噪声采样：已在训练算法明确。
3. 3D U-Net 且去 Attention：已在结构约束明确。
4. DDIM 对数加速：已在采样流程明确。
5. 训练超参：已写入 `configs/config.yaml`（batch=8, steps=400000, lr=1e-4）。
