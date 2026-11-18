# RG-Cycle: Reliability-Gated Cycle Prior for Long-Term Time-Series Forecasting

RG-Cycle 是一个可插到任意预测主干上的周期先验模块。它在 CycleNet 的 RCF（Residual Cycle Forecasting）基础上，
为每个样本、每个通道和每个预测步长学习“用多少周期先验”。本仓库包含 iTransformer 主干（与 CycleNet 官方代码逐位一致）、RCF 基线、RG-Cycle、
训练脚本和第一天试点实验（iTransformer × {Traffic, Electricity, ETTh1} × H∈{96, 720}，按 CycleNet 论文的设置）的调度器。

RG-Cycle is a plug-in cycle prior for any forecasting backbone. It extends CycleNet's Residual Cycle Forecasting (RCF)
with a reliability gate that decides, per sample, channel and horizon step, how much of the learned cycle to remove from
the input and add back to the output.

## Method

All quantities live in the instance-normalised (RevIN) space, as in CycleNet / CycleiTransformer.
`x_n ∈ R^{L×C}` is the normalised lookback window and `Q ∈ R^{W×C}` is a learnable cycle table initialised to zero.
With cycle phase `k = (t0 + L) mod W` (official CycleNet convention, `t0` = absolute row index of the first lookback step):

```
c_in[i]  = Q[(k + i) mod W]          i = 0..L-1
c_out[j] = Q[(k + L + j) mod W]      j = 0..H-1

orig : y_n = f(x_n)
RCF  : y_n = f(x_n − c_in) + c_out                                (CycleNet, Lin et al., NeurIPS 2024)
RG   : y_n = f(x_n − g_in ⊙ c_in) + g_out ⊙ c_out
```

Reliability features per sample/channel, computed on the lookback with `c_in` detached (stop-gradient, so Q is trained
only by the forecasting loss):

```
R2  = clamp(1 − ||(x − mean x) − c_in||² / ||x − mean x||², −1, 1)
rho = corr(x, c_in)
```

Gate head (`2C + 2H + 4` parameters, all zero-initialised; `b0 = −2` is a fixed buffer, so g = 0.119 at init):

```
g_in [c]    = σ(b0 + β_in[c] + a_in·[R2, rho])
g_out[h, c] = σ(b0 + β_out[c] + η[h] + (a_out,0 + κ[h])·R2 + a_out,1·rho)
```

Because `Q = 0` at initialisation, RCF and RG reproduce the backbone output exactly at step 0 (verified bitwise).
Gate parameters use 10× the scheduled learning rate (their own Adam group); everything else uses the backbone recipe.

| Dataset | W | RCF extra params | RG extra params (H=96 / 720) |
|---|---:|---:|---:|
| ETTh1 (7 ch) | 24 | 168 | 378 / 1,626 |
| Electricity (321 ch) | 168 | 53,928 | 54,766 / 56,014 |
| Traffic (862 ch) | 168 | 144,816 | 146,736 / 147,984 |

## Backbone and recipe

The day-1 pilot follows the CycleNet paper (Lin et al., NeurIPS 2024, Table 4) and its official code
(github.com/ACAT-SCUT/CycleNet @ d807e51):

- Backbone `CycleNetITransformer` (`rgcycle/backbones.py`) is CycleNet's `models/CycleiTransformer.py` without the
  cycle: inverted embedding, `e_layers` encoder layers (full attention, Conv1d-k1 FFN, GELU, post-LN), final LayerNorm,
  linear projector, Non-stationary-Transformer normalisation, no timestamp tokens. With the same seed its initial
  weights and its outputs (orig, and +RCF with any Q) are bitwise identical to the official module
  (`pilot/smoke_test_cyclenet.py`).
- Hyper-parameters per dataset/H from `scripts/iTransformer/{traffic,electricity}.sh`; ETTh1 (not part of CycleNet's
  iTransformer experiments) from the official iTransformer ETTh1 script and iTransformer `run.py` defaults.
  Every value and its source file/line is in `configs/pilot_day1_cyclenet.json`.
- Training (`rgcycle/trainer.py`, recipe `cyclenet`): Adam, MSE, shuffled training batches with drop_last, at most
  `train_epochs` epochs, early stopping with `patience` on the validation loss, best-validation checkpoint.
  Learning rate as the official code actually runs it (`lradj=type3` plus an un-stepped `OneCycleLR` whose constructor
  sets lr/25): epoch 1 at lr/25, epochs 2–4 at lr, then ×0.8 per epoch. Validation loss = mean of per-batch MSE over a
  shuffled, drop_last validation loader; test = every window (drop_last=False, as in CycleNet).
- orig, +RCF and +RG use the same recipe. RG adds only the gate (b0 = −2, gate lr ×10).
- FP32, TF32 at PyTorch defaults (off for matmul, on for cuDNN convolutions, as in the official code), plus
  deterministic algorithms (the official code is not deterministic; this changes no hyper-parameter).

The older PAC-Former-recipe configuration (`configs/pilot_day1.json`, backbone `pacformer_b`, recipe `pacformer`)
is kept for reference; it reproduces the PAC-Former variant-B runs bit-for-bit.

## Layout

```
rgcycle/backbones.py   iTransformer backbone + RevIN
rgcycle/cycle.py       RecurrentCycle (Q), reliability features, ReliabilityGate, CycleForecaster wrapper, build_model
rgcycle/data.py        chronological splits, train-only scaler, GPU windows with absolute row index
rgcycle/trainer.py     training / evaluation / gate statistics
train.py               single-job CLI
configs/datasets.json  dataset audit (SHA-256, splits)
configs/pilot_day1_cyclenet.json  day-1 pilot under the CycleNet protocol (values + source file/line)
configs/pilot_day1.json  earlier PAC-Former-recipe variant (reference only)
pilot/run_pilot.py     --init (freeze jobs) / scheduler (2 workers, resumable) / per-group reports
pilot/launch.sh        background launcher (refuses if running, completed, or GPU busy)
pilot/smoke_test_cyclenet.py  GPU smoke test (official-code equivalence, determinism, timing, 2-worker memory)
pilot/smoke_test.py    GPU smoke test for the PAC-Former-recipe variant
tests/test_rgcycle.py  CPU unit tests
results/smoke_report_20261119.json  GPU smoke test on RTX 4090 (19 Nov 2026)
```

## Usage

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
# data: the seven benchmark CSVs (same files and SHA-256 as configs/datasets.json) in data/benchmark/

# single job (any group of a pilot config)
python train.py --config configs/pilot_day1_cyclenet.json --dataset Traffic --horizon 96 --mode rg --seed 2026 \
    --out outputs/traffic_h96_rg_s2026

# day-1 pilot (CycleNet protocol, 54 jobs: orig / +RCF / +RG x 6 groups x 3 seeds)
python pilot/run_pilot.py --run-dir RUN --init --data-dir data/benchmark --audit configs/datasets.json
bash pilot/launch.sh RUN python
```

## Third-party code

`rgcycle/backbones.py` (CycleNet iTransformer section) and `rgcycle/cycle.py` (`RecurrentCycle`) are adapted from
CycleNet (Apache-2.0); see `THIRD_PARTY_NOTICES.md`.

## References

- S. Lin et al., *CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns*, NeurIPS 2024. https://github.com/ACAT-SCUT/CycleNet
- Y. Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024.
