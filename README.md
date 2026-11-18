# RG-Cycle: Reliability-Gated Cycle Prior for Long-Term Time-Series Forecasting

RG-Cycle 是一个可插到任意预测主干上的周期先验模块。它在 CycleNet 的 RCF（Residual Cycle Forecasting）基础上，
为每个样本、每个通道和每个预测步长学习“用多少周期先验”。本仓库包含 iTransformer 主干、RCF 基线、RG-Cycle、
训练脚本和第一天试点实验（iTransformer × {Traffic, Electricity, ETTh1} × H∈{96, 720}）的调度器。

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

`rgcycle/backbones.py` is the iTransformer-style channel-token backbone (2 layers, 8 heads, `d_ff = 2·d_model`,
dropout 0.1, RevIN without affine, no timestamp tokens). With `rng_compat_kernel` it consumes random numbers in the same
order as the PAC-Former code, so its initial weights are bit-identical to the PAC-Former ablation variant B.

Training (`rgcycle/trainer.py`): Adam (β = 0.9/0.999, no weight decay), batch 32, at most 50 epochs, early stopping
with patience 5 on full-validation MSE, best-validation checkpoint, 1-epoch linear warm-up from 0.1× then cosine to
0.01× over 50 planned epochs (per step), FP32 with TF32 disabled for matmul, deterministic algorithms, lookback 96,
every test window evaluated (no drop_last).

## Layout

```
rgcycle/backbones.py   iTransformer backbone + RevIN
rgcycle/cycle.py       RecurrentCycle (Q), reliability features, ReliabilityGate, CycleForecaster wrapper, build_model
rgcycle/data.py        chronological splits, train-only scaler, GPU windows with absolute row index
rgcycle/trainer.py     training / evaluation / gate statistics
train.py               single-job CLI
configs/datasets.json  dataset audit (SHA-256, splits)
configs/pilot_day1.json  day-1 pilot: groups, hyper-parameters, W, gate settings, reference numbers
pilot/run_pilot.py     --init (freeze jobs) / scheduler (2 workers, resumable) / per-group reports
pilot/launch.sh        background launcher (refuses if running, completed, or GPU busy)
pilot/smoke_test.py    GPU smoke test
tests/test_rgcycle.py  CPU unit tests
results/smoke_report_20261119.json  GPU smoke test on RTX 4090 (19 Nov 2026)
```

## Usage

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
# data: the seven benchmark CSVs (same files and SHA-256 as configs/datasets.json) in data/benchmark/

# single job
python train.py --dataset Traffic --horizon 96 --mode rg --seed 2026 --lr 0.001 --d-model 256 \
    --kernel-size 5 --cycle 168 --out outputs/traffic_h96_rg_s2026

# day-1 pilot
python pilot/run_pilot.py --run-dir RUN --init --data-dir data/benchmark --audit configs/datasets.json \
    [--orig-dir PATH_TO_EXISTING_B_RUNS]
bash pilot/launch.sh RUN python
```

Without `--orig-dir` the pilot also trains the backbone alone (`orig`), giving 54 jobs; with it, existing backbone
results are reused and 36 jobs are trained.

## References

- S. Lin et al., *CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns*, NeurIPS 2024. https://github.com/ACAT-SCUT/CycleNet
- Y. Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024.
