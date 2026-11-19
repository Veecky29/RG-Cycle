# Day-1 pilot results (CycleNet protocol)

**Status: complete, 54/54 jobs.** The scheduler finished at **2026-11-19 10:24:11 (Malaysia Time, UTC+8)** after 31,407 s (8 h 43 m). All six groups are reported below. The earlier 35/54 snapshot has been replaced by this final set.

本目录是第一天试点在 CycleNet 论文设定下的完整复现结果：冻结的协议与超参、54 个任务的逐 seed 指标，以及全部 6 组的汇总。主表是 **3 个 seed（2026 / 2027 / 2028）的均值±标准差**，不是 best-of-seed。

This directory is the complete day-1 pilot under the CycleNet (NeurIPS 2024) iTransformer / CycleiTransformer protocol: the frozen protocol and hyper-parameters, per-seed metrics for all 54 jobs, and summaries for all six groups. The main tables are **mean ± standard deviation over three seeds** (2026 / 2027 / 2028), not a best-of-seed selection.

## Files

| File | What it is |
|---|---|
| [`protocol.json`](protocol.json) | Frozen `--init` protocol (config hash, code hashes, 54 jobs, references). Host paths (`data_dir`, `audit`, `config_file`) are from the machine that ran the pilot. Created 2026-11-19 01:31:45 (Malaysia Time, UTC+8). |
| [`jobs.csv`](jobs.csv) | All 54 jobs and their hyper-parameters |
| [`results.csv`](results.csv) | Per-seed test MSE / MAE and training diagnostics for all 54 jobs |
| [`group_done/`](group_done/) | Per-group JSON written when all 9 jobs of a group finished |
| [`group_done.log`](group_done.log) | One-line group summaries, in completion order |
| [`status.json`](status.json) | Final scheduler status (`completed`, 54/54) |
| [`scheduler.out`](scheduler.out) | Scheduler start / done log |
| [`../../configs/pilot_day1_cyclenet.json`](../../configs/pilot_day1_cyclenet.json) | Source config (SHA-256 `482cda3f5f810b5118c77cba7e032868ca954744dc0a29ea533d35eb0b4c11ed`, identical to `protocol.json` `config_sha256`). Code SHA-256 values in the protocol match the files in this repository. |

## Protocol and hyper-parameters

Common settings (every job): lookback `seq_len = 96`, `n_heads = 8`, GELU, RevIN / Non-stationary-Transformer normalisation, Adam, MSE, CycleNet learning-rate schedule (epoch 1 at `lr/25`, epochs 2–4 at `lr`, then ×0.8 per epoch), shuffled training and validation with `drop_last=True`, test over every window (`drop_last=False`), early stopping on validation MSE. Seeds: 2026, 2027, 2028. Two workers. +RG only: gate bias `b0 = −2` (gate ≈ 0.12 at init) and 10× learning rate for the gate parameters. The source of every value is recorded in `configs/pilot_day1_cyclenet.json`.

| Dataset | H | e_layers | d_model | d_ff | dropout | batch | lr | epochs | patience | cycle W | Hyper-parameter source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | 96 | 2 | 256 | 256 | 0.1 | 32 | 1e-4 | 10 | 3 | 24 | Official iTransformer ETTh1 script and `run.py` defaults; cycle length from CycleNet `scripts/CycleNet/Linear-Input-96/etth1.sh`. CycleNet Table 4 has no iTransformer / +RCF run on ETTh1. |
| ETTh1 | 720 | 2 | 512 | 512 | 0.1 | 32 | 1e-4 | 10 | 3 | 24 | Same, pred_len 720 block of the official iTransformer ETTh1 script. |
| Electricity | 96 | 3 | 512 | 512 | 0.1 | 16 | 5e-4 | 10 | 3 | 168 | CycleNet `scripts/iTransformer/electricity.sh` |
| Electricity | 720 | 3 | 512 | 512 | 0.1 | 16 | 5e-4 | 10 | 3 | 168 | Same script. |
| Traffic | 96 | 4 | 512 | 512 | 0.1 | 16 | 1e-3 | 30 | 5 | 168 | CycleNet `scripts/iTransformer/traffic.sh` |
| Traffic | 720 | 4 | 512 | 512 | 0.1 | 16 | 1e-3 | 30 | 5 | 168 | Same script. |

## Results

Comparison bases, stated once and used throughout:

- **orig** is iTransformer trained in this pilot, under the CycleNet protocol.
- **+RCF** is CycleNet Residual Cycle Forecasting on the same backbone and recipe (CycleiTransformer).
- **+RG** is RG-Cycle on the same backbone and recipe.
- **vs orig** is the relative change of the 3-seed mean against the orig mean from this same pilot (`100 × (method − orig) / orig`). Negative is an improvement (lower error).
- **iTransformer paper** is Liu et al., ICLR 2024, official reported test MSE / MAE.
- **CycleNet Table 4** is Lin et al., NeurIPS 2024, Table 4, the iTransformer + RCF numbers. That table covers Electricity and Traffic only; ETTh1 has no official iTransformer + RCF reference.

`3/3` (or `0/3`) means how many seeds of that method have a lower test MSE than the same seed of orig.

### Test MSE / MAE (mean ± SD over 3 seeds)

| Dataset | H | orig MSE | orig MAE | +RCF MSE | +RCF MAE | +RG MSE | +RG MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.3852±0.0019 | 0.4026±0.0004 | 0.3841±0.0020 | 0.4019±0.0004 | 0.3827±0.0021 | 0.4009±0.0004 |
| ETTh1 | 720 | 0.5005±0.0084 | 0.4868±0.0053 | 0.4994±0.0085 | 0.4860±0.0053 | 0.4996±0.0084 | 0.4861±0.0053 |
| Electricity | 96 | 0.1480±0.0008 | 0.2407±0.0010 | 0.1352±0.0007 | 0.2312±0.0005 | 0.1337±0.0003 | 0.2296±0.0002 |
| Electricity | 720 | 0.2302±0.0068 | 0.3138±0.0063 | 0.1991±0.0100 | 0.2921±0.0086 | 0.2017±0.0142 | 0.2942±0.0107 |
| Traffic | 96 | 0.3913±0.0007 | 0.2646±0.0002 | 0.4223±0.0003 | 0.2597±0.0007 | 0.4328±0.0021 | 0.2623±0.0003 |
| Traffic | 720 | 0.4567±0.0004 | 0.2981±0.0009 | 0.4887±0.0053 | 0.2926±0.0013 | 0.5207±0.0028 | 0.2969±0.0006 |

### Relative change and paper references

| Dataset | H | orig vs iTransformer paper (MSE) | +RCF vs orig (MSE / MAE) | +RG vs orig (MSE / MAE) | +RCF vs CycleNet Table 4 (MSE) | +RG vs CycleNet Table 4 (MSE) |
|---|---|---|---|---|---|---|
| ETTh1 | 96 | 0.3852 vs 0.386 / 0.405, **−0.22%** | **−0.28% / −0.20%**, 3/3 better | **−0.65% / −0.43%**, 3/3 better | no official number | no official number |
| ETTh1 | 720 | 0.5005 vs 0.503 / 0.491, **−0.49%** | **−0.24% / −0.17%**, 3/3 better | **−0.18% / −0.14%**, 3/3 better | no official number | no official number |
| Electricity | 96 | 0.1480 vs 0.148 / 0.240, **+0.03%** | **−8.71% / −3.95%**, 3/3 better | **−9.71% / −4.59%**, 3/3 better | 0.1352 vs 0.136 / 0.231, **−0.62%** | 0.1337 vs 0.136, **−1.71%** |
| Electricity | 720 | 0.2302 vs 0.225 / 0.317, **+2.31%** | **−13.52% / −6.92%**, 3/3 better | **−12.39% / −6.26%**, 3/3 better | 0.1991 vs 0.194 / 0.287, **+2.62%** | 0.2017 vs 0.194, **+3.96%** |
| Traffic | 96 | 0.3913 vs 0.395 / 0.268, **−0.94%** | **+7.93% / −1.84%**, 0/3 better | **+10.60% / −0.88%**, 0/3 better | 0.4223 vs 0.415 / 0.263, **+1.76%** | 0.4328 vs 0.415, **+4.28%** |
| Traffic | 720 | 0.4567 vs 0.467 / 0.302, **−2.21%** | **+7.02% / −1.84%**, 0/3 better | **+14.03% / −0.40%**, 0/3 better | 0.4887 vs 0.491 / 0.294, **−0.46%** | 0.5207 vs 0.491, **+6.06%** |

orig is within 1% of the iTransformer paper on ETTh1 and Electricity H96, 2.21% better on Traffic H720, and 2.31% higher on Electricity H720 (seed 2028, 0.2379, pulls that mean up; the other two orig seeds are 0.2254 and 0.2273).

+RCF reproduces the CycleNet Table 4 pattern. On Electricity it helps iTransformer (H96 −8.71%, H720 −13.52%, 3/3 seeds), and the H96 number matches Table 4 (0.1352 vs 0.136). On Traffic it raises MSE (H96 +7.93%, H720 +7.02%, 0/3 seeds) while MAE improves by 1.84% at both horizons; the H720 +RCF mean (0.4887) is within 0.5% of Table 4 (0.491).

+RG is the best of the three methods on ETTh1 H96 and Electricity H96, and stays with +RCF on ETTh1 H720 and Electricity H720 (still well below orig on Electricity H720). On Traffic it does not recover the MSE loss: +10.60% (H96) and +14.03% (H720) versus orig, and worse than +RCF on both horizons, all six Traffic seeds.

### +RG gate statistics (test set, mean over 3 seeds)

| Dataset | H | mean R² | mean g_in | mean g_out | corr(g_in, R²) | corr(g_out, R²) | +RCF Q rms | +RG Q rms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.041 | 0.025 | 0.790 | −0.906 | +0.937 | 0.031 | 0.058 |
| ETTh1 | 720 | 0.012 | 0.067 | 0.320 | −0.863 | +0.966 | 0.012 | 0.018 |
| Electricity | 96 | 0.293 | 0.729 | 0.984 | +0.484 | +0.631 | 0.229 | 0.320 |
| Electricity | 720 | 0.340 | 0.546 | 0.990 | +0.077 | +0.576 | 0.249 | 0.353 |
| Traffic | 96 | 0.390 | 0.750 | 0.985 | +0.483 | +0.432 | 0.295 | 0.427 |
| Traffic | 720 | 0.470 | 0.666 | 0.983 | +0.364 | +0.428 | 0.336 | 0.517 |

The gate tracks cycle reliability. On ETTh1, periodicity is weak (mean lookback R² 0.041 / 0.012): `g_in` stays near zero and `g_out` is reduced, more so at H720 (0.32), with a strong positive correlation between `g_out` and R². On Electricity and Traffic the cycle fits well (R² 0.29–0.47) and `g_out` stays open at 0.98–0.99, nearly flat across the horizon. The Electricity H720 `g_in` mean of 0.546 is the average of 0.755, 0.780 and 0.103; seed 2028 is the outlier (`corr(g_in, R²)` on that group is 0.27 / 0.37 / −0.42).

## Per-seed test metrics

All three seeds are listed. None is selected as “best”.

| Dataset | H | Mode | seed 2026 MSE / MAE | seed 2027 MSE / MAE | seed 2028 MSE / MAE |
|---|---|---|---|---|---|
| ETTh1 | 96 | orig | 0.3848 / 0.4022 | 0.3835 / 0.4027 | 0.3872 / 0.4031 |
| ETTh1 | 96 | +RCF | 0.3838 / 0.4015 | 0.3823 / 0.4018 | 0.3862 / 0.4023 |
| ETTh1 | 96 | +RG | 0.3825 / 0.4005 | 0.3806 / 0.4008 | 0.3849 / 0.4014 |
| ETTh1 | 720 | orig | 0.5030 / 0.4882 | 0.4912 / 0.4810 | 0.5075 / 0.4913 |
| ETTh1 | 720 | +RCF | 0.5017 / 0.4874 | 0.4900 / 0.4801 | 0.5064 / 0.4905 |
| ETTh1 | 720 | +RG | 0.5020 / 0.4875 | 0.4903 / 0.4803 | 0.5066 / 0.4906 |
| Electricity | 96 | orig | 0.1488 / 0.2414 | 0.1481 / 0.2411 | 0.1472 / 0.2395 |
| Electricity | 96 | +RCF | 0.1358 / 0.2317 | 0.1353 / 0.2310 | 0.1344 / 0.2308 |
| Electricity | 96 | +RG | 0.1339 / 0.2298 | 0.1338 / 0.2297 | 0.1334 / 0.2294 |
| Electricity | 720 | orig | 0.2254 / 0.3089 | 0.2273 / 0.3116 | 0.2379 / 0.3209 |
| Electricity | 720 | +RCF | 0.1911 / 0.2849 | 0.2104 / 0.3016 | 0.1958 / 0.2899 |
| Electricity | 720 | +RG | 0.1942 / 0.2893 | 0.1928 / 0.2868 | 0.2181 / 0.3065 |
| Traffic | 96 | orig | 0.3907 / 0.2645 | 0.3921 / 0.2649 | 0.3911 / 0.2645 |
| Traffic | 96 | +RCF | 0.4222 / 0.2595 | 0.4227 / 0.2606 | 0.4220 / 0.2592 |
| Traffic | 96 | +RG | 0.4327 / 0.2621 | 0.4349 / 0.2627 | 0.4307 / 0.2622 |
| Traffic | 720 | orig | 0.4567 / 0.2970 | 0.4571 / 0.2989 | 0.4562 / 0.2983 |
| Traffic | 720 | +RCF | 0.4941 / 0.2941 | 0.4834 / 0.2917 | 0.4887 / 0.2920 |
| Traffic | 720 | +RG | 0.5240 / 0.2963 | 0.5195 / 0.2975 | 0.5188 / 0.2968 |
