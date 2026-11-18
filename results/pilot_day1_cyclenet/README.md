# Day-1 pilot results (CycleNet protocol)

**Status: partial.** Snapshot at **2026-11-19 03:10:35 (Malaysia Time, UTC+8)**: **35/54 jobs** finished, the scheduler still running. Three of six groups are complete. A final update will replace this snapshot when all 54 jobs finish.

本目录是第一天试点在 CycleNet 论文设定下的复现快照：冻结的协议与超参、已完成任务的逐 seed 指标，以及已完成组的汇总。主表是 **3 个 seed（2026 / 2027 / 2028）的均值±标准差**，不是 best-of-seed。

This directory is a snapshot of the day-1 pilot under the CycleNet (NeurIPS 2024) iTransformer / CycleiTransformer protocol: the frozen protocol and hyper-parameters, per-seed metrics for finished jobs, and summaries for completed groups. The main tables are **mean ± standard deviation over three seeds** (2026 / 2027 / 2028), not a best-of-seed selection.

## Files

| File | What it is |
|---|---|
| [`protocol.json`](protocol.json) | Frozen `--init` protocol (config hash, code hashes, 54 jobs, references). Host paths (`data_dir`, `audit`, `config_file`) are from the machine that ran the pilot. |
| [`jobs.csv`](jobs.csv) | All 54 jobs and their hyper-parameters |
| [`results.csv`](results.csv) | Per-seed test MSE / MAE and training diagnostics for the 35 finished jobs (includes 8 unfinished-group rows for Electricity H720; those are **not** summarised below) |
| [`group_done/`](group_done/) | Per-group JSON written when all 9 jobs of a group finish |
| [`group_done.log`](group_done.log) | One-line group summaries, in completion order |
| [`status.json`](status.json) | Scheduler snapshot at the time above |
| [`scheduler.out`](scheduler.out) | Scheduler start / done log |
| [`SNAPSHOT.md`](SNAPSHOT.md) | Snapshot time, progress, and hash checks |
| [`../../configs/pilot_day1_cyclenet.json`](../../configs/pilot_day1_cyclenet.json) | Source config (SHA-256 `482cda3f5f810b5118c77cba7e032868ca954744dc0a29ea533d35eb0b4c11ed`, identical to `protocol.json` `config_sha256`). Code SHA-256 values in the protocol match the files in this repository. |

## Protocol and hyper-parameters

Common settings (every job): lookback `seq_len = 96`, `n_heads = 8`, GELU, RevIN / Non-stationary-Transformer normalisation, Adam, MSE, CycleNet learning-rate schedule (epoch 1 at `lr/25`, epochs 2–4 at `lr`, then ×0.8 per epoch), shuffled training and validation with `drop_last=True`, test over every window (`drop_last=False`), early stopping on validation MSE. Seeds: 2026, 2027, 2028. Two workers. +RG only: gate bias `b0 = −2` (gate ≈ 0.12 at init) and 10× learning rate for the gate parameters. The source of every value is recorded in `configs/pilot_day1_cyclenet.json`.

| Dataset | H | e_layers | d_model | d_ff | dropout | batch | lr | epochs | patience | cycle W | Hyper-parameter source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | 96 | 2 | 256 | 256 | 0.1 | 32 | 1e-4 | 10 | 3 | 24 | Official iTransformer ETTh1 script and `run.py` defaults; cycle length from CycleNet `scripts/CycleNet/Linear-Input-96/etth1.sh`. CycleNet Table 4 has no iTransformer / +RCF run on ETTh1. |
| ETTh1 | 720 | 2 | 512 | 512 | 0.1 | 32 | 1e-4 | 10 | 3 | 24 | Same, pred_len 720 block of the official iTransformer ETTh1 script. |
| Electricity | 96 | 3 | 512 | 512 | 0.1 | 16 | 5e-4 | 10 | 3 | 168 | CycleNet `scripts/iTransformer/electricity.sh` |
| Electricity | 720 | 3 | 512 | 512 | 0.1 | 16 | 5e-4 | 10 | 3 | 168 | Same script (not yet a complete group). |
| Traffic | 96 | 4 | 512 | 512 | 0.1 | 16 | 1e-3 | 30 | 5 | 168 | CycleNet `scripts/iTransformer/traffic.sh` (not started at this snapshot, except orig seed 2026 which had just begun). |
| Traffic | 720 | 4 | 512 | 512 | 0.1 | 16 | 1e-3 | 30 | 5 | 168 | Same script (not started). |

## Completed groups

Comparison bases, stated once and used throughout:

- **orig** is iTransformer trained in this pilot, under the CycleNet protocol.
- **+RCF** is CycleNet Residual Cycle Forecasting on the same backbone and recipe (CycleiTransformer).
- **+RG** is RG-Cycle on the same backbone and recipe.
- **vs orig** is the relative change of the 3-seed mean against the orig mean from this same pilot (`100 × (method − orig) / orig`). Negative is an improvement (lower error).
- **iTransformer paper** is Liu et al., ICLR 2024, official reported test MSE / MAE.
- **CycleNet Table 4** is Lin et al., NeurIPS 2024, Table 4, the iTransformer + RCF numbers. That table covers Electricity and Traffic only; ETTh1 has no official iTransformer + RCF reference.

`3/3` means all three seeds of that method have a lower test MSE than the same seed of orig.

### Test MSE / MAE (mean ± SD over 3 seeds)

| Dataset | H | orig MSE | orig MAE | +RCF MSE | +RCF MAE | +RG MSE | +RG MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.3852±0.0019 | 0.4026±0.0004 | 0.3841±0.0020 | 0.4019±0.0004 | 0.3827±0.0021 | 0.4009±0.0004 |
| ETTh1 | 720 | 0.5005±0.0084 | 0.4868±0.0053 | 0.4994±0.0085 | 0.4860±0.0053 | 0.4996±0.0084 | 0.4861±0.0053 |
| Electricity | 96 | 0.1480±0.0008 | 0.2407±0.0010 | 0.1352±0.0007 | 0.2312±0.0005 | 0.1337±0.0003 | 0.2296±0.0002 |

### Relative change and paper references

| Dataset | H | orig vs iTransformer paper (MSE) | +RCF vs orig (MSE / MAE) | +RG vs orig (MSE / MAE) | +RCF vs CycleNet Table 4 (MSE) | +RG vs CycleNet Table 4 (MSE) |
|---|---|---|---|---|---|---|
| ETTh1 | 96 | 0.3852 vs 0.386 / 0.405, **−0.22%** | **−0.28% / −0.20%**, 3/3 better | **−0.65% / −0.43%**, 3/3 better | no official number | no official number |
| ETTh1 | 720 | 0.5005 vs 0.503 / 0.491, **−0.49%** | **−0.24% / −0.17%**, 3/3 better | **−0.18% / −0.14%**, 3/3 better | no official number | no official number |
| Electricity | 96 | 0.1480 vs 0.148 / 0.240, **+0.03%** | **−8.71% / −3.95%**, 3/3 better | **−9.71% / −4.59%**, 3/3 better | 0.1352 vs 0.136 / 0.231, **−0.62%** | 0.1337 vs 0.136, **−1.71%** |

On ETTh1 the three methods are close. Periodicity is weak (mean lookback R² 0.041 at H96, 0.012 at H720), the learned cycle is small, and the H720 gaps sit inside seed noise (orig MSE SD ≈ 0.008). On Electricity H96 the orig run matches the iTransformer paper, +RCF matches CycleNet Table 4, and +RG improves on both, with a tighter seed spread than +RCF.

### +RG gate statistics (test set, mean over 3 seeds)

| Dataset | H | mean R² | mean g_in | mean g_out | corr(g_out, R²) | +RCF Q rms | +RG Q rms |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.041 | 0.025 | 0.790 | +0.937 | 0.031 | 0.058 |
| ETTh1 | 720 | 0.012 | 0.067 | 0.320 | +0.966 | 0.012 | 0.018 |
| Electricity | 96 | 0.293 | 0.729 | 0.984 | +0.631 | 0.229 | 0.320 |

The gate tracks cycle reliability: on Electricity, where the cycle fits well, both gates stay mostly open; on ETTh1, where it does not, `g_in` stays near zero and `g_out` is reduced, more so at H720.

## Pending (at this snapshot)

- **Electricity H720** — 8/9 jobs finished; +RG seed 2027 was still training. No group mean is reported here.
- **Traffic H96** — orig seed 2026 had just started; the other 8 jobs had not started.
- **Traffic H720** — not started.

Traffic is the remaining comparison with CycleNet Table 4 (that table reports iTransformer + RCF at 0.415 / 0.263 for H96 and 0.491 / 0.294 for H720, versus the iTransformer paper values 0.395 / 0.268 and 0.467 / 0.302).

## Per-seed test metrics (completed groups only)

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
