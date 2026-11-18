"""Train a single job.

    python train.py --dataset Traffic --horizon 96 --mode rg --seed 2026 \
        --lr 0.001 --d-model 256 --kernel-size 5 --cycle 168 --data-dir data/benchmark --out outputs/run
"""
import argparse
import json
from pathlib import Path

from rgcycle import trainer as T
from rgcycle.data import Series, load_audit

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--mode', choices=['orig', 'rcf', 'rg'], required=True)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--d-model', type=int, required=True)
    p.add_argument('--kernel-size', type=int, default=None, help='only reproduces PAC-Former random-draw order')
    p.add_argument('--cycle', type=int, default=None, help='cycle length W (needed for rcf/rg)')
    p.add_argument('--gate-init-bias', type=float, default=-2.0)
    p.add_argument('--gate-lr-mult', type=float, default=10.0)
    p.add_argument('--max-epochs', type=int, default=T.MAX_EPOCHS)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'benchmark')
    p.add_argument('--audit', type=Path, default=ROOT / 'configs' / 'datasets.json')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.mode != 'orig' and a.cycle is None:
        p.error('--cycle is required for rcf/rg')
    T.setup_cuda()
    cfg = dict(learning_rate=a.lr, d_model=a.d_model)
    if a.kernel_size is not None:
        cfg['kernel_size'] = a.kernel_size
    s = Series(a.dataset, load_audit(a.audit), a.data_dir)
    try:
        r = T.fit_job(s, a.horizon, cfg, a.seed, a.out, a.mode, a.cycle, a.gate_init_bias, a.gate_lr_mult,
                      max_epochs=a.max_epochs)
    finally:
        s.close()
    print(json.dumps(dict(test=r['test'], best_epoch=r['best_epoch'], gates=r['gates']), indent=2))


if __name__ == '__main__':
    main()
