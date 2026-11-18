"""Train a single job.

    # take all hyper-parameters from a pilot config group
    python train.py --config configs/pilot_day1_cyclenet.json --dataset Traffic --horizon 96 --mode rg --seed 2026 \
        --out outputs/traffic_h96_rg_s2026
"""
import argparse
import json
from pathlib import Path

from rgcycle import trainer as T
from rgcycle.data import Series, load_audit

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT / 'configs' / 'pilot_day1_cyclenet.json')
    p.add_argument('--dataset', required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--mode', choices=['orig', 'rcf', 'rg'], required=True)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--cycle', type=int, default=None, help='override cycle length W')
    p.add_argument('--max-epochs', type=int, default=T.MAX_EPOCHS)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'benchmark')
    p.add_argument('--audit', type=Path, default=ROOT / 'configs' / 'datasets.json')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    P = json.loads(a.config.read_text())
    g = next((x for x in P['groups'] if x['dataset'] == a.dataset and x['horizon'] == a.horizon), None)
    if g is None:
        p.error(f'{a.dataset} H{a.horizon} not in {a.config}')
    cycle = a.cycle or P['cycle_len'][a.dataset]
    T.setup_cuda()
    s = Series(a.dataset, load_audit(a.audit), a.data_dir)
    try:
        r = T.fit_job(s, a.horizon, g['config'], a.seed, a.out, a.mode, cycle, P['gate']['init_bias'],
                      P['gate']['lr_mult'], max_epochs=a.max_epochs)
    finally:
        s.close()
    print(json.dumps(dict(test=r['test'], best_epoch=r['best_epoch'], gates=r['gates']), indent=2))


if __name__ == '__main__':
    main()
