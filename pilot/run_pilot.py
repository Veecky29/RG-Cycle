"""Day-1 pilot runner (resumable, 2 GPU workers).

    # 1) freeze protocol + job list (no training)
    python pilot/run_pilot.py --run-dir RUN --init --data-dir DATA --audit AUDIT [--orig-dir DIR]
    # 2) scheduler (normally started through pilot/launch.sh)
    python pilot/run_pilot.py --run-dir RUN

Outputs in RUN: protocol.json, jobs.csv, runs/<ds>/H<h>/<mode>/seed_<s>/{result.json,best.pt,epochs.jsonl,worker.out},
results/status.json, results/results.csv, results/group_done/<ds>_H<h>.json, results/group_done.log, scheduler.out.

orig (the backbone alone) is reused from existing PAC-Former variant-B runs when --orig-dir is given
(<orig-dir>/<ds>/H<h>/B/seed_<s>/result.json); otherwise orig jobs are trained here.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CODE_FILES = ['rgcycle/__init__.py', 'rgcycle/backbones.py', 'rgcycle/cycle.py', 'rgcycle/data.py',
              'rgcycle/trainer.py', 'pilot/run_pilot.py']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jdir(run, j):
    return Path(run) / 'runs' / j['dataset'] / f"H{j['horizon']}" / j['mode'] / f"seed_{j['seed']}"


def done(run, j):
    p = jdir(run, j) / 'result.json'
    return p.exists() and read(p).get('status') == 'completed'


def init(run, data_dir, audit, orig_dir, config):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    cfgall = read(config)
    seeds, jobs, orig = cfgall['seeds'], [], {}
    modes = [m for m in cfgall['modes'] if m != 'orig']
    if not orig_dir:
        modes = ['orig'] + modes
    for g in cfgall['groups']:
        ds, h, cfg = g['dataset'], g['horizon'], g['config']
        key = f'{ds}/H{h}'
        if orig_dir:
            rs = [read(Path(orig_dir) / ds / f'H{h}' / 'B' / f'seed_{s}' / 'result.json') for s in seeds]
            for r, s in zip(rs, seeds):
                assert r['status'] == 'completed' and r['seed'] == s, (key, s)
                assert all(r['config'].get(k) == v for k, v in cfg.items() if k in r['config']), (key, s)
            orig[key] = dict(source=str(Path(orig_dir) / ds / f'H{h}' / 'B'), seeds=seeds,
                             test_mse=[r['test']['mse'] for r in rs], test_mae=[r['test']['mae'] for r in rs],
                             best_epoch=[r['best_epoch'] for r in rs], epochs_ran=[r['epochs_ran'] for r in rs],
                             elapsed_s=[r['elapsed_seconds'] for r in rs])
        for m in modes:
            for s in seeds:
                jobs.append(dict(dataset=ds, horizon=h, mode=m, seed=s, cycle_len=cfgall['cycle_len'][ds], config=cfg))
    protocol = dict(name=cfgall['name'], created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    data_dir=str(data_dir), audit=str(audit), audit_sha256=sha(audit), orig_dir=str(orig_dir or ''),
                    gate=cfgall['gate'], cycle_len=cfgall['cycle_len'], seeds=seeds, workers=cfgall['workers'],
                    config_file=str(Path(config).resolve()), config_sha256=sha(config),
                    pilot_config={k: v for k, v in cfgall.items() if k not in ('refs',)},
                    code_sha256={f: sha(REPO / f) for f in CODE_FILES}, refs=cfgall['refs'], orig=orig, jobs=jobs)
    (run / 'results' / 'group_done').mkdir(parents=True, exist_ok=True)
    (run / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    with open(run / 'jobs.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        keys = [k for k in jobs[0]['config'] if k not in ('backbone', 'recipe')]
        w.writerow(['#', 'dataset', 'H', 'mode', 'seed'] + keys + ['cycle_len', 'gate_init_bias', 'gate_lr_mult'])
        for i, j in enumerate(jobs, 1):
            rg = j['mode'] == 'rg'
            w.writerow([i, j['dataset'], j['horizon'], j['mode'], j['seed']] + [j['config'].get(k, '') for k in keys]
                       + [j['cycle_len'] if j['mode'] != 'orig' else '', cfgall['gate']['init_bias'] if rg else '',
                          cfgall['gate']['lr_mult'] if rg else ''])
    print(f'{len(jobs)} jobs frozen in {run}/protocol.json (orig reused: {bool(orig_dir)})')


def worker(run, ds, h, mode, seed):
    from rgcycle import trainer as T
    from rgcycle.data import Series, load_audit
    P = read(Path(run) / 'protocol.json')
    for f, s in P['code_sha256'].items():
        assert sha(REPO / f) == s, f'code changed since --init: {f}'
    j = next(x for x in P['jobs'] if x['dataset'] == ds and x['horizon'] == h and x['mode'] == mode and x['seed'] == seed)
    assert sha(P['config_file']) == P['config_sha256'], 'pilot config changed since --init'
    cfg = j['config']
    path = jdir(run, j)
    T.setup_cuda()
    s = Series(ds, load_audit(P['audit']), P['data_dir'])
    try:
        T.fit_job(s, h, cfg, seed, path, mode, j['cycle_len'] if mode != 'orig' else None,
                  P['gate']['init_bias'], P['gate']['lr_mult'], log=lambda m: print(m, flush=True),
                  status_cb=lambda x: T.atomic_json(path / 'worker_status.json', x))
    finally:
        s.close()


def collect(run, P):
    rows = []
    for j in P['jobs']:
        if not done(run, j):
            continue
        r = read(jdir(run, j) / 'result.json')
        g = r.get('gates') or {}
        rows.append(dict(dataset=j['dataset'], horizon=j['horizon'], mode=j['mode'], seed=j['seed'],
                         lr=j['config']['learning_rate'], d_model=j['config']['d_model'],
                         batch_size=j['config'].get('batch_size', 32), cycle_len=j['cycle_len'],
                         best_epoch=r['best_epoch'], epochs_ran=r['epochs_ran'], val_mse=r['best_val_mse'],
                         test_mse=r['test']['mse'], test_mae=r['test']['mae'],
                         extra_params=r['extra_parameters']['total'], Q_rms=g.get('Q_rms'),
                         mean_g_in=g.get('mean_g_in'), mean_g_out=g.get('mean_g_out'),
                         corr_g_out_R2=g.get('corr_g_out_R2'), elapsed_s=round(r['elapsed_seconds'], 1)))
    if rows:
        out = Path(run) / 'results' / 'results.csv'
        with open(out.with_suffix('.tmp'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        out.with_suffix('.tmp').replace(out)
    return rows


def _stats(v):
    v = np.asarray(v, dtype=float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                cv_pct=float(100 * v.std(ddof=1) / v.mean()) if len(v) > 1 else 0.0, values=v.tolist())


def group_summary(run, P, ds, h):
    key = f'{ds}/H{h}'
    run = Path(run)
    res = {m: sorted([read(jdir(run, j) / 'result.json') for j in P['jobs']
                      if j['dataset'] == ds and j['horizon'] == h and j['mode'] == m and done(run, j)],
                     key=lambda r: r['seed']) for m in ('orig', 'rcf', 'rg')}
    if P['orig'].get(key):
        o = P['orig'][key]
        orig_mse, orig_mae, orig_src = o['test_mse'], o['test_mae'], 'reused PAC-Former B runs: ' + o['source']
    else:
        orig_mse = [r['test']['mse'] for r in res['orig']]
        orig_mae = [r['test']['mae'] for r in res['orig']]
        orig_src = 'trained in this pilot'
    om, refs = np.mean(orig_mse), P['refs']
    itr = refs['iTransformer_paper'].get(key)
    cyc = refs['CycleNet_Table4_iTransformer_plus_RCF'].get(key)
    out = dict(dataset=ds, horizon=h, orig=dict(source=orig_src, mse=_stats(orig_mse), mae=_stats(orig_mae),
                                                vs_iTransformer_paper_pct=float(100 * (om / itr['mse'] - 1)) if itr else None,
                                                best_epoch=[r['best_epoch'] for r in res['orig']],
                                                elapsed_s=[round(r['elapsed_seconds']) for r in res['orig']]),
               refs=dict(iTransformer_paper=itr, CycleNet_iTr_RCF_paper=cyc), modes={})
    for m in ('rcf', 'rg'):
        rr = res[m]
        if not rr:
            continue
        mse = np.array([r['test']['mse'] for r in rr])
        mae = np.array([r['test']['mae'] for r in rr])
        paired = [float(r['test']['mse']) < float(orig_mse[P['seeds'].index(r['seed'])]) for r in rr]
        x = dict(seeds=[r['seed'] for r in rr], mse=_stats(mse), mae=_stats(mae),
                 vs_orig_mse_pct=float(100 * (mse.mean() / om - 1)),
                 vs_orig_mae_pct=float(100 * (mae.mean() / np.mean(orig_mae) - 1)),
                 seeds_beating_orig=f'{sum(paired)}/{len(rr)}',
                 vs_iTransformer_paper_pct=float(100 * (mse.mean() / itr['mse'] - 1)) if itr else None,
                 vs_CycleNet_iTr_RCF_paper_pct=float(100 * (mse.mean() / cyc['mse'] - 1)) if cyc else None,
                 best_epoch=[r['best_epoch'] for r in rr], extra_parameters=rr[0]['extra_parameters'],
                 elapsed_s=[round(r['elapsed_seconds']) for r in rr])
        gs = [r['gates'] for r in rr if r.get('gates')]
        if gs:
            x['Q_rms_mean'] = float(np.mean([g['Q_rms'] for g in gs]))
        if m == 'rg' and gs:
            x['gates'] = dict(mean_g_in=float(np.mean([g['mean_g_in'] for g in gs])),
                              mean_g_out=float(np.mean([g['mean_g_out'] for g in gs])),
                              mean_g_out_by_horizon_bucket={b: float(np.mean([g['mean_g_out_by_horizon_bucket'][b] for g in gs]))
                                                            for b in gs[0]['mean_g_out_by_horizon_bucket']},
                              corr_g_in_R2=[g['corr_g_in_R2'] for g in gs], corr_g_out_R2=[g['corr_g_out_R2'] for g in gs],
                              mean_R2=float(np.mean([g['mean_R2'] for g in gs])), per_seed=gs)
        out['modes'][m] = x
    out['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    gd = run / 'results' / 'group_done'
    (gd / f'{ds}_H{h}.json').write_text(json.dumps(out, indent=2))
    parts = [f"GROUP_DONE {ds} H{h} | orig {om:.4f}" + (f" iTr(paper) {itr['mse']:.3f}" if itr else '')
             + (f" CycleNet-iTr+RCF(paper) {cyc['mse']:.3f}" if cyc else '')]
    for m, x in out['modes'].items():
        t = (f"{m} {x['mse']['mean']:.4f}±{x['mse']['sd']:.4f} ({x['vs_orig_mse_pct']:+.2f}% vs orig, "
             f"{x['seeds_beating_orig']} better, CV {x['mse']['cv_pct']:.2f}%)")
        if 'gates' in x:
            g = x['gates']
            t += f" g_in {g['mean_g_in']:.3f} g_out {g['mean_g_out']:.3f} buckets " + '/'.join(
                f'{v:.3f}' for v in g['mean_g_out_by_horizon_bucket'].values())
        parts.append(t)
    line = ' | '.join(parts)
    with open(run / 'results' / 'group_done.log', 'a') as fh:
        fh.write(out['utc'] + ' ' + line + '\n')
    print(line, flush=True)


def scheduler(run):
    run = Path(run)
    P = read(run / 'protocol.json')
    jobs, nw = P['jobs'], P['workers']
    groups = {}
    for j in jobs:
        groups.setdefault((j['dataset'], j['horizon']), []).append(j)
    pending = [j for j in jobs if not done(run, j)]
    active, t0 = [], time.time()

    def status(state='running'):
        act = []
        for x in active:
            ws = jdir(run, x) / 'worker_status.json'
            act.append(dict({k: x[k] for k in ('dataset', 'horizon', 'mode', 'seed')}, pid=x['proc'].pid,
                            **(read(ws) if ws.exists() else {})))
        tmp = run / 'results' / 'status.json.tmp'
        tmp.write_text(json.dumps(dict(state=state, completed=sum(done(run, j) for j in jobs), total=len(jobs),
                                       active=act, elapsed_s=round(time.time() - t0),
                                       utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=2))
        tmp.replace(run / 'results' / 'status.json')

    print(f'SCHEDULER START {len(pending)} pending of {len(jobs)}', flush=True)
    status()
    while pending or active:
        while pending and len(active) < nw:
            j = pending.pop(0)
            jdir(run, j).mkdir(parents=True, exist_ok=True)
            fh = open(jdir(run, j) / 'worker.out', 'a')
            p = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--run-dir', str(run), '--worker',
                                  '--dataset', j['dataset'], '--horizon', str(j['horizon']), '--mode', j['mode'],
                                  '--seed', str(j['seed'])], stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
            active.append(dict(j, proc=p, fh=fh))
            print(f"START {j['dataset']} H{j['horizon']} {j['mode']} seed{j['seed']}", flush=True)
        time.sleep(5)
        for x in list(active):
            rc = x['proc'].poll()
            if rc is None:
                continue
            x['fh'].close()
            active.remove(x)
            if rc != 0 or not done(run, x):
                status('failed')
                print(f"FAILED {x['dataset']} H{x['horizon']} {x['mode']} seed{x['seed']} rc={rc}", flush=True)
                for y in active:
                    y['proc'].wait()
                raise SystemExit(1)
            print(f"DONE {x['dataset']} H{x['horizon']} {x['mode']} seed{x['seed']}", flush=True)
            collect(run, P)
            if all(done(run, k) for k in groups[(x['dataset'], x['horizon'])]):
                group_summary(run, P, x['dataset'], x['horizon'])
        status()
    collect(run, P)
    status('completed')
    print('ALL_PILOT_JOBS_COMPLETE', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--init', action='store_true')
    ap.add_argument('--data-dir')
    ap.add_argument('--audit')
    ap.add_argument('--orig-dir')
    ap.add_argument('--config', default=str(REPO / 'configs' / 'pilot_day1_cyclenet.json'))
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--dataset')
    ap.add_argument('--horizon', type=int)
    ap.add_argument('--mode')
    ap.add_argument('--seed', type=int)
    a = ap.parse_args()
    if a.init:
        init(a.run_dir, a.data_dir, a.audit, a.orig_dir, a.config)
    elif a.worker:
        worker(a.run_dir, a.dataset, a.horizon, a.mode, a.seed)
    else:
        scheduler(a.run_dir)
