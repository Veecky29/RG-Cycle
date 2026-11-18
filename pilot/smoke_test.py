"""GPU smoke test (a few minutes). Writes JSON to --out.

1. backbone RNG compatibility with the original PAC-Former variant B (needs --pacformer-code)
2. exact output equality orig == +RCF == +RG at initialisation on real batches, shapes
3. 2-epoch ETTh1 H96 training for rcf and rg, run twice -> bitwise determinism
4. full rerun of orig ETTh1 H96/H720 seed 2026 -> bitwise comparison with the existing B results (--orig-dir)
5. timing: ms per training step for orig/rcf/rg on Electricity and Traffic
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402

from rgcycle import build_model, trainer as T  # noqa: E402
from rgcycle.cycle import extra_parameters  # noqa: E402
from rgcycle.data import Series, load_audit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--audit', required=True)
    ap.add_argument('--orig-dir')
    ap.add_argument('--pacformer-code')
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-full-rerun', action='store_true')
    a = ap.parse_args()
    P = json.loads((REPO / 'configs' / 'pilot_day1.json').read_text())
    groups = {(g['dataset'], g['horizon']): g['config'] for g in P['groups']}
    T.setup_cuda()
    audit = load_audit(a.audit)
    rep = dict(checks={})
    tmp = Path(tempfile.mkdtemp())
    ok = True
    try:
        # 1-2
        B = None
        if a.pacformer_code:
            sys.path.insert(0, a.pacformer_code)
            import ablation_cosine50 as abl
            B = abl
        for (ds, h), cfg in groups.items():
            s = Series(ds, audit, a.data_dir)
            a0, _ = s.bounds(h, 'test')
            x, y, t0 = s.batch(torch.arange(a0, a0 + 8), h)
            outs, extra = {}, {}
            for m in ('orig', 'rcf', 'rg'):
                T.seed_all(2026)
                mm = build_model(m, s.channels, h, cfg, P['cycle_len'][ds]).eval()
                extra[m] = extra_parameters(mm)
                with torch.no_grad():
                    outs[m] = mm(x, t0)
                if m == 'orig':
                    sd_orig = mm.state_dict()
                del mm
            c = dict(shape=list(outs['orig'].shape), target_shape=list(y.shape),
                     rcf_equals_orig=bool(torch.equal(outs['rcf'], outs['orig'])),
                     rg_equals_orig=bool(torch.equal(outs['rg'], outs['orig'])), extra_params=extra,
                     backbone_params=sum(v.numel() for v in sd_orig.values()))
            if B is not None:
                T.seed_all(2026)
                old = B.create_variant(s.channels, h, cfg, 'B').eval()
                so = old.state_dict()
                c['backbone_weights_identical_to_PACFormer_B'] = (sorted(so) == sorted(sd_orig)) and all(torch.equal(so[k], sd_orig[k]) for k in so)
                with torch.no_grad():
                    c['output_identical_to_PACFormer_B'] = bool(torch.equal(old(x), outs['orig']))
                del old
                ok &= c['backbone_weights_identical_to_PACFormer_B'] and c['output_identical_to_PACFormer_B']
            ok &= c['rcf_equals_orig'] and c['rg_equals_orig'] and c['shape'] == c['target_shape']
            rep['checks'][f'{ds}/H{h}'] = c
            print(f'{ds} H{h}', c, flush=True)
            s.close()
        # 3
        s = Series('ETTh1', audit, a.data_dir)
        cfg = groups[('ETTh1', 96)]
        det = {}
        for m in ('rcf', 'rg'):
            runs = []
            for k in range(2):
                r = T.fit_job(s, 96, cfg, 2026, tmp / f'det_{m}_{k}', m, 24, -2.0, 10.0, log=lambda _: None, max_epochs=2)
                ep = [json.loads(l) for l in (tmp / f'det_{m}_{k}' / 'epochs.jsonl').read_text().splitlines()]
                runs.append(dict(val=[e['val_mse'] for e in ep], train=[e['train_mse'] for e in ep],
                                 test_mse=r['test']['mse'], test_mae=r['test']['mae'], gates=r['gates'],
                                 sec_per_epoch=[round(e['seconds'], 2) for e in ep]))
            same = runs[0]['val'] == runs[1]['val'] and runs[0]['train'] == runs[1]['train'] and runs[0]['test_mse'] == runs[1]['test_mse']
            det[m] = dict(bitwise_identical_reruns=same, run=runs[0])
            ok &= same
            print('determinism', m, same, runs[0]['val'], runs[0]['test_mse'], flush=True)
        rep['determinism_2epochs_ETTh1_H96'] = det
        # 4
        if a.orig_dir and not a.skip_full_rerun:
            rr = {}
            for h in (96, 720):
                r = T.fit_job(s, h, groups[('ETTh1', h)], 2026, tmp / f'orig_{h}', 'orig', log=lambda _: None)
                old = json.loads((Path(a.orig_dir) / 'ETTh1' / f'H{h}' / 'B' / 'seed_2026' / 'result.json').read_text())
                same = (r['test']['mse'] == old['test']['mse'] and r['test']['mae'] == old['test']['mae']
                        and r['best_val_mse'] == old['best_val_mse'] and r['best_epoch'] == old['best_epoch'])
                rr[f'H{h}'] = dict(new=dict(test=r['test'], best_val_mse=r['best_val_mse'], best_epoch=r['best_epoch'], epochs_ran=r['epochs_ran']),
                                   old=dict(test=old['test'], best_val_mse=old['best_val_mse'], best_epoch=old['best_epoch']),
                                   bitwise_identical=same)
                ok &= same
                print('orig rerun ETTh1', h, same, r['test']['mse'], old['test']['mse'], flush=True)
            rep['orig_rerun_vs_existing_B'] = rr
        s.close()
        # 5
        tim = {}
        for ds in ('Electricity', 'Traffic'):
            s = Series(ds, audit, a.data_dir)
            for h in (96, 720):
                cfg = groups[(ds, h)]
                n = s.bounds(h, 'train')[1]
                for m in ('orig', 'rcf', 'rg'):
                    T.seed_all(2026)
                    mm = build_model(m, s.channels, h, cfg, P['cycle_len'][ds]).train()
                    opt = T.make_optimizer(mm, cfg['learning_rate'], 10.0 if m == 'rg' else 1.0)
                    g = torch.Generator().manual_seed(0)
                    order = torch.randperm(n, generator=g)
                    for i in range(70):
                        if i == 10:
                            torch.cuda.synchronize()
                            t = time.perf_counter()
                        x, y, t0 = s.batch(order[i * 32:(i + 1) * 32], h)
                        opt.zero_grad()
                        torch.nn.functional.mse_loss(mm(x, t0), y).backward()
                        opt.step()
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t) / 60 * 1000
                    tim[f'{ds}/H{h}/{m}'] = dict(ms_per_step=round(ms, 2), steps_per_epoch=n // 32,
                                                 train_s_per_epoch=round(ms * (n // 32) / 1000, 1))
                    print('timing', ds, h, m, tim[f'{ds}/H{h}/{m}'], flush=True)
                    del mm, opt
            s.close()
        rep['timing_single_process'] = tim
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rep['SMOKE_OK'] = bool(ok)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print('SMOKE_OK' if ok else 'SMOKE_FAILED', flush=True)


if __name__ == '__main__':
    main()
