"""GPU smoke test for the CycleNet-protocol pilot (configs/pilot_day1_cyclenet.json). Writes JSON to --out.

1. equivalence with the official CycleNet code (needs --cyclenet-src, a checkout of ACAT-SCUT/CycleNet):
   same seed -> identical initial weights; identical outputs for CycleiTransformer vs our orig/+RCF wrapper
2. orig == +RCF == +RG at initialisation on real batches (every group), shapes, parameter counts
3. ETTh1 H96, 2 epochs, each mode run twice -> bitwise determinism; checks the CycleNet lr schedule
4. timing: ms per training step for every group x mode (single process)
5. memory/throughput with 2 concurrent workers on the heaviest group (Traffic H720, rg)
"""
import argparse
import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CONFIG = REPO / 'configs' / 'pilot_day1_cyclenet.json'


def _timing(args):
    """ms/step over 60 steps after 10 warm-up steps; returns peak memory too."""
    ds, h, mode, data_dir, audit_path, n_steps = args
    import torch
    from rgcycle import build_model, trainer as T
    from rgcycle.data import Series, load_audit
    T.setup_cuda()
    P = json.loads(CONFIG.read_text())
    g = next(x for x in P['groups'] if x['dataset'] == ds and x['horizon'] == h)
    cfg = g['config']
    s = Series(ds, load_audit(audit_path), data_dir)
    n = s.bounds(h, 'train')[1]
    bs = cfg['batch_size']
    T.seed_all(2026)
    m = build_model(mode, s.channels, h, cfg, P['cycle_len'][ds]).train()
    opt = T.make_optimizer(m, cfg['learning_rate'], 10.0 if mode == 'rg' else 1.0)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    torch.cuda.reset_peak_memory_stats()
    for i in range(10 + n_steps):
        if i == 10:
            torch.cuda.synchronize()
            t = time.perf_counter()
        x, y, t0 = s.batch(order[i * bs:(i + 1) * bs], h)
        opt.zero_grad()
        torch.nn.functional.mse_loss(m(x, t0), y).backward()
        opt.step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / n_steps * 1000
    # one validation pass timing (CycleNet val: shuffled, drop_last)
    t = time.perf_counter()
    T.cyclenet_val_loss(m, s, h, bs, torch.Generator().manual_seed(1))
    torch.cuda.synchronize()
    val_s = time.perf_counter() - t
    out = dict(ms_per_step=round(ms, 2), steps_per_epoch=n // bs, train_s_per_epoch=round(ms * (n // bs) / 1000, 1),
               val_s=round(val_s, 1), epoch_s=round(ms * (n // bs) / 1000 + val_s, 1),
               peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20),
               peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2 ** 20))
    s.close()
    return out


def official_equivalence(cyclenet_src, rep):
    """Compare our CycleNetITransformer(+RCF) with CycleNet's own models/CycleiTransformer.py."""
    import torch
    from rgcycle import build_model, trainer as T
    sys.path.insert(0, str(cyclenet_src))
    from models import CycleiTransformer as CI
    res = {}
    for (C, h, el, dm, dff, W) in [(7, 96, 2, 256, 256, 24), (862, 720, 4, 512, 512, 168)]:
        cfg = dict(backbone='cyclenet_itransformer', d_model=dm, d_ff=dff, e_layers=el, n_heads=8, dropout=0.1,
                   activation='gelu', use_revin=1)
        a = types.SimpleNamespace(seq_len=96, pred_len=h, output_attention=False, use_revin=1, d_model=dm, embed='timeF',
                                  freq='h', dropout=0.1, factor=1, n_heads=8, d_ff=dff, activation='gelu', e_layers=el,
                                  cycle=W, enc_in=C)
        torch.manual_seed(2026)
        off = CI.Model(a).float().cuda().eval()
        T.seed_all(2026)
        ours = build_model('rcf', C, h, cfg, W).eval()
        T.seed_all(2026)
        orig = build_model('orig', C, h, cfg, W).eval()
        so = list(off.state_dict().values())
        sr = list(ours.backbone.state_dict().values()) + [ours.cycle.data]
        # CycleNet registers cycleQueue last; our wrapper keeps it separately
        so_bb, so_q = so[:-1], so[-1]
        init_equal = len(so_bb) == len(sr) - 1 and all(torch.equal(p, q) for p, q in zip(so_bb, sr[:-1])) and torch.equal(so_q, sr[-1])
        with torch.no_grad():
            torch.manual_seed(7)
            q = torch.randn(W, C, device='cuda') * 0.3  # non-zero cycle to exercise RCF
            off.cycleQueue.data.copy_(q)
            ours.cycle.data.copy_(q)
            x = torch.randn(16, 96, C, device='cuda') * 2 + 1
            t0 = torch.randint(0, 10000, (16,), device='cuda')
            ci = (t0 + 96) % W  # CycleNet passes cycle_index[s_end]
            y_off = off(x.clone(), ci.int())
            y_ours = ours(x.clone(), t0)
            off.cycleQueue.data.zero_()
            y_off0 = off(x.clone(), ci.int())
            y_orig = orig(x.clone(), t0)
        res[f'C{C}_H{h}'] = dict(init_weights_identical=bool(init_equal),
                                 rcf_output_max_abs_diff=float((y_off - y_ours).abs().max()),
                                 rcf_output_bitwise_equal=bool(torch.equal(y_off, y_ours)),
                                 orig_vs_official_zero_cycle_max_abs_diff=float((y_off0 - y_orig).abs().max()))
        print('official-equivalence', res[f'C{C}_H{h}'], flush=True)
    rep['official_equivalence'] = res
    return all(r['init_weights_identical'] and r['rcf_output_max_abs_diff'] < 1e-5
               and r['orig_vs_official_zero_cycle_max_abs_diff'] < 1e-5 for r in res.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--audit', required=True)
    ap.add_argument('--cyclenet-src')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    from rgcycle import build_model, trainer as T
    from rgcycle.cycle import extra_parameters
    from rgcycle.data import Series, load_audit
    P = json.loads(CONFIG.read_text())
    T.setup_cuda()
    audit = load_audit(a.audit)
    rep, ok = dict(checks={}), True
    if a.cyclenet_src:
        ok &= official_equivalence(a.cyclenet_src, rep)
    for g in P['groups']:
        ds, h, cfg = g['dataset'], g['horizon'], g['config']
        s = Series(ds, audit, a.data_dir)
        a0, _ = s.bounds(h, 'test')
        x, y, t0 = s.batch(torch.arange(a0, a0 + 8), h)
        outs, extra = {}, {}
        for m in ('orig', 'rcf', 'rg'):
            T.seed_all(2026)
            mm = build_model(m, s.channels, h, cfg, P['cycle_len'][ds]).eval()
            extra[m] = extra_parameters(mm)
            if m == 'orig':
                bb = sum(p.numel() for p in mm.parameters())
            with torch.no_grad():
                outs[m] = mm(x, t0)
            del mm
        c = dict(shape=list(outs['orig'].shape), target_shape=list(y.shape), backbone_params=bb,
                 rcf_equals_orig=bool(torch.equal(outs['rcf'], outs['orig'])),
                 rg_equals_orig=bool(torch.equal(outs['rg'], outs['orig'])), extra_params=extra)
        ok &= c['rcf_equals_orig'] and c['rg_equals_orig'] and c['shape'] == c['target_shape']
        rep['checks'][f'{ds}/H{h}'] = c
        print(f'{ds} H{h}', c, flush=True)
        s.close()
    tmp = Path(tempfile.mkdtemp())
    try:
        s = Series('ETTh1', audit, a.data_dir)
        cfg = next(g['config'] for g in P['groups'] if g['dataset'] == 'ETTh1' and g['horizon'] == 96)
        det = {}
        for m in ('orig', 'rcf', 'rg'):
            runs = []
            for k in range(2):
                r = T.fit_job(s, 96, cfg, 2026, tmp / f'{m}_{k}', m, 24, -2.0, 10.0, log=lambda _: None, max_epochs=2)
                ep = [json.loads(l) for l in (tmp / f'{m}_{k}' / 'epochs.jsonl').read_text().splitlines()]
                runs.append(dict(lr=[e['lr_last'] for e in ep], val=[e['val_mse'] for e in ep],
                                 train=[e['train_mse'] for e in ep], test_mse=r['test']['mse'], test_mae=r['test']['mae'],
                                 full_val_mse=r['val']['mse'], gates=r['gates'], sec_per_epoch=[round(e['seconds'], 2) for e in ep]))
            same = all(runs[0][k] == runs[1][k] for k in ('val', 'train', 'test_mse', 'test_mae'))
            lr_ok = runs[0]['lr'] == [cfg['learning_rate'] / 25, cfg['learning_rate']]
            det[m] = dict(bitwise_identical_reruns=same, lr_schedule_ok=lr_ok, run=runs[0])
            ok &= same and lr_ok
            print('determinism', m, same, 'lr', runs[0]['lr'], 'val', runs[0]['val'], 'test', runs[0]['test_mse'], flush=True)
        rep['determinism_2epochs_ETTh1_H96'] = det
        s.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    torch.cuda.empty_cache()
    ctx = mp.get_context('spawn')
    tim = {}
    for g in P['groups']:
        for m in ('orig', 'rcf', 'rg'):
            with ctx.Pool(1) as pool:
                tim[f"{g['dataset']}/H{g['horizon']}/{m}"] = r = pool.apply(_timing, ((g['dataset'], g['horizon'], m, a.data_dir, a.audit, 60),))
            print('timing', g['dataset'], g['horizon'], m, r, flush=True)
    rep['timing_single_process'] = tim
    conc = {}
    for ds, h in (('Traffic', 720), ('Electricity', 720)):
        with ctx.Pool(2) as pool:
            rr = pool.map(_timing, [(ds, h, 'rg', a.data_dir, a.audit, 100)] * 2)
        conc[f'{ds}/H{h}/rg x2'] = rr
        print('concurrent x2', ds, h, rr, flush=True)
    rep['two_workers_concurrent'] = conc
    rep['gpu'] = torch.cuda.get_device_name(0)
    rep['gpu_total_mib'] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 20)
    rep['SMOKE_OK'] = bool(ok)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print('SMOKE_OK' if ok else 'SMOKE_FAILED', flush=True)


if __name__ == '__main__':
    main()
