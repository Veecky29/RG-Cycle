"""Two training recipes, selected by cfg['recipe'].

'cyclenet' (CycleNet official exp/exp_main.py + utils/tools.py + data_provider/data_factory.py @ d807e51):
  Adam(lr), MSE loss, batch cfg['batch_size'] (train: shuffle, drop_last=True), at most cfg['train_epochs'] epochs,
  EarlyStopping(patience=cfg['patience']) on the validation loss, best-validation checkpoint.
  Learning rate per epoch (lradj='type3', the run.py default; the scripts do not set lradj): exp_main builds a
  OneCycleLR(max_lr=lr) that is never stepped for type3, but its constructor sets the optimiser lr to lr/25,
  so epoch 1 runs at lr/25; after epoch e, adjust_learning_rate sets lr if e < 3 else lr*0.8**(e-3).
  Validation loss = mean over batches of the batch MSE, val loader with shuffle=True and drop_last=True
  (CycleNet data_factory: every flag except test/pred). Test: all windows (drop_last=False), element-wise mean.

'pacformer' (the PAC-Former ablation recipe, kept for the bit-exact check against variant B):

Adam (betas .9/.999, eps 1e-8, no weight decay), batch 32 (last partial training batch dropped, as before),
at most 50 epochs, early stopping patience 5 on full-validation MSE, best-validation checkpoint,
per-step schedule: 1-epoch linear warm-up from 0.1x, then cosine to 0.01x over 50 planned epochs.
FP32, TF32 off for matmul, deterministic algorithms. Training order from a seed-reset CPU generator.

Only addition: optional per-parameter-group lr multipliers (used for the RG gate parameters).
With a single group the optimiser is identical to Adam(model.parameters()).
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import gc
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .cycle import CycleForecaster, build_model, extra_parameters

BATCH, MAX_EPOCHS, PATIENCE = 32, 50, 5


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def atomic_model(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def lr_at(step, steps, peak):
    if step < steps:
        return peak * (0.1 + 0.9 * step / max(1, steps - 1))
    progress = (step - steps) / max(1, 50 * steps - steps - 1)
    return peak * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def cyclenet_epoch_lr(epoch, peak):
    """lr used during (1-based) epoch under CycleNet's lradj='type3' + unstepped OneCycleLR(div_factor=25)."""
    if epoch == 1:
        return peak / 25.0
    e = epoch - 1  # adjust_learning_rate(epoch=e) ran after epoch e
    return peak if e < 3 else peak * (0.8 ** (e - 3))


def cyclenet_val_loss(model, series, h, batch_size, generator):
    """CycleNet vali(): shuffled val loader with drop_last=True, np.average of per-batch MSE."""
    model.eval()
    a, b = series.bounds(h, 'val')
    n = b - a
    order = torch.randperm(n, generator=generator) + a
    losses = []
    with torch.no_grad():
        for i in range(n // batch_size):
            x, y, t0 = series.batch(order[i * batch_size:(i + 1) * batch_size], h)
            losses.append(torch.nn.functional.mse_loss(model(x, t0), y).double())
    return float(torch.stack(losses).mean().item())


def make_optimizer(model, lr, gate_lr_mult=1.0):
    gate = [p for n, p in model.named_parameters() if n.startswith('gate.')]
    ids = {id(p) for p in gate}
    rest = [p for p in model.parameters() if id(p) not in ids]
    groups = [dict(params=rest, lr=lr, lr_mult=1.0)]
    if gate:
        groups.append(dict(params=gate, lr=lr * gate_lr_mult, lr_mult=gate_lr_mult))
    return torch.optim.Adam(groups, lr=lr)


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1.0)


class FullErrors:
    def __init__(self):
        self.sse = torch.zeros((), dtype=torch.float64, device='cuda')
        self.sae = torch.zeros((), dtype=torch.float64, device='cuda')
        self.elements = self.windows = 0

    def add(self, pred, true):
        assert pred.shape == true.shape
        e = pred.double() - true.double()
        self.sse += e.square().sum()
        self.sae += e.abs().sum()
        self.elements += true.numel()
        self.windows += true.shape[0]

    def result(self):
        mse = float((self.sse / self.elements).item())
        mae = float((self.sae / self.elements).item())
        if not np.isfinite([mse, mae]).all():
            raise FloatingPointError('non-finite metric')
        return dict(mse=mse, mae=mae, windows=self.windows, elements=self.elements)


def evaluate(model, series, h, split, batch_size=32):
    model.eval()
    a, b = series.bounds(h, split)
    acc = FullErrors()
    with torch.no_grad():
        for start in range(a, b, batch_size):
            x, y, t0 = series.batch(torch.arange(start, min(b, start + batch_size), dtype=torch.long), h)
            acc.add(model(x, t0), y)
    out = acc.result()
    assert out['windows'] == b - a and out['elements'] == (b - a) * h * series.channels
    return out


def _pearson(s):
    n, sx, sy, sxx, syy, sxy = s
    cov = sxy / n - (sx / n) * (sy / n)
    vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
    return float(cov / math.sqrt(vx * vy)) if vx > 1e-12 and vy > 1e-12 else None


def gate_stats(model, series, h, split='test', buckets=4):
    """Gate/feature statistics of a trained RCF/RG model on one split."""
    if not isinstance(model, CycleForecaster):
        return None
    q = model.cycle.data.detach()
    out = dict(Q_rms=float(q.square().mean().sqrt()), Q_absmax=float(q.abs().max()))
    if model.gate is None:
        return out
    model.eval()
    model.record = []
    a, b = series.bounds(h, split)
    edges = [round(i * h / buckets) for i in range(buckets + 1)]
    tot = dict(n=0, r2=0.0, rho=0.0, gin=0.0, gout=0.0, gout_b=[0.0] * buckets)
    c_in = [0.0] * 6
    c_out = [0.0] * 6
    with torch.no_grad():
        for start in range(a, b, 32):
            x, _, t0 = series.batch(torch.arange(start, min(b, start + 32), dtype=torch.long), h)
            model(x, t0)
            r2, rho, gi, go = model.record.pop()
            r2, rho, gi, go = r2.double(), rho.double(), gi.double(), go.double()
            gom = go.mean(dim=1)  # (B, C) mean over horizon
            n = r2.numel()
            tot['n'] += n
            tot['r2'] += float(r2.sum())
            tot['rho'] += float(rho.sum())
            tot['gin'] += float(gi.sum())
            tot['gout'] += float(gom.sum())
            for i in range(buckets):
                tot['gout_b'][i] += float(go[:, edges[i]:edges[i + 1]].mean(dim=1).sum())
            for acc, g in ((c_in, gi), (c_out, gom)):
                for j, v in enumerate((n, r2.sum(), g.sum(), (r2 * r2).sum(), (g * g).sum(), (r2 * g).sum())):
                    acc[j] += float(v)
    model.record = None
    n = tot['n']
    out.update(split=split, pairs=n, mean_R2=tot['r2'] / n, mean_rho=tot['rho'] / n,
               mean_g_in=tot['gin'] / n, mean_g_out=tot['gout'] / n,
               mean_g_out_by_horizon_bucket={f'{edges[i] + 1}-{edges[i + 1]}': tot['gout_b'][i] / n for i in range(buckets)},
               corr_g_in_R2=_pearson(c_in), corr_g_out_R2=_pearson(c_out))
    st = model.gate
    out['gate_params'] = dict(a_in=st.a_in.tolist(), a_out=st.a_out.tolist(),
                              eta_first_last=[float(st.eta[0]), float(st.eta[-1])],
                              kappa_first_last=[float(st.kappa[0]), float(st.kappa[-1])])
    return out


def fit_job(series, h, cfg, seed, path, mode, cycle_len=None, gate_init_bias=-2.0, gate_lr_mult=10.0,
            log=print, status_cb=lambda _: None, max_epochs=MAX_EPOCHS, meta=None):
    """Train one (dataset, H, mode, seed) job; resumable from last.pt; writes result.json + best.pt."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rp = path / 'result.json'
    if rp.exists():
        r = json.loads(rp.read_text())
        if r.get('status') == 'completed':
            return r
    seed_all(seed)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    model = build_model(mode, series.channels, h, cfg, cycle_len, gate_init_bias)
    optimizer = make_optimizer(model, cfg['learning_rate'], gate_lr_mult if mode == 'rg' else 1.0)
    cyclenet = cfg.get('recipe') == 'cyclenet'
    bs = cfg['batch_size'] if cyclenet else BATCH
    if cyclenet:
        max_epochs = min(max_epochs, cfg['train_epochs'])
        patience_limit = cfg['patience']
        val_generator = torch.Generator(device='cpu').manual_seed(seed + 1_000_003)
    else:
        patience_limit = PATIENCE
    a, b = series.bounds(h, 'train')
    n = b - a
    steps = n // bs
    best, best_epoch, patience, epochs, start_epoch = float('inf'), 0, 0, [], 1
    ckpt = path / 'best.pt'
    torch.cuda.reset_peak_memory_stats()
    if (path / 'last.pt').exists():
        s = torch.load(path / 'last.pt', map_location='cpu', weights_only=False)
        assert s['config'] == cfg and s['seed'] == seed and s['mode'] == mode
        model.load_state_dict(s['model_state'])
        optimizer.load_state_dict(s['optimizer_state'])
        generator.set_state(s['shuffle_rng'])
        if cyclenet:
            val_generator.set_state(s['val_rng'])
        torch.set_rng_state(s['torch_rng'])
        torch.cuda.set_rng_state_all(s['cuda_rng'])
        np.random.set_state(s['numpy_rng'])
        random.setstate(s['python_rng'])
        best, best_epoch, patience, epochs = s['best'], s['best_epoch'], s['patience'], s['epochs']
        start_epoch = s['epoch'] + 1
        del s
    if patience >= patience_limit:
        start_epoch = max_epochs + 1
    job = dict(dataset=series.name, horizon=h, mode=mode, seed=seed, config=cfg, cycle_len=cycle_len,
               gate_init_bias=gate_init_bias if mode == 'rg' else None,
               gate_lr_mult=gate_lr_mult if mode == 'rg' else None, **(meta or {}))
    with (path / 'epochs.jsonl').open('w', encoding='utf8', buffering=1) as elog:
        for e in epochs:
            elog.write(json.dumps(e) + '\n')
        for epoch in range(start_epoch, max_epochs + 1):
            t = time.perf_counter()
            model.train()
            order = torch.randperm(n, generator=generator) + a
            tsum = torch.zeros((), dtype=torch.float64, device='cuda')
            if cyclenet:
                lr = cyclenet_epoch_lr(epoch, cfg['learning_rate'])
                set_lr(optimizer, lr)
            for i in range(steps):
                if not cyclenet:
                    lr = lr_at((epoch - 1) * steps + i, steps, cfg['learning_rate'])
                    set_lr(optimizer, lr)
                x, y, t0 = series.batch(order[i * bs:(i + 1) * bs], h)
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(x, t0), y)
                loss.backward()
                optimizer.step()
                tsum += loss.detach().double()
            train_mse = float((tsum / steps).item())
            if not np.isfinite(train_mse):
                raise FloatingPointError(f'non-finite training loss {job}')
            if cyclenet:
                val = dict(mse=cyclenet_val_loss(model, series, h, bs, val_generator), mae=None)
            else:
                val = evaluate(model, series, h, 'val')
            improved = val['mse'] < best
            if improved:
                best, best_epoch, patience = val['mse'], epoch, 0
                atomic_model(ckpt, dict(model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                        job=job, best_epoch=best_epoch, best_val_mse=best))
            else:
                patience += 1
            rec = dict(epoch=epoch, lr_last=lr, lr_gate=lr * gate_lr_mult if mode == 'rg' else None, train_mse=train_mse, val_mse=val['mse'], val_mae=val['mae'],
                       improved=improved, seconds=time.perf_counter() - t, utc=utc())
            epochs.append(rec)
            elog.write(json.dumps(rec) + '\n')
            atomic_model(path / 'last.pt', dict(config=cfg, seed=seed, mode=mode, epoch=epoch,
                                                model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                                optimizer_state=optimizer.state_dict(), shuffle_rng=generator.get_state(),
                                                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                                                numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                                                val_rng=val_generator.get_state() if cyclenet else None,
                                                best=best, best_epoch=best_epoch, patience=patience, epochs=epochs))
            status_cb(dict(epoch=epoch, val_mse=val['mse'], best_val_mse=best, utc=utc()))
            log(f"{series.name} H{h} {mode} seed{seed} epoch {epoch} train {train_mse:.6f} lr {lr:.3g} val {val['mse']:.9f} "
                f"best {best:.9f} {rec['seconds']:.1f}s")
            if patience >= patience_limit:
                break
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True)['model_state'], strict=True)
    val = evaluate(model, series, h, 'val')  # full validation set (reported only)
    if not cyclenet:
        assert val['mse'] == best
    test = evaluate(model, series, h, 'test')
    result = dict(status='completed', **job, best_val_mse=best, best_epoch=best_epoch, epochs_ran=len(epochs),
                  val=val, test=test, parameters=sum(p.numel() for p in model.parameters()),
                  extra_parameters=extra_parameters(model), gates=gate_stats(model, series, h),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                  elapsed_seconds=sum(e['seconds'] for e in epochs), ended_utc=utc())
    atomic_json(rp, result)
    (path / 'last.pt').unlink(missing_ok=True)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result
