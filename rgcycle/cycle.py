"""RCF (CycleNet, Lin et al., NeurIPS 2024) and RG-Cycle wrappers around any normalised-space backbone.

Notation (all in the instance-normalised space, as in CycleNet / CycleiTransformer):
    x_n  : normalised lookback window, (B, L, C)
    Q    : learnable cycle table, (W, C), initialised to zeros
    k    : cycle phase of the window = (t0 + L) mod W, where t0 is the absolute row index of the first
           lookback step. This is the official CycleNet convention (``cycle_index[s_end]``); it is a fixed
           cyclic relabelling of the rows of Q and therefore equivalent to indexing from t0.
    c_in [i] = Q[(k + i) mod W],      i = 0..L-1
    c_out[j] = Q[(k + L + j) mod W],  j = 0..H-1

    orig : y_n = f(x_n)
    rcf  : y_n = f(x_n - c_in) + c_out
    rg   : y_n = f(x_n - g_in * c_in) + g_out * c_out,  g_in in [0,1]^{B x C}, g_out in [0,1]^{B x H x C}

Because Q starts at zero, both wrappers reproduce the backbone output exactly at initialisation,
for any gate value.
"""
import torch
from torch import nn

from .backbones import ITransformer

MODES = ('orig', 'rcf', 'rg')


class RecurrentCycle(nn.Module):
    """Learnable recurrent cycle Q (W x C); identical to CycleNet's RecurrentCycle."""

    def __init__(self, cycle_len, channels):
        super().__init__()
        self.cycle_len = cycle_len
        self.data = nn.Parameter(torch.zeros(cycle_len, channels))

    def forward(self, index, length):
        gather = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather]


def reliability_features(xn, c_in, eps=1e-5):
    """Per sample/channel reliability of the cycle prior on the lookback window.

    R2  = 1 - ||(x - mean x) - c_in||^2 / ||x - mean x||^2, clamped to [-1, 1]
    rho = Pearson correlation between x and c_in over the window (0 when c_in is constant).
    c_in is detached by the caller, so Q is trained only through the forecasting loss.
    Returns two (B, C) tensors.
    """
    xc = xn - xn.mean(dim=1, keepdim=True)
    cc = c_in - c_in.mean(dim=1, keepdim=True)
    den = xc.square().sum(dim=1) + eps
    r2 = (1 - (xc - c_in).square().sum(dim=1) / den).clamp(-1.0, 1.0)
    rho = (xc * cc).sum(dim=1) / torch.sqrt(den * (cc.square().sum(dim=1) + eps))
    return r2, rho


class ReliabilityGate(nn.Module):
    """Tiny gate head.

    g_in [b, c]    = sigmoid(b0 + beta_in[c] + a_in . [R2, rho][b, c])
    g_out[b, h, c] = sigmoid(b0 + beta_out[c] + eta[h] + (a_out0 + kappa[h]) * R2[b, c] + a_out1 * rho[b, c])

    All learnable parameters start at zero; b0 is a fixed buffer (default -2, i.e. g = 0.119 at init).
    Parameters: 2C + 2H + 4.
    """

    def __init__(self, channels, pred_len, init_bias=-2.0):
        super().__init__()
        self.register_buffer('b0', torch.tensor(float(init_bias)))
        self.beta_in = nn.Parameter(torch.zeros(channels))
        self.a_in = nn.Parameter(torch.zeros(2))
        self.beta_out = nn.Parameter(torch.zeros(channels))
        self.eta = nn.Parameter(torch.zeros(pred_len))
        self.kappa = nn.Parameter(torch.zeros(pred_len))
        self.a_out = nn.Parameter(torch.zeros(2))

    def forward(self, r2, rho):
        g_in = torch.sigmoid(self.b0 + self.beta_in + self.a_in[0] * r2 + self.a_in[1] * rho)
        slope = (self.a_out[0] + self.kappa).view(1, -1, 1)
        pre = (self.b0 + self.beta_out.view(1, 1, -1) + self.eta.view(1, -1, 1)
               + slope * r2.unsqueeze(1) + self.a_out[1] * rho.unsqueeze(1))
        return g_in, torch.sigmoid(pre)


class CycleForecaster(nn.Module):
    """Plug-in wrapper: any backbone with ``revin`` (norm/denorm) and ``forward_normalized``.

    forward(x, index): x is the raw (dataset-standardised) window (B, L, C); index (B,) is the absolute
    row index t0 of the first lookback step.
    """

    def __init__(self, backbone, channels, seq_len, pred_len, cycle_len, mode='rcf', gate_init_bias=-2.0):
        super().__init__()
        assert mode in ('rcf', 'rg'), mode
        self.backbone, self.mode = backbone, mode
        self.seq_len, self.pred_len, self.cycle_len = seq_len, pred_len, cycle_len
        self.cycle = RecurrentCycle(cycle_len, channels)
        self.gate = ReliabilityGate(channels, pred_len, gate_init_bias) if mode == 'rg' else None
        self.record = None  # set to a list to collect (r2, rho, g_in, g_out) during evaluation

    def forward(self, x, index):
        rev = self.backbone.revin
        xn = rev(x, 'norm')
        k = (index.long() + self.seq_len) % self.cycle_len
        c_in = self.cycle(k, self.seq_len)
        c_out = self.cycle((k + self.seq_len) % self.cycle_len, self.pred_len)
        if self.mode == 'rcf':
            y = self.backbone.forward_normalized(xn - c_in) + c_out
        else:
            r2, rho = reliability_features(xn, c_in.detach())
            g_in, g_out = self.gate(r2, rho)
            y = self.backbone.forward_normalized(xn - g_in.unsqueeze(1) * c_in) + g_out * c_out
            if self.record is not None:
                self.record.append((r2.detach(), rho.detach(), g_in.detach(), g_out.detach()))
        return rev(y, 'denorm')


def build_model(mode, channels, horizon, cfg, cycle_len=None, gate_init_bias=-2.0, seq_len=96, device='cuda'):
    """Backbone first (same random draws as PAC-Former variant B), then zero/constant-initialised extras."""
    assert mode in MODES, mode
    backbone = ITransformer(channels, seq_len, horizon, d_model=cfg['d_model'], n_heads=8, e_layers=2,
                            d_ff=2 * cfg['d_model'], dropout=0.1, rng_compat_kernel=cfg.get('kernel_size'))
    model = backbone if mode == 'orig' else CycleForecaster(backbone, channels, seq_len, horizon, cycle_len,
                                                            mode, gate_init_bias)
    return model.to(device)


def extra_parameters(model):
    """Parameters added on top of the backbone, by component."""
    if not isinstance(model, CycleForecaster):
        return dict(cycle=0, gate=0, total=0)
    cyc = sum(p.numel() for p in model.cycle.parameters())
    gate = sum(p.numel() for p in model.gate.parameters()) if model.gate is not None else 0
    return dict(cycle=cyc, gate=gate, total=cyc + gate)
