"""iTransformer-style channel-token backbone (identical to the PAC-Former ablation variant "B").

The module creation order matches the PAC-Former code exactly. With ``rng_compat_kernel`` set,
the throw-away PLM convolution and statistics MLP of the full PAC-Former model are created (and
discarded) at the same position, so that under the same seed every backbone parameter is
bit-identical to the existing PAC-Former "B" runs. This only affects initial random draws.
"""
import math

import torch
from torch import nn


class RevIN(nn.Module):
    """Instance normalisation without affine parameters (Kim et al., 2022)."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev
        if mode == 'denorm':
            return x * self.stdev + self.mean
        raise ValueError(mode)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        b, c, d = h.shape
        q = self.q_proj(h).view(b, c, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, c, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, c, self.n_heads, self.head_dim).transpose(1, 2)
        w = self.dropout(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1))
        return self.out_proj(torch.matmul(w, v).transpose(1, 2).contiguous().view(b, c, d))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads=8, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = Attention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ffn(x))


class ITransformer(nn.Module):
    """Variates as tokens: (B, L, C) -> (B, H, C).

    ``forward_normalized`` maps an instance-normalised window to a normalised forecast, which is the
    interface the cycle wrappers use. ``forward`` adds RevIN around it.
    """

    def __init__(self, c_in, seq_len=96, pred_len=96, d_model=128, n_heads=8, e_layers=2, d_ff=None,
                 dropout=0.1, rng_compat_kernel=None):
        super().__init__()
        self.c_in, self.seq_len, self.pred_len = c_in, seq_len, pred_len
        self.revin = RevIN(c_in)
        if rng_compat_kernel is not None:
            # Same random draws as PAC-Former's PLM conv and statistics MLP; modules are discarded.
            k = int(rng_compat_kernel)
            nn.Conv1d(c_in, c_in, kernel_size=k, padding=(k - 1) // 2, groups=c_in, bias=True)
            nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 16))
        self.enc_embedding = nn.Linear(seq_len, d_model)
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff or 2 * d_model, dropout)
                                     for _ in range(e_layers)])
        self.projector = nn.Linear(d_model, pred_len)

    def forward_normalized(self, xn):
        h = self.enc_embedding(xn.permute(0, 2, 1))
        for layer in self.layers:
            h = layer(h)
        return self.projector(h).permute(0, 2, 1)

    def forward(self, x, index=None):
        return self.revin(self.forward_normalized(self.revin(x, 'norm')), 'denorm')
