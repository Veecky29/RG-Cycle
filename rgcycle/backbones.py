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


# ---------------------------------------------------------------------------------------------------------
# iTransformer exactly as used in the CycleNet repository (models/CycleiTransformer.py with layers/Embed.py,
# layers/Transformer_EncDec.py, layers/SelfAttention_Family.py; github.com/ACAT-SCUT/CycleNet @ d807e51,
# Apache-2.0, see THIRD_PARTY_NOTICES.md). Same modules, creation order (hence same initial random draws),
# normalisation arithmetic and forward pass; x_mark_enc is None as in CycleNet's exp_main (no timestamp tokens).
# The RecurrentCycle is not part of this class: it is added by rgcycle.cycle.CycleForecaster.
# ---------------------------------------------------------------------------------------------------------
class CycleNetNorm(nn.Module):
    """Normalisation from Non-stationary Transformer, as written in CycleiTransformer.forecast."""

    def forward(self, x, mode):
        if mode == 'norm':
            self.means = x.mean(1, keepdim=True).detach()
            x = x - self.means
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            return x / self.stdev
        if mode == 'denorm':
            h = x.shape[1]
            x = x * (self.stdev[:, 0, :].unsqueeze(1).repeat(1, h, 1))
            return x + (self.means[:, 0, :].unsqueeze(1).repeat(1, h, 1))
        raise ValueError(mode)


class _Identity(nn.Module):
    def forward(self, x, mode):
        return x


class FullAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = 1. / math.sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum('bhls,bshd->blhd', A, values).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        q = self.query_projection(x).view(B, L, H, -1)
        k = self.key_projection(x).view(B, L, H, -1)
        v = self.value_projection(x).view(B, L, H, -1)
        return self.out_projection(self.inner_attention(q, k, v).view(B, L, -1))


class CNEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = torch.nn.functional.relu if activation == 'relu' else torch.nn.functional.gelu

    def forward(self, x):
        x = x + self.dropout(self.attention(x))
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class CycleNetITransformer(nn.Module):
    """CycleNet's iTransformer (CycleiTransformer without the cycle). (B, L, C) -> (B, H, C)."""

    def __init__(self, c_in, seq_len=96, pred_len=96, d_model=512, n_heads=8, e_layers=2, d_ff=2048,
                 dropout=0.1, activation='gelu', use_revin=True):
        super().__init__()
        self.c_in, self.seq_len, self.pred_len = c_in, seq_len, pred_len
        # DataEmbedding_inverted
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.emb_dropout = nn.Dropout(p=dropout)
        self.attn_layers = nn.ModuleList([
            CNEncoderLayer(AttentionLayer(FullAttention(attention_dropout=dropout), d_model, n_heads),
                           d_model, d_ff, dropout=dropout, activation=activation) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, pred_len, bias=True)
        self.revin = CycleNetNorm() if use_revin else _Identity()

    def forward_normalized(self, xn):
        n = xn.shape[2]
        h = self.emb_dropout(self.value_embedding(xn.permute(0, 2, 1)))
        for layer in self.attn_layers:
            h = layer(h)
        h = self.norm(h)
        return self.projector(h).permute(0, 2, 1)[:, :, :n]

    def forward(self, x, index=None):
        return self.revin(self.forward_normalized(self.revin(x, 'norm')), 'denorm')
