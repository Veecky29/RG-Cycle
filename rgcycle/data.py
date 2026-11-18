"""Chronological 70/10/20 (ETT: 12/4/4 months) splits, train-only StandardScaler, GPU-resident windows.

Identical windowing to the PAC-Former training code: a window starting at absolute row t0 uses rows
[t0, t0+L) as input and [t0+L, t0+L+H) as target. Validation/test windows keep the L preceding rows,
and every window is evaluated (no drop_last).
"""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

SEQ_LEN = 96


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Series:
    def __init__(self, name, audit, data_dir, seq_len=SEQ_LEN, device='cuda'):
        self.name, self.audit, self.seq_len = name, audit['datasets'][name], seq_len
        p = Path(data_dir) / (name + '.csv')
        assert sha(p) == self.audit['sha256'], f'SHA-256 mismatch for {p}'
        frame = pd.read_csv(p)
        self.columns = self.audit['columns']
        raw = frame[self.columns].to_numpy(dtype=np.float64)
        s = self.audit['splits_steps']
        self.ntrain, self.nval, self.ntest = s['train'], s['val'], s['test']
        self.used = self.ntrain + self.nval + self.ntest
        self.channels = raw.shape[1]
        self.scaler = StandardScaler().fit(raw[:self.ntrain])
        scaled = self.scaler.transform(raw[:self.used]).astype(np.float32)
        assert np.isfinite(scaled).all()
        self.gpu = torch.from_numpy(scaled).to(device)
        self.xoffset = torch.arange(seq_len, device=device)
        self._yoff = {}
        self.device = device

    def bounds(self, h, split):
        L = self.seq_len
        if split == 'train':
            start, end = 0, self.ntrain
        elif split == 'val':
            start, end = self.ntrain - L, self.ntrain + self.nval
        elif split == 'test':
            start, end = self.ntrain + self.nval - L, self.used
        else:
            raise ValueError(split)
        return start, end - L - h + 1

    def batch(self, starts, h):
        """starts: absolute row indices t0 (LongTensor). Returns x (B,L,C), y (B,H,C), t0 on device."""
        if h not in self._yoff:
            self._yoff[h] = torch.arange(h, device=self.device) + self.seq_len
        idx = starts.to(self.device)
        return self.gpu[idx[:, None] + self.xoffset[None, :]], self.gpu[idx[:, None] + self._yoff[h][None, :]], idx

    def close(self):
        del self.gpu
        self._yoff.clear()
        gc.collect()
        torch.cuda.empty_cache()


def load_audit(path):
    return json.loads(Path(path).read_text())
