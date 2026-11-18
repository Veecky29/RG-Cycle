"""CPU unit tests: shapes, exact equality with the backbone at initialisation, gate init, cycle indexing."""
import torch

from rgcycle import build_model
from rgcycle.cycle import RecurrentCycle, extra_parameters, reliability_features

CFG = dict(learning_rate=1e-3, d_model=64, kernel_size=5)


def _models(c=7, h=96, w=24):
    out = {}
    for m in ('orig', 'rcf', 'rg'):
        torch.manual_seed(0)
        out[m] = build_model(m, c, h, CFG, w, device='cpu').eval()
    return out


def test_shapes_and_init_equality():
    ms = _models()
    x, t0 = torch.randn(4, 96, 7), torch.tensor([0, 5, 23, 1000])
    with torch.no_grad():
        y = {m: ms[m](x, t0) for m in ms}
    assert y['orig'].shape == (4, 96, 7)
    assert torch.equal(y['orig'], y['rcf']) and torch.equal(y['orig'], y['rg'])
    sd = ms['orig'].state_dict()
    for m in ('rcf', 'rg'):
        assert all(torch.equal(sd[k], ms[m].state_dict()['backbone.' + k]) for k in sd)


def test_gate_init_and_params():
    ms = _models(c=7, h=720)
    assert extra_parameters(ms['rcf']) == dict(cycle=24 * 7, gate=0, total=168)
    assert extra_parameters(ms['rg'])['gate'] == 2 * 7 + 2 * 720 + 4
    r2, rho = torch.zeros(2, 7), torch.zeros(2, 7)
    gi, go = ms['rg'].gate(r2, rho)
    assert gi.shape == (2, 7) and go.shape == (2, 720, 7)
    assert torch.allclose(gi, torch.sigmoid(torch.tensor(-2.0)).expand_as(gi))


def test_cycle_indexing_matches_cyclenet():
    q = RecurrentCycle(24, 3)
    with torch.no_grad():
        q.data.copy_(torch.arange(24).float().view(24, 1).repeat(1, 3))
    out = q(torch.tensor([22]), 5)[0, :, 0]
    assert out.tolist() == [22, 23, 0, 1, 2]


def test_reliability_features():
    t = torch.arange(96).float()
    x = torch.sin(2 * torch.pi * t / 24).view(1, 96, 1)
    x = (x - x.mean(1, keepdim=True)) / x.std(1, unbiased=False, keepdim=True)
    r2, rho = reliability_features(x, x.clone())
    assert r2.item() > 0.99 and rho.item() > 0.99
    r2, rho = reliability_features(x, torch.zeros_like(x))
    assert abs(r2.item()) < 1e-4 and rho.item() == 0.0
