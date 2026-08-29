import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.quant import fake_quant, qmax_from_bits, check_quantizer_health, MODE_UNIFORM, MODE_DITHER, MODE_SR, MODE_PERCH, MODE_TERNARY, MODE_BINARY
from src.metrics import feature_dims, b_crit_from_curve, prefix_fraction


def _q(bits, mode=MODE_UNIFORM, K=1, m=8, n=64, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(K, m, n, generator=g, requires_grad=True)
    qmax = qmax_from_bits(torch.tensor([float(bits)] * K)).view(K, 1, 1)
    mo = torch.tensor([mode] * K).view(K, 1, 1)
    out, wq, s = fake_quant(w, qmax, mo, return_effective=True, **kw)
    return w, out, wq, s


def test_levels_are_2b_minus_1():
    for bits, expect in ((2, 3), (3, 7), (4, 15), (8, 255)):
        g = torch.Generator().manual_seed(0)
        w = torch.rand(1, 8, 8192, generator=g) * 2 - 1          # uniform: every level reachable
        qmax = qmax_from_bits(torch.tensor([float(bits)])).view(1, 1, 1)
        mo = torch.zeros(1, 1, 1, dtype=torch.long)
        _, wq, s = fake_quant(w, qmax, mo, return_effective=True)
        levels = torch.unique(wq).numel()
        assert levels == expect, (bits, levels)                  # 2^b - 1, NOT 2^b
        assert (wq.min() / s).round().item() >= -(2 ** (bits - 1) - 1)


def test_fp32_arm_identity():
    w, out, wq, s = _q(float("inf"))
    assert torch.equal(wq, w.detach())


def test_ste_gradient_is_identity():
    w, out, wq, s = _q(3)
    out.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_dead_columns_stay_exactly_zero():
    w = torch.randn(1, 8, 64); w[:, :, :10] = 0.0
    qmax = qmax_from_bits(torch.tensor([3.0])).view(1, 1, 1); mo = torch.zeros(1, 1, 1, dtype=torch.long)
    _, wq, _ = fake_quant(w, qmax, mo, return_effective=True)
    assert (wq[:, :, :10] == 0).all()


def test_residual_bounded_by_half_step():
    for mode in (MODE_UNIFORM, MODE_PERCH):
        w, out, wq, s = _q(4, mode=mode)
        assert ((wq - w).abs() <= 0.5 * s + 1e-6).all()
    w, out, wq, s = _q(4, mode=MODE_DITHER, dither_v=torch.rand(1, 8, 64))
    assert ((wq - w).abs() <= 0.5 * s + 1e-6).all()
    w, out, wq, s = _q(4, mode=MODE_SR, sr_u=torch.rand(1, 8, 64))
    assert ((wq - w).abs() <= s + 1e-6).all()


def test_dither_shifts_lattice_per_element():
    v = torch.rand(1, 8, 64)
    w, out, wq, s = _q(4, mode=MODE_DITHER, dither_v=v)
    frac = ((wq / s) - v).round() - ((wq / s) - v)
    assert frac.abs().max() < 1e-4


def test_health_check_catches_dead_arm():
    wq = torch.zeros(2, 4, 4); wq[1, 0, 0] = 1.0
    qmax = torch.tensor([1.0, 1.0]).view(2, 1, 1)
    assert check_quantizer_health(wq, qmax) == [0]


def test_binary_ternary_modes():
    w, out, wq, s = _q(1, mode=MODE_BINARY)
    assert torch.unique(wq).numel() == 2
    w, out, wq, s = _q(1.58, mode=MODE_TERNARY)
    assert torch.unique(wq).numel() == 3


def test_feature_dims_invariant():
    W = torch.randn(3, 10, 200)
    D = feature_dims(W)
    assert (D.sum(1) <= 10 + 1e-3).all()
    I = torch.eye(10).unsqueeze(0)
    assert torch.allclose(feature_dims(I), torch.ones(1, 10))


def test_bcrit_interpolation():
    bits = [2, 3, 4, 5, 6, 7, 8]
    assert b_crit_from_curve(bits, [10, 20, 40, 60, 62, 64, 64], 64)[0] == 5.0 + (0.95 * 64 - 60) / 2
    assert b_crit_from_curve(bits, [64] * 7, 64)[1] == "low"
    assert b_crit_from_curve(bits, [1] * 7, 64)[1] == "high"


def test_prefix_fraction():
    r = torch.tensor([1, 1, 1, 0, 1, 0, 0, 0], dtype=torch.bool)
    strict, overlap = prefix_fraction(r)
    assert abs(strict - 3 / 4) < 1e-9 and abs(overlap - 3 / 4) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
