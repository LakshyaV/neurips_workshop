"""Fake quantizers and post-hoc damage operators for batched TMS weights W:[K,m,n].

Every per-model parameter (qmax, mode) is a tensor so that a single group of K models
with different bit-widths / modes trains in one batched forward.

mode codes (integer dispatch — never compare floats):
  0 = uniform symmetric, per-tensor max-abs scale      (primary)
  1 = binary  (sign * mean|w|)
  2 = ternary (round(w/mean|w|) clipped to ±1)
  3 = dithered uniform: Q_u(w) = Δ·(round(w/Δ − v) + v), v fixed per element in [0,1)
  4 = stochastic rounding: Δ·floor(w/Δ + u), u ~ U[0,1) fresh per call (or fixed at eval)
  5 = uniform symmetric, per-channel (per input column j) max-abs scale
"""
import torch

MODE_UNIFORM, MODE_BINARY, MODE_TERNARY, MODE_DITHER, MODE_SR, MODE_PERCH = 0, 1, 2, 3, 4, 5
MODE_NAMES = {0: "uniform", 1: "binary", 2: "ternary", 3: "dither", 4: "stochastic", 5: "perchannel"}


def qmax_from_bits(bits):
    """bits: float tensor; inf -> inf (fp32 arm). Only used for mode 0/3/4/5."""
    return torch.where(torch.isinf(bits), bits, 2.0 ** (bits - 1) - 1)


def fake_quant(w, qmax, mode, dither_v=None, sr_u=None, return_effective=False):
    """w:[K,m,n] fp32. qmax:[K,1,1] float (inf = fp32 arm). mode:[K,1,1] int64.
    dither_v: [K,m,n] in [0,1) (mode 3). sr_u: [K,m,n] in [0,1) (mode 4).
    Returns STE tensor  w + (wq - w).detach()  (and wq itself if return_effective)."""
    absw = w.abs()
    amax = absw.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8).detach()
    amean = absw.mean(dim=(1, 2), keepdim=True).clamp_min(1e-8).detach()
    isfp = torch.isinf(qmax)
    qm = torch.where(isfp, torch.ones_like(qmax), qmax)          # keep arithmetic finite
    s = (amax / qm).clamp_min(1e-12)                              # Δ per tensor

    wq0 = torch.clamp(torch.round(w / s), -qm, qm) * s
    wq1 = torch.sign(w) * amean
    wq2 = torch.clamp(torch.round(w / amean), -1.0, 1.0) * amean
    v = dither_v if dither_v is not None else torch.zeros_like(w)
    wq3 = (torch.round(w / s - v) + v) * s
    u = sr_u if sr_u is not None else torch.full_like(w, 0.5)
    wq4 = torch.floor(w / s + u) * s
    amax_c = absw.amax(dim=1, keepdim=True).clamp_min(1e-8).detach()   # [K,1,n]
    s_c = (amax_c / qm).clamp_min(1e-12)
    wq5 = torch.clamp(torch.round(w / s_c), -qm, qm) * s_c

    wq = torch.where(mode == 1, wq1,
         torch.where(mode == 2, wq2,
         torch.where(mode == 3, wq3,
         torch.where(mode == 4, wq4,
         torch.where(mode == 5, wq5, wq0)))))
    wq = torch.where(isfp, w, wq)
    out = w + (wq - w).detach()
    if return_effective:
        return out, wq.detach(), s.detach()
    return out


def check_quantizer_health(wq, qmax):
    """Assert every non-fp32 model has >1 distinct effective value (guards the silent-dead-arm bug)."""
    K = wq.shape[0]
    bad = []
    for k in range(K):
        if torch.isinf(qmax[k]).item():
            continue
        nuniq = torch.unique(wq[k].reshape(-1)).numel()
        if nuniq <= 1:
            bad.append(k)
    return bad


# ---------------------------------------------------------------- post-hoc damage (E3)
def damage_gaussian(W, sigma_rel, gen):
    """W + sigma_rel * rms(W) * z, z ~ N(0,1) (fixed draw from gen). W:[K,m,n]."""
    rms = W.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
    z = torch.randn(W.shape, generator=gen, device="cpu").to(W.device)
    return W + sigma_rel * rms * z


def damage_rank_truncate(W, s):
    """Keep the top-s singular directions of each W_k (SVD on CPU)."""
    Wc = W.detach().cpu()
    U, S, Vh = torch.linalg.svd(Wc, full_matrices=False)
    S = S.clone()
    S[:, s:] = 0.0
    return (U * S[:, None, :]) @ Vh


def damage_column_dropout(W, d, gen):
    """Zero a random fraction d of the n columns (features) of each model."""
    K, m, n = W.shape
    keep = (torch.rand(K, 1, n, generator=gen, device="cpu") >= d).to(W.device, W.dtype)
    return W * keep
