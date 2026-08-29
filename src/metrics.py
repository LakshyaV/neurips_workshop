"""All metrics defined once. Evaluate with the quantizer active on fresh samples."""
import math
import torch
from .model import BatchedTMS


def nu0(p):
    return p / 3.0 - p * p / 4.0


def sample_x(shape_KBn, p_K11, device, gen=None):
    """x[k,b,i] = Bernoulli(p_k) * U[0,1]."""
    if gen is None:
        r1 = torch.rand(shape_KBn, device=device)
        r2 = torch.rand(shape_KBn, device=device)
    else:
        r1 = torch.rand(shape_KBn, generator=gen, device=gen.device).to(device)
        r2 = torch.rand(shape_KBn, generator=gen, device=gen.device).to(device)
    return (r1 < p_K11).to(r2.dtype) * r2


@torch.no_grad()
def eval_rho(Wq, b, p_K, n_samples, chunk=1024, device=None, seed=12345):
    """Wq:[K,m,n] effective weights, b:[K,n], p_K:[K]. Returns rho:[K,n], nu:[K,n] (Monte Carlo)."""
    device = device or Wq.device
    K, m, n = Wq.shape
    p3 = p_K.view(K, 1, 1).to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sse = torch.zeros(K, n, device=device)
    done = 0
    while done < n_samples:
        B = min(chunk, n_samples - done)
        # generate on CPU with explicit generator (reproducible eval stream), move once
        r1 = torch.rand((K, B, n), generator=gen)
        r2 = torch.rand((K, B, n), generator=gen)
        x = ((r1 < p_K.view(K, 1, 1)).to(r2.dtype) * r2).to(device)
        xh = BatchedTMS.forward_with(Wq, b, x)
        sse += ((x - xh) ** 2).sum(dim=1)
        done += B
    nu = sse / n_samples
    rho = 1.0 - nu / nu0(p_K.view(K, 1).to(device))
    return rho, nu


def n_ret(rho, thr=0.5):
    return (rho >= thr).sum(dim=-1)


@torch.no_grad()
def feature_dims(W):
    """D_i = ||W_i||^4 / sum_j (W_i . W_j)^2  for W:[K,m,n] (columns = features). Sum_i D_i <= m."""
    G = torch.einsum("kmi,kmj->kij", W, W)            # Gram [K,n,n]
    norm2 = torch.diagonal(G, dim1=1, dim2=2)         # [K,n]
    denom = (G ** 2).sum(dim=2).clamp_min(1e-30)
    D = norm2 ** 2 / denom
    D = torch.where(norm2 > 1e-12, D, torch.zeros_like(D))
    return D


@torch.no_grad()
def kappa(W):
    """max|W| / rms(W) per model, W:[K,m,n]."""
    amax = W.abs().amax(dim=(1, 2))
    rms = W.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-12)
    return amax / rms


def prefix_fraction(retained_mask):
    """retained_mask:[n] bool, features already in importance order (index 0 = most important).
    strict: length of the all-retained prefix / #retained.
    overlap: |retained ∩ top-|retained| by importance| / |retained|."""
    r = retained_mask
    R = int(r.sum())
    if R == 0:
        return float("nan"), float("nan")
    nz = torch.nonzero(~r)
    L = int(nz[0]) if nz.numel() > 0 else int(r.numel())
    strict = min(L, R) / R
    overlap = int(r[:R].sum()) / R
    return strict, overlap


def b_crit_from_curve(bits, nret, nret_fp32, frac=0.95, fit_min_bits=3.0, fit_max_bits=8.0):
    """bits: sorted finite list; nret: same length. Returns (b_crit, censor) with censor in
    {'ok','low','high'}. Linear interpolation on the first upward crossing of frac*nret_fp32,
    scanning from low bits. Crossings below fit_min_bits are 'low' (censored), never reached -> 'high'."""
    target = frac * nret_fp32
    pairs = [(b, v) for b, v in zip(bits, nret) if math.isfinite(b)]
    pairs.sort()
    # censored low: already above target at the smallest fitted bit-width
    fitted = [(b, v) for b, v in pairs if fit_min_bits <= b <= fit_max_bits]
    if len(fitted) == 0:
        return float("nan"), "none"
    if fitted[0][1] >= target:
        return fitted[0][0], "low"
    for (b0, v0), (b1, v1) in zip(fitted[:-1], fitted[1:]):
        if v0 < target <= v1:
            if v1 == v0:
                return b1, "ok"
            return b0 + (target - v0) / (v1 - v0) * (b1 - b0), "ok"
    return fitted[-1][0], "high"
