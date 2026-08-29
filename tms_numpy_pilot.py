"""Minimal numpy reference TMS (single model, manual gradients, scalar bits) used only to
cross-check the batched PyTorch port. Tied weights, ReLU, per-tensor symmetric fake-quant with STE.
Usage: from tms_numpy_pilot import train_numpy"""
import numpy as np


def fake_quant_np(W, bits):
    if not np.isfinite(bits):
        return W.copy()
    qmax = 2.0 ** (bits - 1) - 1
    amax = max(np.abs(W).max(), 1e-8)
    s = amax / qmax
    return np.clip(np.round(W / s), -qmax, qmax) * s


def forward_np(Wq, b, X):
    H = X @ Wq.T                      # [B,m]
    Y = H @ Wq + b                    # [B,n]
    return np.maximum(Y, 0.0), H, Y


def train_numpy(W0, b0, X_stream, bits, imp, lr=1e-3, warmup=200, steps=None, betas=(0.9, 0.999), eps=1e-8):
    """X_stream: iterable of [B,n] batches (one per step). Returns W, b, losses (Adam, lr warmup+cosine)."""
    W, b = W0.astype(np.float32).copy(), b0.astype(np.float32).copy()
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    losses = []
    steps = steps or len(X_stream)
    for t, X in enumerate(X_stream):
        if t >= steps:
            break
        lr_t = lr * (t + 1) / warmup if t < warmup else lr * 0.5 * (1 + np.cos(np.pi * (t - warmup) / max(1, steps - warmup)))
        Wq = fake_quant_np(W, bits)
        Xh, H, Y = forward_np(Wq, b, X)
        B = X.shape[0]
        loss = float((imp * (X - Xh) ** 2).sum(1).mean())
        losses.append(loss)
        dY = (-2.0 * imp * (X - Xh) * (Y > 0)) / B          # mean over batch
        db = dY.sum(0)
        dH = dY @ Wq.T                                     # [B,m]
        dW = H.T @ dY + dH.T @ X                            # tied: both uses, STE passes to W
        for P, g, mo, vo in ((W, dW, mW, vW), (b, db, mb, vb)):
            mo *= betas[0]; mo += (1 - betas[0]) * g
            vo *= betas[1]; vo += (1 - betas[1]) * g * g
            mhat = mo / (1 - betas[0] ** (t + 1)); vhat = vo / (1 - betas[1] ** (t + 1))
            P -= lr_t * mhat / (np.sqrt(vhat) + eps)
    return W, b, losses
