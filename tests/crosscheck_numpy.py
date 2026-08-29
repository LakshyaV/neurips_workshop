"""Cross-check the batched torch port against the numpy reference on one config (same init, same x stream)."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from tms_numpy_pilot import train_numpy, fake_quant_np, forward_np
from src.model import BatchedTMS
from src.quant import fake_quant, qmax_from_bits
from src.train import importance_vec

m, n, p, B, steps, bits = 10, 80, 0.02, 256, 300, 4.0
rng = np.random.default_rng(0)
W0 = (rng.standard_normal((m, n)) * math.sqrt(2.0 / (m + n))).astype(np.float32)
b0 = np.zeros(n, np.float32)
X = [((rng.random((B, n)) < p) * rng.random((B, n))).astype(np.float32) for _ in range(steps)]
imp = np.ones(n, np.float32)
t0 = time.time(); Wn, bn, Ln = train_numpy(W0, b0, X, bits, imp, steps=steps); tn = time.time() - t0

for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
    model = BatchedTMS(1, n, m, device=dev)
    with torch.no_grad():
        model.W.copy_(torch.from_numpy(W0)[None]); model.b.zero_()
    qmax = qmax_from_bits(torch.tensor([bits])).view(1, 1, 1).to(dev); mode = torch.zeros(1, 1, 1, dtype=torch.long, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Lt = []
    t0 = time.time()
    for t in range(steps):
        lr_t = 1e-3 * (t + 1) / 200 if t < 200 else 1e-3 * 0.5 * (1 + math.cos(math.pi * (t - 200) / max(1, steps - 200)))
        for g in opt.param_groups: g["lr"] = lr_t
        x = torch.from_numpy(X[t])[None].to(dev)
        y = model(x, qmax, mode)
        loss = ((x - y) ** 2).sum(-1).mean(1).sum()
        opt.zero_grad(); loss.backward(); opt.step(); Lt.append(loss.item())
    tt = time.time() - t0
    Wt = model.W.detach().cpu().numpy()[0]; bt = model.b.detach().cpu().numpy()[0]
    print(f"[{dev}] max|W_torch-W_numpy| = {np.abs(Wt-Wn).max():.2e} (|W| max {np.abs(Wn).max():.3f}); "
          f"max|b diff| = {np.abs(bt-bn).max():.2e}; loss diff mean {np.abs(np.array(Lt)-np.array(Ln)).mean():.2e}; "
          f"final loss torch {Lt[-1]:.5f} numpy {Ln[-1]:.5f}; time torch {tt:.1f}s numpy {tn:.1f}s")
    assert np.abs(Wt - Wn).max() < 2e-3 and np.abs(np.array(Lt) - np.array(Ln)).mean() < 1e-4
print("CROSSCHECK OK")
