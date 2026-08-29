import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant import fake_quant


class BatchedTMS(nn.Module):
    """K independent tied-weight toy models: h = W x, x_hat = ReLU(W^T h + b).  W:[K,m,n]."""

    def __init__(self, K, n, m, init_seeds=None, device="cpu"):
        super().__init__()
        W = torch.empty(K, m, n)
        std = (2.0 / (m + n)) ** 0.5                       # xavier-normal for an [n->m] map
        if init_seeds is None:
            init_seeds = list(range(K))
        cache = {}
        for k, sd in enumerate(init_seeds):                # same seed -> identical init (paired arms)
            if sd not in cache:
                g = torch.Generator(device="cpu").manual_seed(int(sd) * 7919 + 17)
                cache[sd] = torch.randn(m, n, generator=g) * std
            W[k] = cache[sd]
        self.W = nn.Parameter(W.to(device))
        self.b = nn.Parameter(torch.zeros(K, n, device=device))

    def effective_weights(self, qmax, mode, dither_v=None, sr_u=None):
        return fake_quant(self.W, qmax, mode, dither_v, sr_u, return_effective=True)

    @staticmethod
    def forward_with(Wq, b, x):
        """Wq:[K,m,n], b:[K,n], x:[K,B,n] -> x_hat:[K,B,n]. Same Wq used in both directions (tied)."""
        h = torch.einsum("kmn,kbn->kbm", Wq, x)
        y = torch.einsum("kmn,kbm->kbn", Wq, h) + b[:, None, :]
        return F.relu(y)

    def forward(self, x, qmax, mode, dither_v=None, sr_u=None):
        Wq = fake_quant(self.W, qmax, mode, dither_v, sr_u)
        return self.forward_with(Wq, self.b, x)
