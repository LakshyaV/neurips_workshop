"""§5E analysis: per-SAE-feature critical bits vs feature strength (GPT-2 small, layer-6 residual SAE)."""
import os, sys, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from .analysis import ols, COL, RES, FIGS
from .analysis_lm import bootstrap_slope
from .lm_sae import BITS, SIGMAS


def run(path=os.path.join(RES, "lm_sae.parquet"), min_act=20, thr=0.5):
    d = pd.read_parquet(path); N = int(d.n_tokens.iloc[0])
    live = d[d.n_act0_fp32 >= min_act].copy()
    print(f"{len(d)} SAE features; {len(live)} fire on >= {min_act} of {N} tokens (live)")
    live["freq"] = live.n_act0_fp32 / N
    live["strength"] = np.sqrt(live.sum_a0sq_fp32 / N) * live.dec_norm          # rms contribution to the residual stream
    live["mean_act"] = live.sum_a0_fp32 / live.n_act0_fp32 * live.dec_norm         # typical magnitude when active
    conds = [f"rtn{b}" for b in BITS] + [f"g{s}" for s in SIGMAS]
    for c in conds:
        live[f"rho_{c}"] = 1 - live[f"sum_err_{c}"] / live.sum_a0sq_fp32.clip(lower=1e-12)
        live[f"fire_{c}"] = live[f"n_joint_{c}"] / live.n_act0_fp32
    # per-feature critical bits: smallest b such that rho >= thr at b and all larger b
    okm = np.stack([(live[f"rho_rtn{b}"] >= thr).values for b in BITS], axis=1)
    suffix = np.flip(np.cumprod(np.flip(okm, 1), 1), 1).astype(bool)
    live["b_crit"] = np.where(suffix.any(1), np.array(BITS, float)[suffix.argmax(1)], 9.0)
    live["ls"] = np.log2(live.strength); live["lf"] = np.log2(live.freq); live["lm"] = np.log2(live.mean_act)
    print("fraction of live features retained (rho >= 0.5) per condition:")
    print("  " + "  ".join(f"{c}:{(live[f'rho_{c}'] >= thr).mean():.3f}" for c in conds))
    print("mean firing retention per condition:")
    print("  " + "  ".join(f"{c}:{live[f'fire_{c}'].mean():.3f}" for c in conds))
    f = live[live.b_crit < 9]
    for xn, x in [("log2 strength (rms residual contribution)", f.ls), ("log2 frequency", f.lf), ("log2 typical magnitude", f.lm)]:
        sl, ic, r2 = ols(x, f.b_crit); lo, hi = bootstrap_slope(x.values, f.b_crit.values, nboot=300)
        print(f"b_crit ~ {xn}: slope {sl:.3f} [{lo:.3f},{hi:.3f}] R²={r2:.3f} n={len(f)}")
    A = np.vstack([np.ones(len(f)), f.lm, f.lf]).T; c, *_ = np.linalg.lstsq(A, f.b_crit, rcond=None)
    print(f"joint: b_crit = {c[0]:.2f} + {c[1]:.3f} log2 magnitude + {c[2]:.3f} log2 frequency")
    bins = np.arange(np.floor(live.ls.min()), np.ceil(live.ls.max()) + 1, 1.0); live["sbin"] = pd.cut(live.ls, bins)
    tab = live.groupby("sbin", observed=True).agg(n=("b_crit", "size"), b_mean=("b_crit", "mean"), never=("b_crit", lambda v: (v >= 9).mean()),
                                                 **{f"ret{b}": (f"rho_rtn{b}", lambda v: (v >= thr).mean()) for b in BITS}).reset_index()
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    tb = tab[tab.n >= 50]; xb = np.array([iv.mid for iv in tb.sbin]); sb, ib, r2b = ols(xb, tb.b_mean); print(f"binned means: slope {sb:.3f} R²={r2b:.3f} over {len(tb)} bins")
    live.to_csv(os.path.join(RES, "lm_sae_features.csv"), index=False); tab.to_csv(os.path.join(RES, "lm_sae_strength_bins.csv"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    ax = axes[0]
    for b, col in zip(BITS, [COL["red"], COL["orange"], COL["yellow"], COL["aqua"], COL["blue"], COL["violet"]]):
        ax.plot([iv.mid for iv in tab.sbin], 1 - tab[f"ret{b}"], "-o", ms=3, lw=1.2, color=col, label=f"{b}-bit RTN (per-channel, blocks 0–5)")
    ax.set_xlabel("log2 feature strength (rms contribution to the residual stream, fp32)"); ax.set_ylabel("P(feature lost: ρ_f < 0.5)"); ax.set_title("GPT-2 small, 24,576 SAE features at layer 6: which are erased", fontsize=8); ax.legend(fontsize=6, frameon=False)
    ax = axes[1]
    ax.errorbar([iv.mid for iv in tab.sbin], tab.b_mean, fmt="o", ms=4, color=COL["ink"], label="mean per-feature critical bits (9 = never)")
    xs = np.linspace(xb.min() - 0.5, xb.max() + 0.5, 10); ax.plot(xs, ib + sb * xs, color=COL["blue"], lw=1.3, label=f"binned slope {sb:.2f}")
    ax.set_xlabel("log2 feature strength"); ax.set_ylabel("per-feature critical bits"); ax.legend(fontsize=6.5, frameon=False); ax.set_title("feature critical bit-width vs strength", fontsize=9)
    ax = axes[2]
    for s, col in zip(SIGMAS, [COL["red"], COL["orange"], COL["aqua"], COL["blue"]]):
        lost = live.groupby("sbin", observed=True)[f"rho_g{s}"].apply(lambda v: (v < thr).mean())
        ax.plot([iv.mid for iv in lost.index], lost.values, "-o", ms=3, lw=1.2, color=col, label=f"Gaussian σ_rel={s}")
    ax.set_xlabel("log2 feature strength"); ax.set_ylabel("P(feature lost)"); ax.set_title("same ordering under Gaussian weight noise", fontsize=9); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig11_sae_features.png"), bbox_inches="tight"); plt.close(fig)
    return live, tab


if __name__ == "__main__":
    run()
