"""Per-token critical bits vs fp32 logit margin (GPT-2 small)."""
import os, sys, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from .analysis import ols, COL, RES, FIGS
from .analysis_lm import bootstrap_slope
from .lm_tokens import BITS, SIGMAS


def run():
    d = pd.read_parquet(os.path.join(RES, "lm_tokens.parquet"))
    c = d[d.ok_fp32].copy()                                     # tokens the fp32 model predicts correctly
    print(f"{len(d)} tokens; fp32-correct {len(c)} ({len(c)/len(d):.3f})")
    # per-token critical bits: smallest b with the token still correct at b and all larger b (monotone closure)
    okm = np.stack([c[f"ok_rtn{b}"].values for b in BITS], axis=1)           # [N, B] for BITS ascending
    suffix_ok = np.flip(np.cumprod(np.flip(okm, axis=1), axis=1), axis=1).astype(bool)   # correct at b and all above
    bcrit = np.array([BITS[i] if suffix_ok[t].any() else 9 for t, i in zip(range(len(c)), suffix_ok.argmax(axis=1))])
    bcrit = np.where(suffix_ok.any(axis=1), bcrit, 9.0).astype(float)
    c["b_crit"] = bcrit
    c["lm"] = np.log2(c.margin_fp32.clip(lower=1e-3)); c["lf"] = np.log2(c.freq.clip(lower=1))
    # bin by margin (log2 bins), summarize
    bins = np.arange(-4, 7, 1.0); c["mbin"] = pd.cut(c.lm, bins)
    tab = c.groupby("mbin", observed=True).agg(n=("b_crit", "size"), b_mean=("b_crit", "mean"), never=("b_crit", lambda v: (v >= 9).mean()),
                                               **{f"flip{b}": (f"ok_rtn{b}", lambda v: 1 - v.mean()) for b in BITS}).reset_index()
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    # regression of per-token b_crit on log2 margin (tokens with a finite crossing, margin >= 2^-4)
    f = c[(c.b_crit < 9) & (c.lm > -4)]
    sl, ic, r2 = ols(f.lm, f.b_crit); lo, hi = bootstrap_slope(f.lm.values, f.b_crit.values, nboot=300)
    print(f"per-token b_crit ~ log2 margin: slope {sl:.3f} [{lo:.3f},{hi:.3f}] R²={r2:.3f} n={len(f)}")
    # binned (mean per margin bin) regression, which is what a law predicts for the conditional mean
    tb = tab[(tab.n >= 200) & (tab.never < 0.5)]; xb = np.array([iv.mid for iv in tb.mbin]); sb, ib, r2b = ols(xb, tb.b_mean)
    print(f"binned means: slope {sb:.3f} R²={r2b:.3f} over {len(tb)} bins")
    # partial out frequency: b_crit ~ log2 margin + log2 freq
    A = np.vstack([np.ones(len(f)), f.lm, f.lf]).T; coef, *_ = np.linalg.lstsq(A, f.b_crit, rcond=None)
    print(f"joint: b_crit = {coef[0]:.2f} + {coef[1]:.3f} log2 margin + {coef[2]:.3f} log2 freq")
    # critical margin per bit-width: margin at which P(flip) = 0.5 (logistic on log2 margin)
    from scipy.special import expit
    rows = []
    for b in BITS:
        y = 1 - c[f"ok_rtn{b}"].values.astype(float); x = c.lm.values
        # grid search logistic fit of P(flip) = expit(a (x0 - x))
        best = None
        for x0 in np.linspace(-4, 8, 121):
            for a in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
                p = expit(a * (x0 - x)).clip(1e-6, 1 - 1e-6); ll = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
                if best is None or ll > best[0]: best = (ll, x0, a)
        rows.append(dict(bits=b, log2_margin_crit=best[1], steepness=best[2], flip_rate=y.mean()))
    mc = pd.DataFrame(rows); print(mc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    use = mc[(mc.flip_rate > 0.02) & (mc.flip_rate < 0.9)]
    sm, im, r2m = ols(use.bits, use.log2_margin_crit); print(f"log2 margin_crit vs bits: slope {sm:.3f} (predicted −1) R²={r2m:.3f} over bits {list(use.bits)}")
    # Gaussian control: flip rate vs sigma per margin bin, mapped to RTN via sigma_rel of per-channel RTN? report curves
    tab.to_csv(os.path.join(RES, "lm_tokens_margin_bins.csv"), index=False); mc.to_csv(os.path.join(RES, "lm_tokens_margin_crit.csv"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    ax = axes[0]
    for b, col in zip(BITS, [COL["red"], COL["orange"], COL["yellow"], COL["aqua"], COL["blue"], COL["violet"]]):
        ax.plot([iv.mid for iv in tab.mbin], tab[f"flip{b}"], "-o", ms=3, lw=1.2, color=col, label=f"{b}-bit RTN (per-channel)")
    ax.set_xlabel("log2 fp32 logit margin of the correct token"); ax.set_ylabel("P(prediction flips)"); ax.set_title("GPT-2 small: which tokens break, by margin", fontsize=9); ax.legend(fontsize=6, frameon=False)
    ax = axes[1]
    ax.errorbar([iv.mid for iv in tab.mbin], tab.b_mean, fmt="o", ms=4, color=COL["ink"], label="mean per-token critical bits (9 = never)")
    xs = np.linspace(-3.5, 6.5, 10); ax.plot(xs, ib + sb * xs, color=COL["blue"], lw=1.3, label=f"binned slope {sb:.2f} (predicted −1)")
    ax.set_xlabel("log2 fp32 logit margin"); ax.set_ylabel("per-token critical bits"); ax.legend(fontsize=6.5, frameon=False); ax.set_title("token critical bit-width vs margin", fontsize=9)
    ax = axes[2]
    gcols = [COL["red"], COL["orange"], COL["yellow"], COL["aqua"], COL["blue"], COL["violet"], COL["green"]]
    for sg, col in zip(SIGMAS, gcols):
        fl = c.groupby("mbin", observed=True)[f"ok_g{sg}"].apply(lambda v: 1 - v.mean())
        ax.plot([iv.mid for iv in fl.index], fl.values, "-o", ms=3, lw=1.2, color=col, label=f"Gaussian σ_rel={sg}")
    ax.set_xlabel("log2 fp32 logit margin of the correct token"); ax.set_ylabel("P(prediction flips)"); ax.set_title("same ordering under Gaussian weight noise (all matrices)", fontsize=9); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig10_token_margins.png"), bbox_inches="tight"); plt.close(fig)
    return c, tab, mc


if __name__ == "__main__":
    run()
