"""Activation-quantization arm analysis: per-matrix activation b_crit vs activation outlier ratio (and weight kappa)."""
import os, sys, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from .analysis import ols, COL, RES, FIGS
from .analysis_lm import bootstrap_slope, MAT_COL


def run(path=os.path.join(RES, "lm_act.parquet"), thr=0.05):
    d = pd.read_parquet(path); rows = []
    for (L, name), g in d.groupby(["block", "matrix"]):
        g = g.sort_values("bits", ascending=False); xb, yb = -g.bits.values, g.dnll.values     # damage increases as bits decrease
        bc, cb = np.nan, "none"
        if yb[0] >= thr: bc, cb = -xb[0], "low"
        else:
            for i in range(len(xb) - 1):
                if yb[i] < thr <= yb[i + 1]:
                    y0 = max(yb[i], 1e-6); t = (math.log(thr) - math.log(y0)) / (math.log(yb[i + 1]) - math.log(y0)); bc, cb = -(xb[i] + t * (xb[i + 1] - xb[i])), "ok"; break
            if cb == "none": bc, cb = -xb[-1], "high"
        rows.append(dict(block=L, matrix=name, kappa_act=g.kappa_act.iloc[0], kappa_w=g.kappa_w.iloc[0], b_act=bc, censor=cb))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "lm_act_thresholds.csv"), index=False)
    ok = t[t.censor == "ok"]
    print(f"{len(t)} matrices, {len(ok)} with an interpolable activation b_crit (censored: {t.censor.value_counts().to_dict()})")
    print(f"kappa_act range {t.kappa_act.min():.1f}-{t.kappa_act.max():.1f}; b_act range {ok.b_act.min():.2f}-{ok.b_act.max():.2f}")
    res = {}
    for xn, x in [("log2 kappa_act", np.log2(ok.kappa_act)), ("log2 kappa_w", np.log2(ok.kappa_w))]:
        sl, ic, r2 = ols(x, ok.b_act); lo, hi = bootstrap_slope(x.values, ok.b_act.values); res[xn] = (sl, lo, hi, r2)
        print(f"activation b_crit ~ {xn}: slope {sl:.2f} [{lo:.2f},{hi:.2f}] R²={r2:.2f}")
    A = np.vstack([np.ones(len(ok)), np.log2(ok.kappa_act), np.log2(ok.kappa_w)]).T; c, *_ = np.linalg.lstsq(A, ok.b_act, rcond=None)
    print(f"joint: b_act = {c[0]:.2f} + {c[1]:.2f} log2 kappa_act + {c[2]:.2f} log2 kappa_w")
    print(t.groupby("matrix").agg(k_act=("kappa_act", "mean"), b_act=("b_act", "mean")).to_string(float_format=lambda v: f"{v:.2f}"))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for name, col in MAT_COL.items():
        o = ok[ok.matrix == name]; ax.scatter(np.log2(o.kappa_act), o.b_act, s=18, color=col, label=name)
    x = np.log2(ok.kappa_act); sl, ic, r2 = ols(x, ok.b_act); lo, hi = bootstrap_slope(x.values, ok.b_act.values); xs = np.linspace(x.min(), x.max(), 10)
    ax.plot(xs, ic + sl * xs, color=COL["ink"], lw=1.3, label=f"OLS slope {sl:.2f} [{lo:.2f},{hi:.2f}], R²={r2:.2f}")
    ax.set_xlabel("log2 κ_act  (max|x| / rms x of the matrix input)"); ax.set_ylabel("activation b_crit (ΔNLL = 0.05)"); ax.set_title("GPT-2 small: activation outliers set activation bits", fontsize=9)
    ax.legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig12_activation_kappa.png"), bbox_inches="tight"); plt.close(fig)
    return t, res


if __name__ == "__main__":
    run()
