"""Figure 1 for the 4-page version: (a) b_crit flat in F (E1), (b) kappa causal (E6) + GPT-2 per-matrix, (c) SAE features."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from .analysis import ols, COL, P_COL, M_MK, RES, FIGS, predicted_bcrit
plt.rcParams.update({"font.size": 8})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
# (a) E1
cm = pd.read_csv(os.path.join(RES, "E1_bcrit_R_live_95.csv")); ok = cm[cm.n_ok == cm.n_seeds]
ax = axes[0]
for _, r in ok.iterrows():
    ax.errorbar(np.log2(r.F), r.b_crit, yerr=r.b_sd, fmt=M_MK.get(int(r.m), "o"), color=P_COL.get(r.p, COL["gray"]), ms=4, capsize=1.5, lw=0.8)
xs = np.linspace(1.8, 6.1, 20); sl, ic, r2 = ols(np.log2(ok.F), ok.b_crit)
ax.plot(xs, ic + sl * xs, color=COL["ink"], lw=1.4, label=f"measured: slope {sl:.2f} [−0.12, −0.09]")
ax.plot(xs, predicted_bcrit(2 ** xs, ok.kappa.mean()), "--", color=COL["violet"], lw=1.2, label="preregistered law (slope ½)")
for p, c in P_COL.items(): ax.plot([], [], "o", color=c, label=f"p = {p}")
ax.set_xlabel("log$_2$ F  (features per dimension, measured)"); ax.set_ylabel("critical bit-width $b_{crit}$"); ax.set_title("(a) precision cost is flat in superposition F", fontsize=8.5)
ax.legend(fontsize=5.8, frameon=False, ncol=2, loc="upper left"); ax.set_ylim(2.8, 6.2)
# (b) kappa: E6 within-cell + GPT-2 per-matrix RTN cliff vs kappa
ax = axes[1]
e6 = pd.read_csv(os.path.join(RES, "E6_bcrit_by_kclip.csv")); e6 = e6[e6.n_ok == e6.n_seeds]
for (n, p), g in e6.groupby(["n", "p"]):
    g = g.sort_values("kappa"); ax.errorbar(np.log2(g.kappa), g.b_crit, yerr=g.b_sd, fmt="-o", ms=3.5, lw=1, capsize=1.5, color=P_COL.get(p, COL["gray"]), alpha=0.9, label=f"toy, κ clipped at fixed config (n={n}, p={p})")
lm = pd.read_csv(os.path.join(RES, "lm_layer_thresholds.csv")); lm = lm[lm["censor_rtn_0.05"] == "ok"]
ax2 = ax.twinx() if False else ax
sc = ax.scatter(np.log2(lm.kappa), lm["b_rtn_0.05"], s=12, color=COL["ink"], alpha=0.55, marker="^", label="GPT-2, 48 matrices: RTN cliff at ΔNLL=0.05")
slk, ick, r2k = ols(np.log2(lm.kappa), lm["b_rtn_0.05"]); xs = np.linspace(3, 7.6, 10); ax.plot(xs, ick + slk * xs, color=COL["ink"], lw=1, ls=":", label=f"GPT-2 fit: slope {slk:.2f}, R²={r2k:.2f}")
ax.set_xlabel("log$_2$ κ  (max|W| / rms W)"); ax.set_ylabel("critical bit-width"); ax.set_title("(b) …and set by the outlier ratio κ", fontsize=8.5)
ax.legend(fontsize=5.5, frameon=False, loc="upper left")
# (c) SAE
tab = pd.read_csv(os.path.join(RES, "lm_sae_strength_bins.csv"))
mid = tab.sbin.str.extract(r"\(([-\d.]+), ([-\d.]+)\]").astype(float).mean(axis=1)
ax = axes[2]
for b, col in zip([3, 4, 5, 6], [COL["orange"], COL["yellow"], COL["aqua"], COL["blue"]]):
    ax.plot(mid, 1 - tab[f"ret{b}"], "-o", ms=3, lw=1.2, color=col, label=f"{b}-bit RTN")
ax.set_xlabel("log$_2$ feature strength (fp32, GPT-2 layer-6 SAE)"); ax.set_ylabel("P(feature erased)"); ax.set_title("(c) GPT-2: the weakest features are erased first", fontsize=8.5)
ax.legend(fontsize=6, frameon=False); ax.set_xlim(-8.5, 1.5)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_hero.png"), dpi=200, bbox_inches="tight"); print("hero figure written")
