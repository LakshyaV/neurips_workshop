"""Concept figure for the per-feature margin law, drawn from real E5 data (alpha=1, n=128, p=0.005, seed 0):
sorted rho profile at fp32 vs 4-bit, the retention threshold, and the margin of each feature."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from .analysis import COL, RES, FIGS
df = pd.read_parquet(os.path.join(RES, "E5.parquet"))
g = df[(df.alpha == 1.0) & (df.n == 128) & (df.p == 0.005) & (df.seed == 0) & (df["mode"] == 0)]
fp = g[np.isinf(g.bits)].sort_values("feature_idx"); b4 = g[g.bits == 4].sort_values("feature_idx"); b3 = g[g.bits == 3].sort_values("feature_idx")
live = fp.w_norm.values >= 0.5
i = np.arange(len(fp))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axes[0]
ax.plot(i, fp.rho_i.values, "-", color=COL["ink"], lw=1.5, label="fp32")
ax.plot(i, b4.rho_i.values, "-", color=COL["blue"], lw=1.2, label="4-bit RTN (QAT)")
ax.plot(i, b3.rho_i.values, "-", color=COL["orange"], lw=1.2, label="3-bit RTN (QAT)")
ax.axhline(0.5, color=COL["gray"], ls="--", lw=1); ax.text(2, 0.52, "retention threshold ρ = 0.5", fontsize=7, color=COL["gray"])
k = 40; ax.annotate("", xy=(k, 0.5), xytext=(k, fp.rho_i.values[k]), arrowprops=dict(arrowstyle="<->", color=COL["violet"], lw=1.2))
ax.text(k + 2, 0.5 * (0.5 + fp.rho_i.values[k]), "margin of feature k\n(ρ_k − ½ at fp32)", fontsize=7, color=COL["violet"], va="center")
ax.set_xlabel("feature index i  (importance I_i = (i+1)^−1, decreasing)"); ax.set_ylabel("ρ_i  (fraction of feature variance reconstructed)")
ax.set_ylim(-0.05, 1.05); ax.set_title("per-feature retention profile: quantization pushes the low-margin tail under the threshold", fontsize=8); ax.legend(fontsize=7, frameon=False)
ax = axes[1]
m = (fp.rho_i.values - 0.5); ok = live & (m > 0)
t = pd.read_csv(os.path.join(RES, "E5_per_feature_bits.csv")); t = t[(t.alpha == 1.0) & (t.n == 128) & (t.p == 0.005)]
mm = t.groupby("feature_idx").agg(b=("b_i", "mean"), r=("rho_inf", "mean")).reset_index(); mm = mm[mm.r > 0.5]
ax.scatter(np.log2(mm.r - 0.5), mm.b, s=10, color=COL["violet"], alpha=0.7, label="features (mean over 4 seeds)")
x = np.log2(mm.r - 0.5); A = np.vstack([np.ones(len(x)), x]).T; c, *_ = np.linalg.lstsq(A, mm.b, rcond=None); xs = np.linspace(x.min(), x.max(), 10)
ax.plot(xs, c[0] + c[1] * xs, color=COL["ink"], lw=1.3, label=f"fit: b_i = {c[0]:.1f} {c[1]:+.2f}·log2(margin)")
ax.set_xlabel("log2 margin_i  (ρ_i^fp32 − ½)"); ax.set_ylabel("per-feature critical bits b_i"); ax.set_title("the margin sets the bit-width at which the feature dies (this configuration)", fontsize=8); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig13_margin_concept.png"), dpi=200, bbox_inches="tight"); print("margin concept figure written")
