"""§5A analysis: per-matrix damage thresholds in GPT-2 vs kappa and the superposition proxy."""
import os, sys, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analysis import ols, COL, RES, FIGS

MAT_COL = {"attn.c_attn": COL["blue"], "attn.c_proj": COL["orange"], "mlp.c_fc": COL["aqua"], "mlp.c_proj": COL["violet"]}


def loglog_crossing(x, y, thr):
    """First crossing of y(x) through thr on log-log axes, x ascending (damage increasing). Returns x_crit, censor."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x); x, y = x[o], y[o]
    if y[0] >= thr:
        return x[0], "low"
    for i in range(len(x) - 1):
        if y[i] < thr <= y[i + 1]:
            y0, y1 = max(y[i], 1e-6), y[i + 1]
            t = (math.log(thr) - math.log(y0)) / (math.log(y1) - math.log(y0))
            return float(np.exp(math.log(x[i]) + t * (math.log(x[i + 1]) - math.log(x[i])))), "ok"
    return x[-1], "high"


def bootstrap_slope(x, y, nboot=2000, seed=0):
    rng = np.random.default_rng(seed); x, y = np.asarray(x), np.asarray(y); n = len(x); sl = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) > 1:
            sl.append(ols(x[i], y[i])[0])
    return np.percentile(sl, [2.5, 97.5])


def run_lm(path=os.path.join(RES, "lm_layers.parquet"), thresholds=(0.02, 0.05, 0.10)):
    df = pd.read_parquet(path)
    rows = []
    for (L, name), g in df.groupby(["block", "matrix"]):
        ga = g[g.family == "gaussian"]; gr = g[g.family == "rtn"]
        r = dict(block=L, matrix=name, kappa=g.kappa.iloc[0], F_hat=g.F_hat.iloc[0], PR=g.PR.iloc[0], d_in=g.d_in.iloc[0])
        for thr in thresholds:
            sc, cen = loglog_crossing(ga.sigma_rel.values, ga.dnll.values, thr)
            r[f"sigma_crit_{thr}"] = sc; r[f"censor_{thr}"] = cen
            # equivalent RTN bit-width whose residual equals sigma_crit: sigma_rel = kappa/(sqrt(12) qmax)
            r[f"b_equiv_{thr}"] = 1 + np.log2(1 + r["kappa"] / (math.sqrt(12) * sc))
            if len(gr):
                # RTN: dnll vs bits (damage increases as bits decrease) -> crossing on -bits
                xb = -gr.knob.values; o = np.argsort(xb); xb, yb = xb[o], gr.dnll.values[o]
                bc = np.nan; cb = "none"
                if yb[0] >= thr: bc, cb = -xb[0], "low"
                else:
                    for i in range(len(xb) - 1):
                        if yb[i] < thr <= yb[i + 1]:
                            y0 = max(yb[i], 1e-6); t = (math.log(thr) - math.log(y0)) / (math.log(yb[i + 1]) - math.log(y0))
                            bc, cb = -(xb[i] + t * (xb[i + 1] - xb[i])), "ok"; break
                    if cb == "none": bc, cb = -xb[-1], "high"
                r[f"b_rtn_{thr}"] = bc; r[f"censor_rtn_{thr}"] = cb
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "lm_layer_thresholds.csv"), index=False)
    # fits
    res = {}
    for thr in thresholds:
        ok = t[t[f"censor_{thr}"] == "ok"]
        x = np.log2(ok.F_hat); y = np.log2(ok[f"sigma_crit_{thr}"])
        sl, ic, r2 = ols(x, y); lo, hi = bootstrap_slope(x, y)
        yk = ok[f"b_equiv_{thr}"] - np.log2(ok.kappa); slk, ick, r2k = ols(x, yk); lok, hik = bootstrap_slope(x, yk)
        # kappa alone and partial: does b_equiv track log2 kappa with slope 1?
        slkap, ickap, r2kap = ols(np.log2(ok.kappa), ok[f"b_equiv_{thr}"])
        # multiple regression b_equiv ~ log2 kappa + log2 F_hat
        A = np.vstack([np.ones(len(ok)), np.log2(ok.kappa), x]).T; coef, *_ = np.linalg.lstsq(A, ok[f"b_equiv_{thr}"], rcond=None)
        res[thr] = dict(n=len(ok), slope_sigma=sl, ci_sigma=(lo, hi), r2_sigma=r2, slope_bkc=slk, ci_bkc=(lok, hik), r2_bkc=r2k,
                        slope_kappa=slkap, r2_kappa=r2kap, multi=coef)
        print(f"thr={thr}: n={len(ok)} | log2 σ_crit ~ log2 F̂: slope {sl:.3f} [{lo:.3f},{hi:.3f}] R²={r2:.2f} | "
              f"(b_equiv − log2κ) ~ log2 F̂: slope {slk:.3f} [{lok:.3f},{hik:.3f}] R²={r2k:.2f} | b_equiv ~ log2κ: slope {slkap:.3f} R²={r2kap:.2f} | "
              f"multi: b = {coef[0]:.2f} + {coef[1]:.2f}·log2κ + {coef[2]:.2f}·log2F̂")
        if f"b_rtn_{thr}" in ok.columns:
            okr = ok[ok[f"censor_rtn_{thr}"] == "ok"]
            if len(okr) >= 3:
                sr, ir, r2r = ols(okr[f"b_equiv_{thr}"], okr[f"b_rtn_{thr}"])
                print(f"   RTN check: b_rtn vs b_equiv (n={len(okr)}): slope {sr:.2f}, R²={r2r:.2f}; mean diff {np.mean(okr[f'b_rtn_{thr}'] - okr[f'b_equiv_{thr}']):+.2f} bits")
    # figure
    thr = 0.05
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    ax = axes[0]
    for (L, name), g in df[df.family == "gaussian"].groupby(["block", "matrix"]):
        ax.plot(g.sigma_rel, g.dnll.clip(lower=1e-4), "-", lw=0.8, color=MAT_COL[name], alpha=0.35 + 0.65 * L / 11)
    for name, c in MAT_COL.items():
        ax.plot([], [], color=c, label=name)
    ax.axhline(thr, color=COL["gray"], ls="--", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("σ_rel (Gaussian weight noise, one matrix at a time)"); ax.set_ylabel("ΔNLL (nats)"); ax.set_title("GPT-2 small: per-matrix damage curves (darker = later block)", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1]
    ok = t[t[f"censor_{thr}"] == "ok"]
    for name, c in MAT_COL.items():
        o = ok[ok.matrix == name]; ax.scatter(np.log2(o.F_hat), np.log2(o[f"sigma_crit_{thr}"]), s=18, color=c, label=name)
        for _, r in o.iterrows():
            ax.annotate(str(int(r.block)), (np.log2(r.F_hat), np.log2(r[f"sigma_crit_{thr}"])), fontsize=5, color=c, xytext=(2, 2), textcoords="offset points")
    x = np.log2(ok.F_hat); y = np.log2(ok[f"sigma_crit_{thr}"]); sl, ic, r2 = ols(x, y); lo, hi = bootstrap_slope(x, y)
    xs = np.linspace(x.min(), x.max(), 10); ax.plot(xs, ic + sl * xs, color=COL["ink"], lw=1.3, label=f"OLS slope {sl:.2f} [{lo:.2f},{hi:.2f}], R²={r2:.2f}")
    ax.plot(xs, (ic + sl * xs.mean() + 0.5 * xs.mean()) - 0.5 * xs, ":", color=COL["violet"], lw=1.2, label="slope −½ reference")
    ax.set_xlabel("log2 F̂  (F̂ = d_in / PR of input covariance)"); ax.set_ylabel(f"log2 σ_crit  (ΔNLL = {thr})"); ax.set_title("noise threshold vs superposition proxy", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax = axes[2]
    for name, c in MAT_COL.items():
        o = ok[ok.matrix == name]; ax.scatter(np.log2(o.kappa), o[f"b_equiv_{thr}"], s=18, color=c, label=name)
    xk = np.log2(ok.kappa); yk = ok[f"b_equiv_{thr}"]; sl, ic, r2 = ols(xk, yk); lo, hi = bootstrap_slope(xk, yk)
    xs = np.linspace(xk.min(), xk.max(), 10); ax.plot(xs, ic + sl * xs, color=COL["ink"], lw=1.3, label=f"OLS slope {sl:.2f} [{lo:.2f},{hi:.2f}], R²={r2:.2f}")
    ax.set_xlabel("log2 κ  (κ = max|W|/rms(W))"); ax.set_ylabel("b_crit,equiv = 1 + log2(1 + κ/(√12 σ_crit))"); ax.set_title("κ sets the bit-width (predicted slope 1)", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig7_lm_layers.png"), bbox_inches="tight"); plt.close(fig)
    return t, res


def run_pc(path=os.path.join(RES, "lm_layers_pc.parquet"), thresholds=(0.02, 0.05, 0.10)):
    """Per-channel RTN: measured b_crit per matrix vs the norm-weighted per-channel κ_eff and vs F̂."""
    df = pd.read_parquet(path); df = df[df.family == "rtn_pc"]
    rows = []
    for (L, name), g in df.groupby(["block", "matrix"]):
        r = dict(block=L, matrix=name, kappa=g.kappa.iloc[0], kappa_eff=g.kappa_eff.iloc[0], F_hat=g.F_hat.iloc[0])
        xb = -g.knob.values; o = np.argsort(xb); xb, yb = xb[o], g.dnll.values[o]     # damage increases as bits decrease
        for thr in thresholds:
            bc, cb = np.nan, "none"
            if yb[0] >= thr: bc, cb = -xb[0], "low"
            else:
                for i in range(len(xb) - 1):
                    if yb[i] < thr <= yb[i + 1]:
                        y0 = max(yb[i], 1e-6); t = (math.log(thr) - math.log(y0)) / (math.log(yb[i + 1]) - math.log(y0))
                        bc, cb = -(xb[i] + t * (xb[i + 1] - xb[i])), "ok"; break
                if cb == "none": bc, cb = -xb[-1], "high"
            r[f"b_pc_{thr}"], r[f"censor_pc_{thr}"] = bc, cb
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "lm_layer_thresholds_pc.csv"), index=False)
    res = {}
    for thr in thresholds:
        ok = t[t[f"censor_pc_{thr}"] == "ok"]
        if len(ok) < 5: continue
        y = ok[f"b_pc_{thr}"]
        for xn, x in [("log2 kappa_eff", np.log2(ok.kappa_eff)), ("log2 kappa(per-tensor)", np.log2(ok.kappa)), ("log2 F_hat", np.log2(ok.F_hat))]:
            sl, ic, r2 = ols(x, y); lo, hi = bootstrap_slope(x, y)
            print(f"per-channel thr={thr} n={len(ok)}: b_crit ~ {xn}: slope {sl:.2f} [{lo:.2f},{hi:.2f}] R²={r2:.2f}")
            res[(thr, xn)] = (sl, lo, hi, r2)
        A = np.vstack([np.ones(len(ok)), np.log2(ok.kappa_eff), np.log2(ok.F_hat)]).T; c, *_ = np.linalg.lstsq(A, y, rcond=None)
        print(f"   joint: b = {c[0]:.2f} + {c[1]:.2f} log2 κ_eff + {c[2]:.2f} log2 F̂")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5)); thr = 0.05; ok = t[t[f"censor_pc_{thr}"] == "ok"]
    for ax, xn, x in [(axes[0], "log2 κ_eff (norm-weighted per-channel outlier ratio)", np.log2(ok.kappa_eff)), (axes[1], "log2 F̂", np.log2(ok.F_hat))]:
        for name, c in MAT_COL.items():
            o = ok[ok.matrix == name]; xx = np.log2(o.kappa_eff) if "κ" in xn else np.log2(o.F_hat)
            ax.scatter(xx, o[f"b_pc_{thr}"], s=18, color=c, label=name)
        sl, ic, r2 = ols(x, ok[f"b_pc_{thr}"]); lo, hi = bootstrap_slope(x, ok[f"b_pc_{thr}"]); xs = np.linspace(x.min(), x.max(), 10)
        ax.plot(xs, ic + sl * xs, color=COL["ink"], lw=1.3, label=f"OLS slope {sl:.2f} [{lo:.2f},{hi:.2f}], R²={r2:.2f}")
        ax.set_xlabel(xn); ax.set_ylabel(f"measured per-channel RTN b_crit (ΔNLL = {thr})"); ax.legend(fontsize=6, frameon=False)
    fig.suptitle("GPT-2 small, per-channel RTN: outliers predict the bit-width, superposition proxy does not", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig7b_lm_perchannel.png"), bbox_inches="tight"); plt.close(fig)
    return t, res


def run_mixed(path=os.path.join(RES, "lm_mixed.parquet")):
    d = pd.read_parquet(path)
    piv = d.pivot_table(index="budget", columns="alloc", values="dnll")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    cols = {"uniform": COL["gray"], "kappa": COL["blue"], "law": COL["violet"], "oracle": COL["orange"]}
    for alloc, c in cols.items():
        g = d[d.alloc == alloc].sort_values("avg_bits")
        if len(g): ax.errorbar(g.avg_bits, g.dnll, yerr=g.sem, fmt="-o", ms=4, lw=1.2, capsize=2, color=c, label=alloc)
    r = d[d.alloc.str.startswith("random")]
    if len(r): ax.scatter(r.avg_bits, r.dnll, s=12, color=COL["red"], alpha=0.6, label="random (same widths as κ)", zorder=1)
    ax.set_yscale("log"); ax.set_xlabel("average bits per weight (parameter-weighted)"); ax.set_ylabel("ΔNLL on held-out tokens (nats)")
    ax.set_title("GPT-2 small, per-channel RTN mixed precision", fontsize=9); ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig9_mixed_precision.png"), bbox_inches="tight"); plt.close(fig)
    return piv


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "noise"
    {"noise": run_lm, "pc": run_pc, "mixed": run_mixed}[what]()
