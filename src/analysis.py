"""Analysis: per-model summaries, retention curves, b_crit extraction, fits with seed bootstrap, figures."""
import os, sys, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIGS = os.path.join(ROOT, "results"), os.path.join(ROOT, "figs")
COL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
           green="#008300", violet="#4a3aa7", red="#e34948", gray="#8a8984", ink="#0b0b0b")
P_COL = {0.002: COL["blue"], 0.005: COL["orange"], 0.02: COL["aqua"], 0.05: COL["yellow"]}
M_MK = {10: "o", 16: "s", 30: "^"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "figure.dpi": 150})
FIT_BITS = (3.0, 8.0)          # b = 2 excluded from every fit (PREDICTIONS.md)
CFG = ["exp", "group", "m", "n", "p", "alpha", "kclip", "seed"]


def load(*exps):
    dfs = [pd.read_parquet(os.path.join(RES, f"{e}.parquet")) for e in exps if os.path.exists(os.path.join(RES, f"{e}.parquet"))]
    return pd.concat(dfs, ignore_index=True)


def model_summary(df):
    """One row per model (run_id) with counts and constants."""
    d = df.assign(r25=(df.rho_i >= 0.25), r50=(df.rho_i >= 0.5), r75=(df.rho_i >= 0.75),
                  rpos=df.rho_i.clip(lower=0), live=(df.w_norm >= 0.5), wn2=df.w_norm ** 2)
    const = [c for c in ["exp", "group", "m", "n", "p", "alpha", "bits", "mode", "mode_name", "damage_type", "knob",
                         "seed", "kclip", "kappa", "kappa_eff", "sigma_eps_measured", "delta", "amax", "rms_w",
                         "loss_final", "sigma_rel", "draw", "steps", "eval_samples"] if c in d.columns]
    g = d.groupby("run_id", sort=False)
    s = g[const].first()
    s["nret_25"], s["nret_50"], s["nret_75"] = g.r25.sum(), g.r50.sum(), g.r75.sum()
    s["soft"], s["soft_c"] = g.rho_i.sum(), g.rpos.sum()            # unclipped Σρ_i (unbiased) is primary; clipped kept for reference
    s["live"], s["wn2_sum"], s["wn2_max"] = g.live.sum(), g.wn2.sum(), g.wn2.max()
    s["rho_mean_live"] = (d.rho_i * d.live).groupby(d.run_id, sort=False).sum() / g.live.sum().clamp(1) if hasattr(g.live.sum(), "clamp") else \
        (d.rho_i * d.live).groupby(d.run_id, sort=False).sum() / g.live.sum().clip(lower=1)
    s["nfeat"] = g.size()
    return s.reset_index()


def add_soft_live(df, s, is_ref, keys):
    """soft_live = Σ_{i live in the reference model (‖W_i‖ ≥ 0.5)} ρ_i, per model; reference = same keys."""
    ref = df[is_ref][keys + ["feature_idx", "w_norm"]].rename(columns={"w_norm": "w_norm_ref"})
    d = df[keys + ["run_id", "feature_idx", "rho_i"]].merge(ref, on=keys + ["feature_idx"], how="inner")
    d["rl"] = d.rho_i * (d.w_norm_ref >= 0.5)
    sl = d.groupby("run_id", sort=False).rl.sum().rename("soft_live")
    return s.merge(sl, left_on="run_id", right_index=True, how="left")


def attach_fp32_ref(s, ref_mode=0):
    """Attach the fp32 (bits=inf, mode=0) arm of the same (config, seed) as *_inf columns."""
    ref = s[np.isinf(s.bits) & (s["mode"] == ref_mode)]
    cols = [c for c in ["soft", "soft_c", "soft_live", "nret_50", "nret_25", "nret_75", "live", "kappa", "wn2_sum", "wn2_max", "rms_w", "amax", "loss_final"] if c in ref.columns]
    ref = ref[CFG + cols]
    ref = ref.rename(columns={c: c + "_inf" for c in ref.columns if c not in CFG})
    out = s.merge(ref, on=CFG, how="left")
    out["R_soft"] = out.soft / out.soft_inf
    if "soft_live" in out.columns:
        out["R_live"] = out.soft_live / out.soft_live_inf
        out["R_softc"] = out.soft_c / out.soft_c_inf
    out["R_hard"] = out.nret_50 / out.nret_50_inf.replace(0, np.nan)
    out["R_hard25"] = out.nret_25 / out.nret_25_inf.replace(0, np.nan)
    out["R_hard75"] = out.nret_75 / out.nret_75_inf.replace(0, np.nan)
    out["F"] = out.live_inf / out.m
    out["ell_inf"] = out.live_inf / out.n
    out["kappa_live_inf"] = out.kappa_inf * np.sqrt(out.ell_inf)          # amax relative to rms of the live weights
    out["kappa_live"] = out.kappa * np.sqrt(out.live.clip(lower=1) / out.n)
    out["F_soft"] = out.wn2_sum_inf / out.wn2_max_inf / out.m
    # continuous axis: effective bits from the measured residual (PREDICTIONS.md rule 5)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["b_eff"] = np.where(np.isinf(out.bits), np.inf,
                                1 + np.log2(1 + out.amax / (np.sqrt(12) * out.sigma_eps_measured.replace(0, np.nan))))
    return out


def crossing(xs, ys, target, xmin=FIT_BITS[0], xmax=FIT_BITS[1]):
    """First upward crossing of ys(xs) through target on xs in [xmin, xmax]; censor flags as in metrics."""
    pts = sorted((float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y) and xmin <= x <= xmax)
    if not pts:
        return np.nan, "none"
    if pts[0][1] >= target:
        return pts[0][0], "low"
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if y0 < target <= y1:
            return (x1 if y1 == y0 else x0 + (target - y0) / (y1 - y0) * (x1 - x0)), "ok"
    return pts[-1][0], "high"


def bcrit_table(s, curve="R_soft", frac=0.95, axis="bits", arm_cols=("mode",), extra_keys=()):
    """One row per (config, seed, arm): b_crit on `axis` from `curve`."""
    keys = CFG + list(arm_cols) + list(extra_keys)
    rows = []
    for k, g in s[~np.isinf(s.bits)].groupby(keys, sort=False):
        ref = s[(s[CFG] == pd.Series(dict(zip(CFG, k[:len(CFG)])))).all(axis=1) & np.isinf(s.bits) & (s["mode"] == 0)]
        if len(ref) == 0 or g[curve].isna().all():
            continue
        bc, cen = crossing(g[axis].values, g[curve].values, frac)
        r = dict(zip(keys, k)); r.update(b_crit=bc, censor=cen, F=g.F.iloc[0], F_soft=g.F_soft.iloc[0],
                                         kappa_inf=g.kappa_inf.iloc[0], live_inf=g.live_inf.iloc[0], soft_inf=g.soft_inf.iloc[0])
        rows.append(r)
    return pd.DataFrame(rows)


def config_means(tab, keys=("m", "n", "p", "alpha", "kclip", "mode")):
    keys = [k for k in keys if k in tab.columns]
    ok = tab[tab.censor == "ok"]
    if "F_soft" not in tab.columns:
        tab = tab.assign(F_soft=tab.F)
    g = tab.groupby(keys)
    out = g.agg(F=("F", "mean"), F_soft=("F_soft", "mean"), kappa=("kappa_inf", "mean"), n_seeds=("seed", "nunique"),
                n_low=("censor", lambda c: (c == "low").sum()), n_high=("censor", lambda c: (c == "high").sum())).reset_index()
    okm = ok.groupby(keys).agg(b_crit=("b_crit", "mean"), b_sd=("b_crit", "std"), n_ok=("b_crit", "size")).reset_index()
    return out.merge(okm, on=keys, how="left")


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    return coef[1], coef[0], r2


def fit_slope_bootstrap(tab, xcol="F", ycol="b_crit", nboot=1000, seed=0, min_ok=None, keys=("m", "n", "p", "alpha", "kclip", "mode")):
    """OLS of mean-over-seeds b_crit on log2 x, configs with all seeds 'ok'; bootstrap over seeds."""
    keys = [k for k in keys if k in tab.columns]
    seeds = sorted(tab.seed.unique())
    cm = config_means(tab, keys)
    min_ok = len(seeds) if min_ok is None else min_ok
    use = cm[(cm.n_ok >= min_ok)]
    if len(use) < 3:
        return dict(slope=np.nan, lo=np.nan, hi=np.nan, intercept=np.nan, r2=np.nan, n_cfg=len(use))
    slope, icpt, r2 = ols(np.log2(use[xcol]), use[ycol])
    ok = tab[tab.censor == "ok"].set_index(keys)
    rng = np.random.default_rng(seed)
    bs = []
    cfg_idx = use.set_index(keys).index
    for _ in range(nboot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        xs, ys = [], []
        for cfg in cfg_idx:
            sub = ok.loc[[cfg]] if cfg in ok.index else None
            if sub is None:
                continue
            vals = pd.concat([sub[sub.seed == sd] for sd in pick])
            if len(vals) == 0:
                continue
            xs.append(np.log2(vals[xcol].mean())); ys.append(vals[ycol].mean())
        if len(xs) >= 3:
            bs.append(ols(xs, ys)[0])
    bs = np.array(bs)
    return dict(slope=slope, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5), intercept=icpt, r2=r2,
                n_cfg=len(use), bs_sd=bs.std())


def predicted_bcrit(F, kappa):
    return 1 + np.log2(1 + kappa * np.sqrt(np.asarray(F, float) / 3.0))


# ------------------------------------------------------------------------------------ figures
def fig_money(cm, fit, fit_kc, title, path, ycol="b_crit", show_pred=True):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    for _, r in cm.iterrows():
        c, mk = P_COL.get(r.p, COL["gray"]), M_MK.get(int(r.m), "o")
        x = np.log2(r.F)
        if r.n_ok >= 1 and np.isfinite(r[ycol]):
            ax.errorbar(x, r[ycol], yerr=(r.b_sd if np.isfinite(r.b_sd) else 0), fmt=mk, color=c, ms=5, mfc=c if r.n_ok == r.n_seeds else "white",
                        capsize=2, lw=1, alpha=0.95)
        if r.n_low > 0:
            ax.annotate("", xy=(x, 2.75), xytext=(x, 3.15), arrowprops=dict(arrowstyle="->", color=c, lw=1))
        if r.n_high > 0:
            ax.annotate("", xy=(x, 8.25), xytext=(x, 7.85), arrowprops=dict(arrowstyle="->", color=c, lw=1))
    xs = np.linspace(np.log2(cm.F.min()) - 0.3, np.log2(cm.F.max()) + 0.3, 50)
    if np.isfinite(fit["slope"]):
        ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color=COL["ink"], lw=1.5,
                label=f"OLS slope {fit['slope']:.2f} [{fit['lo']:.2f}, {fit['hi']:.2f}]")
    if show_pred:
        kbar = cm.kappa.mean()
        ax.plot(xs, predicted_bcrit(2 ** xs, kbar), "--", color=COL["violet"], lw=1.2,
                label=f"preregistered law, κ̄={kbar:.2f} (slope→½)")
    for p, c in P_COL.items():
        ax.plot([], [], "o", color=c, label=f"p={p}")
    for m, mk in M_MK.items():
        if m in cm.m.values:
            ax.plot([], [], mk, color=COL["gray"], mfc="white", label=f"m={m}")
    ax.set_xlabel("log2 F   (F = live features / m, measured at fp32)"); ax.set_ylabel("b_crit (bits)")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=6.5, loc="upper left", ncol=2, frameon=False)
    # panel 2: kappa-corrected
    ax = axes[1]
    for _, r in cm.iterrows():
        if r.n_ok >= 1 and np.isfinite(r[ycol]):
            ax.errorbar(np.log2(r.F), r[ycol] - np.log2(r.kappa), yerr=(r.b_sd if np.isfinite(r.b_sd) else 0), fmt=M_MK.get(int(r.m), "o"),
                        color=P_COL.get(r.p, COL["gray"]), ms=5, capsize=2, lw=1, mfc=P_COL.get(r.p, COL["gray"]) if r.n_ok == r.n_seeds else "white")
    if np.isfinite(fit_kc["slope"]):
        ax.plot(xs, fit_kc["intercept"] + fit_kc["slope"] * xs, color=COL["ink"], lw=1.5,
                label=f"OLS slope {fit_kc['slope']:.2f} [{fit_kc['lo']:.2f}, {fit_kc['hi']:.2f}]")
    ax.plot(xs, 0.5 * xs + (fit_kc["intercept"] + fit_kc["slope"] * xs.mean() - 0.5 * xs.mean() if np.isfinite(fit_kc["slope"]) else 0),
            ":", color=COL["violet"], lw=1.2, label="slope ½ reference")
    ax.set_xlabel("log2 F"); ax.set_ylabel("b_crit − log2 κ  (κ measured per config)"); ax.set_title("κ-corrected", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    # panel 3: measured vs predicted (exact form, no free parameter)
    ax = axes[2]
    ok = cm[(cm.n_ok >= 1) & np.isfinite(cm[ycol])]
    pred = predicted_bcrit(ok.F, ok.kappa)
    for (_, r), pv in zip(ok.iterrows(), pred):
        ax.errorbar(pv, r[ycol], yerr=(r.b_sd if np.isfinite(r.b_sd) else 0), fmt=M_MK.get(int(r.m), "o"), color=P_COL.get(r.p, COL["gray"]), ms=5, capsize=2, lw=1)
    lim = [min(2.5, pred.min() - 0.3) if len(pred) else 2.5, max(8.5, pred.max() + 0.3) if len(pred) else 8.5]
    ax.plot(lim, lim, "--", color=COL["gray"], lw=1, label="y = x")
    if len(ok) >= 3:
        sl, ic, r2 = ols(pred, ok[ycol]); ax.plot(np.array(lim), ic + sl * np.array(lim), color=COL["ink"], lw=1.2, label=f"fit slope {sl:.2f}, R²={r2:.2f}")
    ax.set_xlabel("predicted b_crit = 1 + log2(1 + κ√(F/3))"); ax.set_ylabel("measured b_crit"); ax.set_title("zero-parameter prediction", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_curves(s, path, curve="R_soft", ms=(10, 16), title="retention vs bits"):
    ratios = sorted((s.n // s.m).unique())
    fig, axes = plt.subplots(len(ms), len(ratios), figsize=(2.3 * len(ratios), 2.2 * len(ms)), sharex=True, sharey=True, squeeze=False)
    for i, m in enumerate(ms):
        for j, r in enumerate(ratios):
            ax = axes[i, j]
            g = s[(s.m == m) & (s.n == m * r) & ~np.isinf(s.bits)]
            for p, gp in g.groupby("p"):
                mean = gp.groupby("bits")[curve].mean(); sd = gp.groupby("bits")[curve].std()
                ax.plot(mean.index, mean.values, "-o", ms=3, lw=1.2, color=P_COL.get(p, COL["gray"]), label=f"p={p}")
                ax.fill_between(mean.index, (mean - sd).values, (mean + sd).values, color=P_COL.get(p, COL["gray"]), alpha=0.15, lw=0)
            ax.axhline(0.95, color=COL["gray"], lw=0.8, ls="--")
            ax.set_title(f"m={m}, n/m={r}", fontsize=8)
            if i == len(ms) - 1: ax.set_xlabel("bits")
            if j == 0: ax.set_ylabel(curve)
    axes[0, 0].legend(fontsize=6, frameon=False)
    axes[0, 0].set_ylim(-0.05, 1.1)
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_e0(df, path):
    s = model_summary(df)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    ax = axes[0]
    S = 1 - s.p
    Dstar = s.m / s.wn2_sum
    ax.scatter(1 / (1 - S), Dstar, s=18, color=COL["blue"], zorder=3)
    for v, lab in [(1, "1"), (3 / 4, "3/4"), (2 / 3, "2/3"), (1 / 2, "1/2"), (2 / 5, "2/5"), (3 / 8, "3/8")]:
        ax.axhline(v, color=COL["gray"], lw=0.6, ls="--"); ax.text(1.05, v, lab, fontsize=7, va="bottom", color=COL["gray"])
    ax.set_xscale("log"); ax.set_xlabel("1/(1−S)"); ax.set_ylabel("D* = m / Σ‖W_i‖²"); ax.set_ylim(0, 1.25)
    ax.set_title("E0: dimensions per feature (n=400, m=30, uniform I)", fontsize=8)
    ax = axes[1]
    for p, g in df.groupby("p"):
        ax.hist(g.D_i.values, bins=np.linspace(0, 1.05, 43), histtype="step", lw=1, label=f"p={p}")
    ax.set_xlabel("feature dimensionality D_i"); ax.set_ylabel("count"); ax.set_yscale("log"); ax.legend(fontsize=5.5, ncol=2, frameon=False)
    ax.set_title("D_i distribution", fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def top3_table(df, s):
    """E4: rho of the top-3 features (ranked by fp32 rho, same config+seed) at each bit-width."""
    ref = df[np.isinf(df.bits) & (df["mode"] == 0)]
    top = ref.sort_values("rho_i", ascending=False).groupby(CFG).head(3)[CFG + ["feature_idx"]]
    top["rank"] = top.groupby(CFG).cumcount()
    q = df[(df["mode"] == 0)].merge(top, on=CFG + ["feature_idx"], how="inner")
    return q


def fig_translation(df, s, path, m=16, ratios=(8, 32, 128)):
    q = top3_table(df, s)
    fig, axes = plt.subplots(2, len(ratios), figsize=(3.2 * len(ratios), 5), squeeze=False)
    for j, r in enumerate(ratios):
        ax = axes[0, j]
        g = q[(q.m == m) & (q.n == m * r)]
        for p, gp in g.groupby("p"):
            mean = gp.groupby("bits").rho_i.mean(); sd = gp.groupby("bits").rho_i.std()
            b = mean.index.values; b = np.where(np.isinf(b), 9.5, b)
            ax.errorbar(b, mean.values, yerr=sd.values, fmt="-o", ms=3, lw=1.1, capsize=2, color=P_COL.get(p, COL["gray"]), label=f"p={p}")
        ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9.5]); ax.set_xticklabels(["2", "3", "4", "5", "6", "7", "8", "fp32"])
        ax.set_title(f"top-3 features (fp32-ranked), m={m}, n/m={r}", fontsize=8); ax.set_ylim(-0.05, 1.05)
        if j == 0: ax.set_ylabel("ρ of top-3 features")
        ax.legend(fontsize=6, frameon=False)
        # bottom: sorted rho profiles at each bit-width for one p (the one with the largest F)
        ax = axes[1, j]
        gg = df[(df.m == m) & (df.n == m * r) & (df["mode"] == 0) & (df.seed == 0)]
        if len(gg):
            p_sel = gg.groupby("p").apply(lambda t: (t[np.isinf(t.bits)].w_norm >= 0.5).sum()).idxmax()
            gg = gg[gg.p == p_sel]
            bits_list = sorted(gg.bits.unique())
            for bi, b in enumerate(bits_list):
                prof = np.sort(gg[gg.bits == b].rho_i.values)[::-1]
                ax.plot(np.arange(len(prof)), prof, lw=1, color=plt.cm.Blues(0.25 + 0.75 * bi / max(1, len(bits_list) - 1)),
                        label=("fp32" if np.isinf(b) else f"b={int(b)}"))
            ax.axhline(0.5, color=COL["gray"], lw=0.6, ls="--"); ax.set_ylim(-0.1, 1.05)
            ax.set_xlabel(f"feature rank (sorted by ρ), p={p_sel}, seed 0"); ax.legend(fontsize=5.5, ncol=2, frameon=False)
            if j == 0: ax.set_ylabel("ρ_i")
    fig.suptitle("E4: translation (profile shifts, top features unchanged) vs tilt (top features degrade)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def run_e1(exp="E1"):
    os.makedirs(FIGS, exist_ok=True)
    df = load(exp)
    s = model_summary(df)
    s = add_soft_live(df, s, np.isinf(df.bits) & (df["mode"] == 0), CFG)
    s = attach_fp32_ref(s)
    s.to_csv(os.path.join(RES, f"{exp}_model_summary.csv"), index=False)
    out = {}
    for curve, frac in [("R_live", 0.95), ("R_live", 0.90), ("R_live", 0.99), ("R_soft", 0.95), ("R_softc", 0.95), ("R_hard", 0.95), ("R_hard25", 0.95), ("R_hard75", 0.95)]:
        tab = bcrit_table(s, curve=curve, frac=frac)
        if len(tab) == 0:
            continue
        fit = fit_slope_bootstrap(tab)
        tab_kc = tab.assign(b_crit=tab.b_crit - np.log2(tab.kappa_inf))
        fit_kc = fit_slope_bootstrap(tab_kc)
        cm = config_means(tab)
        out[(curve, frac)] = dict(fit=fit, fit_kc=fit_kc, cm=cm, tab=tab)
        tag = f"{curve}_{int(frac*100)}"
        cm.to_csv(os.path.join(RES, f"{exp}_bcrit_{tag}.csv"), index=False)
        fig_money(cm, fit, fit_kc, f"{exp}: b_crit ({curve} ≥ {frac:.2f}) vs log2 F", os.path.join(FIGS, f"fig1_bcrit_vs_F_{tag}.png"))
        print(f"{exp} {tag}: slope {fit['slope']:.3f} [{fit['lo']:.3f},{fit['hi']:.3f}] R²={fit['r2']:.2f} n_cfg={fit['n_cfg']} | "
              f"κ-corr slope {fit_kc['slope']:.3f} [{fit_kc['lo']:.3f},{fit_kc['hi']:.3f}]")
    # continuous axis (b_eff from measured residual)
    tab = bcrit_table(s, curve="R_live", frac=0.95, axis="b_eff")
    if len(tab):
        fit = fit_slope_bootstrap(tab); print(f"{exp} R_live_95 on b_eff axis: slope {fit['slope']:.3f} [{fit['lo']:.3f},{fit['hi']:.3f}] n_cfg={fit['n_cfg']}")
        out[("R_live_beff", 0.95)] = dict(fit=fit, cm=config_means(tab), tab=tab)
    fig_curves(s, os.path.join(FIGS, "fig2_Nret_vs_bits.png"), curve="R_live", title="retention R_live(b) = Σ_{i live at fp32} ρ_i(b) / same at fp32")
    fig_curves(s, os.path.join(FIGS, "fig2b_hard_Nret_vs_bits.png"), curve="R_hard", title="hard retention N_ret(b)/N_ret(fp32), ρ ≥ 0.5")
    fig_translation(df, s, os.path.join(FIGS, "fig4_translation_vs_tilt.png"))
    return df, s, out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "E1"
    if what == "E0":
        fig_e0(load("E0"), os.path.join(FIGS, "fig0_e0_replication.png")); print("E0 figure written")
    else:
        run_e1(what)
