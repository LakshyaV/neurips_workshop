"""Analysis for the controls: E3 (any-damage null, post-hoc), E2 (lattice), E5 (ordering), E6 (kappa)."""
import os, sys, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analysis import (load, model_summary, attach_fp32_ref, add_soft_live, crossing, bcrit_table, config_means, fit_slope_bootstrap,
                       ols, predicted_bcrit, COL, P_COL, M_MK, RES, FIGS, CFG, fig_money)
from .metrics import prefix_fraction

FAM_COL = {"ptq_uniform": COL["blue"], "ptq_dither": COL["orange"], "ptq_perchannel": COL["aqua"], "ptq_stochastic": COL["yellow"],
           "gaussian": COL["magenta"], "rank": COL["green"], "dropout": COL["violet"], "qat_uniform": COL["ink"]}
FAM_ORDER = ["qat_uniform", "ptq_uniform", "ptq_dither", "ptq_perchannel", "ptq_stochastic", "gaussian", "rank", "dropout"]
MODE_COL = {0: COL["blue"], 3: COL["orange"], 4: COL["yellow"], 5: COL["aqua"]}
MODE_LAB = {0: "uniform (shared lattice)", 3: "dithered lattice", 4: "stochastic rounding", 5: "per-channel"}


# ------------------------------------------------------------------------------------------ E3
def e3_tables(frac=0.95, e3_exp="E3", ref_exp="E1"):
    """Per (family, config, seed, draw): sigma_rel at the R_soft = frac crossing, on the -log2(sigma_rel) axis."""
    df3 = load(e3_exp)
    e3 = model_summary(df3)
    keys3 = ["damage_type", "m", "n", "p", "seed"]
    e3 = add_soft_live(df3, e3, np.isinf(df3.knob), keys3)
    # paired reference: the undamaged copy evaluated inside the same family stack (identical samples)
    ref = e3[np.isinf(e3.knob)][keys3 + ["soft", "soft_live", "nret_50", "live", "kappa"]]
    ref = ref.rename(columns=dict(soft="soft_inf", soft_live="soft_live_inf", nret_50="nret_50_inf", live="live_inf", kappa="kappa_inf"))
    s = e3[~np.isinf(e3.knob)].merge(ref, on=keys3, how="inner")
    s["R_soft"] = s.soft / s.soft_inf
    s["R_live"] = s.soft_live / s.soft_live_inf
    curve = "R_live"
    s["F"] = s.live_inf / s.m
    s["x"] = -np.log2(s.sigma_rel.clip(lower=1e-9))
    rows = []
    for (fam, m, n, p, seed, draw), g in s.groupby(["damage_type", "m", "n", "p", "seed", "draw"]):
        xc, cen = crossing(g.x.values, g[curve].values, frac, xmin=-np.inf, xmax=np.inf)
        rows.append(dict(damage_type=fam, m=m, n=n, p=p, seed=seed, draw=draw, x_crit=xc, sigma_crit=2.0 ** (-xc) if np.isfinite(xc) else np.nan,
                         censor=cen, F=g.F.iloc[0], kappa_inf=g.kappa_inf.iloc[0]))
        # bits-axis b_crit for the ptq families
    tab = pd.DataFrame(rows)
    # PTQ on the bits axis (comparable to E1's QAT b_crit)
    rows = []
    for (fam, m, n, p, seed, draw), g in s[s.damage_type.str.startswith("ptq")].groupby(["damage_type", "m", "n", "p", "seed", "draw"]):
        bc, cen = crossing(g.knob.values, g[curve].values, frac)
        rows.append(dict(damage_type=fam, m=m, n=n, p=p, seed=seed, draw=draw, b_crit=bc, censor=cen, F=g.F.iloc[0], kappa_inf=g.kappa_inf.iloc[0]))
    tab_bits = pd.DataFrame(rows)
    return s, tab, tab_bits


def slope_by_family(tab, ycol="x_crit", sign=-1.0):
    """OLS of (sign * ycol) on log2 F per family over configs with all seeds ok, with a seed bootstrap.
    For x_crit = -log2 sigma_crit the noise-law prediction is log2 sigma_crit = -1/2 log2 F, i.e. slope(x_crit) = +1/2."""
    out = {}
    for fam, t in tab.groupby("damage_type"):
        t2 = t.rename(columns={ycol: "b_crit"}).assign(mode=0, alpha=0.0, kclip=np.inf)
        t2 = t2.groupby(["damage_type", "m", "n", "p", "seed", "mode", "alpha", "kclip", "F", "kappa_inf"], as_index=False).agg(
            b_crit=("b_crit", "mean"), censor=("censor", lambda c: "ok" if (c == "ok").all() else ("low" if (c == "low").any() else "high")))
        out[fam] = fit_slope_bootstrap(t2, keys=("m", "n", "p"))
    return out


def fig_e3(s, tab, tab_bits, e1_cm, path, frac=0.95):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    ax = axes[0]
    fits = slope_by_family(tab)
    for fam in FAM_ORDER:
        if fam not in fits or fam == "qat_uniform":
            continue
        t = tab[(tab.damage_type == fam)]
        cm = t[t.censor == "ok"].groupby(["m", "n", "p"]).agg(F=("F", "mean"), y=("x_crit", "mean"), sd=("x_crit", "std"), nok=("x_crit", "size")).reset_index()
        tot = t.groupby(["m", "n", "p"]).size().reset_index(name="ntot")
        cm = cm.merge(tot, on=["m", "n", "p"])
        cm = cm[cm.nok == cm.ntot]
        f = fits[fam]
        ax.errorbar(np.log2(cm.F), -cm.y, yerr=cm.sd, fmt="o", ms=3.5, color=FAM_COL[fam], capsize=1.5, lw=0.8, alpha=0.9)
        if np.isfinite(f["slope"]):
            xs = np.linspace(np.log2(cm.F.min()) - 0.2, np.log2(cm.F.max()) + 0.2, 20)
            ax.plot(xs, -(f["intercept"] + f["slope"] * xs), color=FAM_COL[fam], lw=1.4,
                    label=f"{fam}: slope {-f['slope']:.2f} [{-f['hi']:.2f}, {-f['lo']:.2f}]")
    ax.set_xlabel("log2 F"); ax.set_ylabel(f"log2 σ_rel at R_live = {frac:.2f}   (σ_rel = ‖D(W)−W‖/‖W‖)")
    ax.set_title("E3: damage threshold vs superposition, post-hoc on fp32 weights\n(noise-law prediction: slope −½)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    # panel b: R_soft vs sigma_rel curves overlaid for one config (largest F with a clean crossing)
    ax = axes[1]
    ok = tab[tab.censor == "ok"].groupby(["m", "n", "p"]).F.mean()
    sel = ok.idxmax() if len(ok) else None
    if sel is not None:
        m, n, p = sel
        g = s[(s.m == m) & (s.n == n) & (s.p == p)]
        for fam in FAM_ORDER:
            gg = g[g.damage_type == fam]
            if len(gg) == 0:
                continue
            mean = gg.groupby("knob").agg(sr=("sigma_rel", "mean"), R=("R_live", "mean"), sd=("R_live", "std")).sort_values("sr")
            ax.plot(mean.sr, mean.R, "-o", ms=2.5, lw=1.1, color=FAM_COL[fam], label=fam)
        ax.axhline(frac, color=COL["gray"], lw=0.8, ls="--"); ax.set_xscale("log")
        ax.set_xlabel("σ_rel"); ax.set_ylabel("R_live"); ax.set_title(f"retention vs damage size, m={m}, n={n}, p={p} (F≈{ok[sel]:.1f})", fontsize=8)
        ax.legend(fontsize=6, frameon=False); ax.set_ylim(-0.05, 1.1)
    # panel c: bits axis: QAT (E1) vs PTQ families
    ax = axes[2]
    if e1_cm is not None:
        ok1 = e1_cm[e1_cm.n_ok == e1_cm.n_seeds]
        ax.errorbar(np.log2(ok1.F), ok1.b_crit, yerr=ok1.b_sd, fmt="o", ms=3.5, color=FAM_COL["qat_uniform"], capsize=1.5, lw=0.8, label="QAT uniform (E1)")
        sl, ic, _ = ols(np.log2(ok1.F), ok1.b_crit); xs = np.linspace(np.log2(ok1.F.min()), np.log2(ok1.F.max()), 10)
        ax.plot(xs, ic + sl * xs, color=FAM_COL["qat_uniform"], lw=1.2)
    for fam in ["ptq_uniform", "ptq_dither", "ptq_perchannel", "ptq_stochastic"]:
        t = tab_bits[tab_bits.damage_type == fam]
        cm = t[t.censor == "ok"].groupby(["m", "n", "p"]).agg(F=("F", "mean"), y=("b_crit", "mean"), sd=("b_crit", "std"), nok=("b_crit", "size")).reset_index()
        tot = t.groupby(["m", "n", "p"]).size().reset_index(name="ntot"); cm = cm.merge(tot, on=["m", "n", "p"]); cm = cm[cm.nok == cm.ntot]
        if len(cm) >= 3:
            sl, ic, _ = ols(np.log2(cm.F), cm.y); xs = np.linspace(np.log2(cm.F.min()), np.log2(cm.F.max()), 10)
            ax.errorbar(np.log2(cm.F), cm.y, yerr=cm.sd, fmt="o", ms=3.5, color=FAM_COL[fam], capsize=1.5, lw=0.8, label=f"{fam} (slope {sl:.2f})")
            ax.plot(xs, ic + sl * xs, color=FAM_COL[fam], lw=1.2)
    ax.set_xlabel("log2 F"); ax.set_ylabel("b_crit (bits)"); ax.set_title("bits axis: QAT vs post-training quantizers", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return fits


def run_e3(e1_out=None, e3_exp="E3", ref_exp="E1", figname="fig3_anydamage_null.png"):
    s, tab, tab_bits = e3_tables(e3_exp=e3_exp, ref_exp=ref_exp)
    tab.to_csv(os.path.join(RES, f"{e3_exp}_sigma_crit.csv"), index=False); tab_bits.to_csv(os.path.join(RES, f"{e3_exp}_bcrit_bits.csv"), index=False)
    e1_cm = None
    if os.path.exists(os.path.join(RES, "E1_bcrit_R_soft_95.csv")):
        e1_cm = pd.read_csv(os.path.join(RES, "E1_bcrit_R_soft_95.csv"))
    fits = fig_e3(s, tab, tab_bits, e1_cm, os.path.join(FIGS, figname))
    for fam, f in fits.items():
        print(f"E3 {fam:15s}: slope of log2 σ_crit vs log2 F = {-f['slope']:.3f}  CI [{-f['hi']:.3f}, {-f['lo']:.3f}]  n_cfg={f['n_cfg']}")
    bfits = {}
    for fam, t in tab_bits.groupby("damage_type"):
        t2 = t.groupby(["damage_type", "m", "n", "p", "seed", "F", "kappa_inf"], as_index=False).agg(b_crit=("b_crit", "mean"), censor=("censor", lambda c: "ok" if (c == "ok").all() else "low" if (c == "low").any() else "high")).assign(mode=0, alpha=0.0, kclip=np.inf)
        bfits[fam] = fit_slope_bootstrap(t2, keys=("m", "n", "p"))
        print(f"E3 {fam:15s}: b_crit (bits axis) slope vs log2 F = {bfits[fam]['slope']:.3f} [{bfits[fam]['lo']:.3f}, {bfits[fam]['hi']:.3f}] n_cfg={bfits[fam]['n_cfg']}")
    return s, tab, tab_bits, fits, bfits


# ------------------------------------------------------------------------------------------ E2
def run_e2():
    df = load("E2")
    s = attach_fp32_ref(add_soft_live(df, model_summary(df), np.isinf(df.bits) & (df["mode"] == 0), CFG))
    tab = bcrit_table(s, curve="R_live", frac=0.95, arm_cols=("mode",))
    fits = {mo: fit_slope_bootstrap(tab[tab["mode"] == mo]) for mo in sorted(tab["mode"].unique())}
    cm = config_means(tab)
    cm.to_csv(os.path.join(RES, "E2_bcrit_by_mode.csv"), index=False)
    ratios = sorted((s.n // s.m).unique()); ps = sorted(s.p.unique())
    fig, axes = plt.subplots(len(ps), len(ratios) + 1, figsize=(2.3 * (len(ratios) + 1), 2.1 * len(ps)), squeeze=False)
    for i, p in enumerate(ps):
        for j, r in enumerate(ratios):
            ax = axes[i, j]
            g = s[(s.p == p) & (s.n == s.m * r) & ~np.isinf(s.bits)]
            for mo, gm in g.groupby("mode"):
                mean = gm.groupby("bits").R_live.mean(); sd = gm.groupby("bits").R_live.std()
                ax.errorbar(mean.index, mean.values, yerr=sd.values, fmt="-o", ms=2.5, lw=1, capsize=1.5, color=MODE_COL.get(mo, COL["gray"]), label=MODE_LAB.get(mo, str(mo)))
            ax.axhline(0.95, color=COL["gray"], lw=0.7, ls="--"); ax.set_ylim(-0.05, 1.1)
            ax.set_title(f"p={p}, n/m={r}", fontsize=7)
            if i == len(ps) - 1: ax.set_xlabel("bits")
            if j == 0: ax.set_ylabel("R_live")
        ax = axes[i, -1]; ax.axis("off")
    axes[0, 0].legend(fontsize=5.5, frameon=False)
    # last column: b_crit vs log2F per mode (single panel spanning)
    ax = fig.add_subplot(1, len(ratios) + 1, len(ratios) + 1)
    for mo in sorted(cm["mode"].unique()):
        c = cm[(cm["mode"] == mo) & (cm.n_ok == cm.n_seeds)]
        f = fits[mo]
        ax.errorbar(np.log2(c.F), c.b_crit, yerr=c.b_sd, fmt="o", ms=3, color=MODE_COL.get(mo, COL["gray"]), capsize=1.5, lw=0.8,
                    label=f"{MODE_LAB.get(mo, mo)}: slope {f['slope']:.2f} [{f['lo']:.2f},{f['hi']:.2f}]")
        if np.isfinite(f["slope"]):
            xs = np.linspace(np.log2(c.F.min()), np.log2(c.F.max()), 10); ax.plot(xs, f["intercept"] + f["slope"] * xs, color=MODE_COL.get(mo, COL["gray"]), lw=1.2)
    ax.set_xlabel("log2 F"); ax.set_ylabel("b_crit"); ax.legend(fontsize=5.5, frameon=False); ax.set_title("E2: b_crit by quantizer", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig5_lattice.png"), bbox_inches="tight"); plt.close(fig)
    for mo, f in fits.items():
        print(f"E2 mode {mo} ({MODE_LAB.get(mo, mo)}): slope {f['slope']:.3f} [{f['lo']:.3f},{f['hi']:.3f}] n_cfg={f['n_cfg']}")
    # paired differences: b_crit(mode) - b_crit(uniform) per config+seed
    piv = tab.pivot_table(index=["m", "n", "p", "seed"], columns="mode", values="b_crit")
    for mo in piv.columns:
        if mo != 0 and 0 in piv.columns:
            d = (piv[mo] - piv[0]).dropna()
            print(f"E2 paired Δb_crit({MODE_LAB.get(mo, mo)} − uniform): mean {d.mean():+.3f} ± {d.std()/max(1,np.sqrt(len(d))):.3f} (n={len(d)})")
    return s, tab, cm, fits, piv


# ------------------------------------------------------------------------------------------ E5
def run_e5():
    df = load("E5")
    s = attach_fp32_ref(add_soft_live(df, model_summary(df), np.isinf(df.bits) & (df["mode"] == 0), CFG))
    rows = []
    for rid, g in df.groupby("run_id", sort=False):
        g = g.sort_values("feature_idx")
        import torch
        r = torch.from_numpy((g.rho_i.values >= 0.5))
        strict, overlap = prefix_fraction(r)
        rows.append(dict(run_id=rid, prefix_strict=strict, prefix_overlap=overlap))
    s = s.merge(pd.DataFrame(rows), on="run_id")
    ref = s[np.isinf(s.bits)][CFG + ["prefix_strict", "prefix_overlap"]].rename(columns=dict(prefix_strict="prefix_strict_inf", prefix_overlap="prefix_overlap_inf"))
    s = s.merge(ref, on=CFG, how="left")
    s.to_csv(os.path.join(RES, "E5_model_summary.csv"), index=False)
    alphas = sorted(s.alpha.unique()); ratios = sorted((s.n // s.m).unique())
    fig, axes = plt.subplots(2, len(ratios), figsize=(3.4 * len(ratios), 5.2), squeeze=False)
    for j, r in enumerate(ratios):
        for row, col in [(0, "prefix_overlap"), (1, "nret_50")]:
            ax = axes[row, j]
            g = s[(s.n == s.m * r)]
            for (a, p), ga in g.groupby(["alpha", "p"]):
                mean = ga.groupby("bits")[col].mean(); sd = ga.groupby("bits")[col].std()
                b = mean.index.values.astype(float); b = np.where(np.isinf(b), 9.5, b)
                ax.errorbar(b, mean.values, yerr=sd.values, fmt="-o" if p == sorted(s.p.unique())[0] else "--s", ms=2.5, lw=1, capsize=1.5,
                            color=[COL["blue"], COL["orange"], COL["aqua"]][alphas.index(a)], label=f"α={a}, p={p}")
            ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9.5]); ax.set_xticklabels(["2", "3", "4", "5", "6", "7", "8", "fp32"])
            ax.set_title(f"n/m={r}: " + ("prefix fraction (top-|R| overlap)" if row == 0 else "N_ret (ρ ≥ 0.5)"), fontsize=8)
            if row == 0: ax.set_ylim(0, 1.05)
            ax.legend(fontsize=5, frameon=False, ncol=2)
    fig.suptitle("E5: power-law importance I_i = (i+1)^−α — features are lost in importance order", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig6_ordering.png"), bbox_inches="tight"); plt.close(fig)
    summ = s[~np.isinf(s.bits) & (s.bits >= 3)].groupby(["alpha", "n", "p", "bits"]).agg(prefix_overlap=("prefix_overlap", "mean"), prefix_inf=("prefix_overlap_inf", "mean"),
                                                                                        prefix_strict=("prefix_strict", "mean"), strict_inf=("prefix_strict_inf", "mean"), nret=("nret_50", "mean"), nret_inf=("nret_50_inf", "mean")).reset_index()
    summ["drop"] = summ.prefix_inf - summ.prefix_overlap
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"))
    tab_h = bcrit_table(s, curve="R_hard", frac=0.95, extra_keys=()); tab_s = bcrit_table(s, curve="R_live", frac=0.95)
    for nm, t in [("hard", tab_h), ("soft", tab_s)]:
        f = fit_slope_bootstrap(t, keys=("m", "n", "p", "alpha")); print(f"E5 b_crit ({nm}) slope vs log2 F over (n,p,α): {f['slope']:.3f} [{f['lo']:.3f},{f['hi']:.3f}] n_cfg={f['n_cfg']}")
    return s, summ


# ------------------------------------------------------------------------------------------ E6
def run_e6():
    df = load("E6")
    s = attach_fp32_ref(add_soft_live(df, model_summary(df), np.isinf(df.bits) & (df["mode"] == 0), CFG))   # ref has the same kclip
    tab = bcrit_table(s, curve="R_live", frac=0.95)
    cm = config_means(tab)
    cm.to_csv(os.path.join(RES, "E6_bcrit_by_kclip.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    ax = axes[0]
    slopes = []
    for (m, n, p), g in cm.groupby(["m", "n", "p"]):
        g = g[g.n_ok == g.n_seeds].sort_values("kappa")
        if len(g) >= 2:
            sl, ic, _ = ols(np.log2(g.kappa), g.b_crit); slopes.append(dict(m=m, n=n, p=p, slope=sl, n_pts=len(g), F=g.F.mean()))
            ax.errorbar(np.log2(g.kappa), g.b_crit, yerr=g.b_sd, fmt="-o", ms=3.5, lw=1, capsize=1.5, color=P_COL.get(p, COL["gray"]),
                        label=f"n={n}, p={p} (F≈{g.F.mean():.0f}): slope {sl:.2f}")
    ax.set_xlabel("log2 κ (measured, fp32 arm; κ moved by weight clipping at c·rms)"); ax.set_ylabel("b_crit"); ax.legend(fontsize=6, frameon=False)
    ax.set_title("E6: κ enters as log2 κ (predicted slope 1)", fontsize=8)
    ax = axes[1]
    ok = cm[cm.n_ok == cm.n_seeds]
    pred = predicted_bcrit(ok.F, ok.kappa)
    ax.scatter(pred, ok.b_crit, s=16, c=[P_COL.get(p, COL["gray"]) for p in ok.p])
    lim = [min(pred.min(), ok.b_crit.min()) - 0.3, max(pred.max(), ok.b_crit.max()) + 0.3]
    ax.plot(lim, lim, "--", color=COL["gray"], lw=1); sl, ic, r2 = ols(pred, ok.b_crit); ax.plot(np.array(lim), ic + sl * np.array(lim), color=COL["ink"], lw=1, label=f"fit slope {sl:.2f}, R²={r2:.2f}")
    ax.set_xlabel("predicted 1 + log2(1 + κ√(F/3))"); ax.set_ylabel("measured b_crit"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig8_kappa.png"), bbox_inches="tight"); plt.close(fig)
    sl = pd.DataFrame(slopes); print(sl.to_string()); print(f"E6 mean slope of b_crit on log2 κ: {sl.slope.mean():.3f} ± {sl.slope.std()/max(1,np.sqrt(len(sl))):.3f}")
    return s, tab, cm, sl


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "E3_pilot":
        run_e3(e3_exp="E3_pilot", ref_exp="pilot", figname="fig3_pilot_anydamage.png")
    else:
        {"E3": run_e3, "E2": run_e2, "E5": run_e5, "E6": run_e6}[what]()
