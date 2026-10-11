"""Preregistered scorer for Phase 2 predictions P4-P9 (PREDICTIONS_V2.md is authoritative;
the BANDS dict below must mirror it and is committed BEFORE any 8B damage data exists).

Each section reads its run output if present and prints the preregistered verdict; missing
inputs are skipped, so this can be run piecewise as arms finish. Conventions mirror v1:
dNLL threshold 0.05, log-log/log-linear crossings as in analysis_lm/validate_port, OLS slope
with a 2,000-resample matrix bootstrap, sklearn ROC-AUC.

Inputs (defaults; override with flags):
  P4/P5/P7: results/llama31_8b_layers.parquet   (lm_eval2, families gaussian,rtn,rtn_gw, --draws 2)
  P6:       results/llama31_8b_rot_layers.parquet (same, rotated model, family rtn)
            + results/llama31_8b_rot_kappa.csv  (rotate.kappa_report: block, matrix,
              kappa_before, kappa_after)
  P8:       results/llama31_8b_act.parquet      (lm_act2)
  P9a:      results/llama31_8b_tokens.parquet   (lm_tokens2, incl. rtn_gw 4)
  P9b:      results/llama31_8b_sae.parquet      (lm_sae2, incl. rtn_gw 4)
  P9c:      from the P8 thresholds
"""
import argparse, math, os
import numpy as np, pandas as pd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
THR = 0.05

BANDS = dict(
    P4_confirm=(0.4, 0.9), P4_dead=0.2,
    P5_e=(0.2, 0.7),
    P7_slope=(0.3, 0.9), P7_dr2=0.10, P7_power_log2range=1.0,
    P8_slope=(0.8, 1.4),
    P9a_auc=0.75, P9b_auc=0.80, P9c_auc=0.85,
)
NBOOT = 2000


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (sl, ic), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - A @ [sl, ic]) ** 2).sum() / ss if ss > 0 else np.nan
    return sl, ic, r2


def boot_slope(x, y, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    sl = []
    for _ in range(NBOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) > 1:
            sl.append(ols(x[i], y[i])[0])
    return np.percentile(sl, [2.5, 97.5])


def boot_mean(v, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NBOOT)]
    return np.percentile(m, [2.5, 97.5])


def rtn_crossing(bits, dnll, thr=THR):
    """b_crit: crossing of dnll through thr as bits DECREASE (log-linear, as validate_port)."""
    xb = -np.asarray(bits, float)
    o = np.argsort(xb)
    xb, yb = xb[o], np.asarray(dnll, float)[o]
    if yb[0] >= thr:
        return -xb[0], "low"
    for i in range(len(xb) - 1):
        if yb[i] < thr <= yb[i + 1]:
            y0 = max(yb[i], 1e-6)
            t = (math.log(thr) - math.log(y0)) / (math.log(yb[i + 1]) - math.log(y0))
            return -(xb[i] + t * (xb[i + 1] - xb[i])), "ok"
    return -xb[-1], "high"


def loglog_crossing(x, y, thr=THR):
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    if y[0] >= thr:
        return x[0], "low"
    for i in range(len(x) - 1):
        if y[i] < thr <= y[i + 1]:
            y0 = max(y[i], 1e-6)
            t = (math.log(thr) - math.log(y0)) / (math.log(y[i + 1]) - math.log(y0))
            return float(np.exp(math.log(x[i]) + t * (math.log(x[i + 1]) - math.log(x[i])))), "ok"
    return x[-1], "high"


def thresholds(df):
    """Per-matrix b_rtn / b_gw / sigma_crit from a lm_eval2 parquet (draws averaged first)."""
    rows = []
    for (L, name), g in df.groupby(["block", "matrix"]):
        r = dict(block=L, matrix=name)
        for c in ("kappa", "kappa_eff", "kc_max", "kc_med", "kc_mean", "kg_med", "kg_p90", "kg_max"):
            if c in g.columns:
                r[c] = g[c].iloc[0]
        gr = g[g.family == "rtn"]
        if len(gr):
            r["b_rtn"], r["cen_rtn"] = rtn_crossing(gr.knob.values, gr.dnll.values)
        gw = g[g.family == "rtn_gw"]
        if len(gw):
            r["b_gw"], r["cen_gw"] = rtn_crossing(gw.knob.values, gw.dnll.values)
        ga = g[g.family == "gaussian"]
        if len(ga):
            m = ga.groupby("knob").agg(dnll=("dnll", "mean"), sigma_rel=("sigma_rel", "mean"))
            r["sigma_crit"], r["cen_g"] = loglog_crossing(m.sigma_rel.values, m.dnll.values)
        rows.append(r)
    return pd.DataFrame(rows)


def in_band(v, band):
    return band[0] <= v <= band[1]


def sel(th, cen_col, val_col):
    """Uncensored rows, or an empty frame when the arm wasn't run."""
    if cen_col not in th.columns or val_col not in th.columns:
        return th.iloc[0:0]
    return th[th[cen_col] == "ok"].dropna(subset=[val_col])


def score_p4_p5_p7(path):
    df = pd.read_parquet(path)
    th = thresholds(df)
    out = {}
    # ---- P4: per-tensor RTN slope on log2 kappa
    ok = sel(th, "cen_rtn", "b_rtn")
    if len(ok) >= 3:
        x, y = np.log2(ok.kappa), ok.b_rtn
        sl, _, r2 = ols(x, y)
        lo, hi = boot_slope(x, y)
        v = ("CONFIRMED" if in_band(sl, BANDS["P4_confirm"]) else
             "DEAD" if abs(sl) < BANDS["P4_dead"] else "PARTIAL")
        print(f"P4  b_rtn ~ log2 kappa: slope {sl:.3f} [{lo:.3f},{hi:.3f}] R2={r2:.2f} "
              f"(n={len(ok)}, censored {len(th)-len(ok)}) -> {v}")
        out["p4"] = (x.values, y.values, sl)
    else:
        print("P4  SKIP: <3 uncensored per-tensor RTN matrices")
    # ---- P5: gaussian exponent e and (1-e) consistency
    okg = sel(th, "cen_g", "sigma_crit")
    if len(okg) >= 3:
        x, y = np.log2(okg.kappa), np.log2(okg.sigma_crit)
        e = ols(x, y)[0]               # e = d log sigma_crit / d log kappa (PREDICTIONS_V2 P5)
        elo, ehi = boot_slope(x, y)
        band = "in" if in_band(e, BANDS["P5_e"]) else "OUT OF"
        print(f"P5  e = {e:.3f} [{elo:.3f},{ehi:.3f}] (n={len(okg)}) -> {band} band {BANDS['P5_e']}")
        if "p4" in out:
            both = th[(th.cen_rtn == "ok") & (th.cen_g == "ok")]
            xb = np.log2(both.kappa.values)
            rng = np.random.default_rng(1)
            diffs = []
            for _ in range(NBOOT):
                i = rng.integers(0, len(both), len(both))
                if len(np.unique(xb[i])) < 2:
                    continue
                s_rtn = ols(xb[i], both.b_rtn.values[i])[0]
                e_i = ols(xb[i], np.log2(both.sigma_crit.values[i]))[0]
                diffs.append(s_rtn - (1 - e_i))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            v = "CONSISTENT" if lo <= 0 <= hi else "INCONSISTENT"
            print(f"P5  paired (slope_rtn - (1-e)): mean {np.mean(diffs):+.3f} [{lo:+.3f},{hi:+.3f}] -> {v}")
        out["e"] = e
    else:
        print("P5  SKIP: <3 uncensored gaussian matrices")
    # ---- P7: group-wise regime, kg_max vs bulk statistics
    okw = sel(th, "cen_gw", "b_gw")
    if len(okw) >= 3 and "kg_max" in okw:
        rng_log2 = np.ptp(np.log2(okw.kg_max))
        if rng_log2 < BANDS["P7_power_log2range"]:
            print(f"P7  UNDERPOWERED by preregistered gate: log2 range of kg_max = "
                  f"{rng_log2:.2f} < {BANDS['P7_power_log2range']}; descriptive only.")
        x, y = np.log2(okw.kg_max), okw.b_gw
        sl, _, r2 = ols(x, y)
        lo, hi = boot_slope(x, y)
        bulk = {c: ols(np.log2(okw[c]), y)[2] for c in ("kappa_eff", "kc_med", "kc_mean") if c in okw}
        best = max(bulk.values()) if bulk else np.nan
        dr2 = r2 - best
        v = ("CONFIRMED" if in_band(sl, BANDS["P7_slope"]) and dr2 >= BANDS["P7_dr2"]
             else "NOT CONFIRMED")
        if rng_log2 < BANDS["P7_power_log2range"]:
            v = "DESCRIPTIVE (underpowered)"
        print(f"P7  b_gw ~ log2 kg_max: slope {sl:.3f} [{lo:.3f},{hi:.3f}] R2={r2:.2f}; "
              f"best bulk R2={best:.2f} ({max(bulk, key=bulk.get) if bulk else '-'}), "
              f"dR2={dr2:+.2f} (n={len(okw)}) -> {v}")
    else:
        print("P7  SKIP: <3 uncensored group-wise matrices or kg_max missing")
    return out


def score_p6(unrot_path, rot_path, kappa_csv, e_prime):
    dfu, dfr = pd.read_parquet(unrot_path), pd.read_parquet(rot_path)
    thu, thr_ = thresholds(dfu), thresholds(dfr)
    k = pd.read_csv(kappa_csv)
    m = (thu[thu.cen_rtn == "ok"][["block", "matrix", "b_rtn"]]
         .merge(thr_[thr_.cen_rtn == "ok"][["block", "matrix", "b_rtn"]],
                on=["block", "matrix"], suffixes=("_un", "_rot"))
         .merge(k, on=["block", "matrix"]))
    db = m.b_rtn_un - m.b_rtn_rot
    naive = np.log2(m.kappa_before / m.kappa_after)
    comp = (1 - e_prime) * naive
    rn, rc = db - naive, db - comp
    nlo, nhi = boot_mean(rn)
    clo, chi = boot_mean(rc)
    n0, c0 = nlo <= 0 <= nhi, clo <= 0 <= chi
    verdict = ("COMPENSATED CONFIRMED" if c0 and not n0 else
               "NAIVE CONFIRMED (compensation refuted)" if n0 and not c0 else
               "INCONCLUSIVE / BOTH " + ("COVERED" if n0 else "REFUTED"))
    print(f"P6  n={len(m)} paired matrices; e'={e_prime:.3f}")
    print(f"P6  mean(db - naive)       = {rn.mean():+.3f} [{nlo:+.3f},{nhi:+.3f}] bits")
    print(f"P6  mean(db - compensated) = {rc.mean():+.3f} [{clo:+.3f},{chi:+.3f}] bits")
    print(f"P6  -> {verdict}")


def score_p8_p9c(path):
    df = pd.read_parquet(path)
    rows = []
    for (L, name), g in df.groupby(["block", "matrix"]):
        b, cen = rtn_crossing(g.bits.values, g.dnll.values)
        rows.append(dict(block=L, matrix=name, kappa_act=g.kappa_act.iloc[0],
                         kappa_w=g.kappa_w.iloc[0], b_act=b, cen=cen))
    th = pd.DataFrame(rows)
    ok = th[th.cen == "ok"]
    if len(ok) >= 3:
        x, y = np.log2(ok.kappa_act), ok.b_act
        sl, _, r2 = ols(x, y)
        lo, hi = boot_slope(x, y)
        v = "CONFIRMED" if in_band(sl, BANDS["P8_slope"]) else "NOT CONFIRMED"
        print(f"P8  b_act ~ log2 kappa_act: slope {sl:.3f} [{lo:.3f},{hi:.3f}] R2={r2:.2f} "
              f"(n={len(ok)}) -> {v}")
    else:
        print("P8  SKIP: <3 uncensored activation matrices")
    from sklearn.metrics import roc_auc_score
    oka = th[th.cen != "none"]
    y = (oka.b_act > 5).astype(int)
    if y.nunique() == 2:
        auc = roc_auc_score(y, oka.kappa_act)
        base = roc_auc_score(y, oka.kappa_w)
        v = "CONFIRMED" if auc >= BANDS["P9c_auc"] else "NOT CONFIRMED"
        print(f"P9c kappa_act flags b_act>5: AUC {auc:.3f} (baseline kappa_w {base:.3f}, "
              f"rate {y.mean():.2f}, n={len(y)}) -> {v} (band >= {BANDS['P9c_auc']})")
    else:
        print("P9c SKIP: b_act>5 event is degenerate on this run")


def score_p9a(path, col="ok_gw4"):
    from sklearn.metrics import roc_auc_score
    d = pd.read_parquet(path)
    if col not in d:
        print(f"P9a SKIP: no {col} column (need rtn_gw 4 in lm_tokens2 run)")
        return
    c = d[d.ok_fp32.astype(bool)]
    y = (~c[col].astype(bool)).astype(int)
    auc = roc_auc_score(y, -c.margin_fp32)
    base = roc_auc_score(y, -np.log(c.freq.clip(lower=1)))
    v = "CONFIRMED" if auc >= BANDS["P9a_auc"] else "NOT CONFIRMED"
    print(f"P9a margin predicts 4-bit g=128 flips: AUC {auc:.3f} (baseline rarity {base:.3f}, "
          f"flip rate {y.mean():.3f}, n={len(y)}) -> {v} (band >= {BANDS['P9a_auc']})")


def score_p9b(path, cond="gw4"):
    from sklearn.metrics import roc_auc_score
    f = pd.read_parquet(path)
    if f"sum_err_{cond}" not in f:
        print(f"P9b SKIP: no {cond} condition in SAE parquet")
        return
    live = f["n_act0_fp32"] > 20
    f = f[live]
    rho = 1 - f[f"sum_err_{cond}"] / f[f"sum_a0sq_{cond}"].clip(lower=1e-12)
    y = (rho < 0.5).astype(int)
    mean_act = f["sum_a0_fp32"] / f["n_act0_fp32"]
    freq = f["n_act0_fp32"] / f["n_tokens"]
    if y.nunique() < 2:
        print("P9b SKIP: erasure event degenerate")
        return
    auc = roc_auc_score(y, -np.log2(mean_act.clip(lower=1e-12)))
    base = roc_auc_score(y, -np.log2(freq.clip(lower=1e-12)))
    v = "CONFIRMED" if auc >= BANDS["P9b_auc"] else "NOT CONFIRMED"
    print(f"P9b fp32 magnitude predicts erasure at {cond}: AUC {auc:.3f} (baseline freq "
          f"{base:.3f}, erased {y.mean():.3f} of {len(y)} live) -> {v} (band >= {BANDS['P9b_auc']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=os.path.join(RES, "llama31_8b_layers.parquet"))
    ap.add_argument("--rot-layers", default=os.path.join(RES, "llama31_8b_rot_layers.parquet"))
    ap.add_argument("--rot-kappa", default=os.path.join(RES, "llama31_8b_rot_kappa.csv"))
    ap.add_argument("--act", default=os.path.join(RES, "llama31_8b_act.parquet"))
    ap.add_argument("--tokens", default=os.path.join(RES, "llama31_8b_tokens.parquet"))
    ap.add_argument("--sae", default=os.path.join(RES, "llama31_8b_sae.parquet"))
    args = ap.parse_args()
    e = None
    if os.path.exists(args.layers):
        out = score_p4_p5_p7(args.layers)
        e = out.get("e")
    else:
        print("P4/P5/P7 SKIP: no layers parquet")
    if os.path.exists(args.rot_layers) and os.path.exists(args.rot_kappa):
        if e is None:
            print("P6  SKIP: e' unavailable (P5 must run on the unrotated model first)")
        else:
            score_p6(args.layers, args.rot_layers, args.rot_kappa, e)
    else:
        print("P6  SKIP: no rotated run / kappa csv")
    if os.path.exists(args.act):
        score_p8_p9c(args.act)
    else:
        print("P8/P9c SKIP: no activation parquet")
    if os.path.exists(args.tokens):
        score_p9a(args.tokens)
    else:
        print("P9a SKIP: no tokens parquet")
    if os.path.exists(args.sae):
        score_p9b(args.sae)
    else:
        print("P9b SKIP: no SAE parquet")


if __name__ == "__main__":
    main()
