"""Per-channel regime investigation (v2 Phase 1, 2026-10-11): why does kappa_eff not predict
b_crit under per-channel scaling (fig7b: slope -0.01 [-0.23,+0.15], R^2=0.00)?

Three questions, all on existing results plus fp32 GPT-2 weight statistics (no new damage runs):
  1. Does the kappa law predict the per-tensor -> per-channel LEVEL drop?
  2. Is the within-regime null a range-restriction / noise-floor effect?
  3. Is kappa_eff (norm-weighted mean of per-channel kappa) the wrong statistic?
     Candidates: bulk aggregates of per-output-channel kappa_c, and tail statistics,
     including group-wise (g=128 along d_in) kappa with max over groups.
Exploratory: nine statistics are tried; ALL are reported (appendix table in main_v5.tex).
"""
import os
import numpy as np, pandas as pd, torch
from transformers import GPT2LMHeadModel

RES = os.path.join(os.path.dirname(__file__), "..", "results")
MATS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
GROUP = 128


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([np.ones(len(x)), x]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ c
    return c[1], c[0], 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def boot(x, y, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 2:
            continue
        s.append(ols(x[i], y[i])[0])
    return np.percentile(s, [2.5, 97.5])


def weight_stats():
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
    rows = []
    for L, blk in enumerate(model.transformer.h):
        for name in MATS:
            m = blk
            for p in name.split("."):
                m = getattr(m, p)
            W = m.weight.detach()                                   # Conv1D: [d_in, d_out]
            kc = (W.abs().amax(dim=0) / W.pow(2).mean(dim=0).sqrt()).numpy()
            ng = W.shape[0] // GROUP
            Wg = W[: ng * GROUP].reshape(ng, GROUP, -1)
            kg = (Wg.abs().amax(dim=1) / Wg.pow(2).mean(dim=1).sqrt()).numpy().ravel()
            rows.append(dict(block=L, matrix=name,
                             kc_mean=kc.mean(), kc_med=float(np.median(kc)),
                             kc_p90=float(np.percentile(kc, 90)), kc_max=float(kc.max()),
                             kc_log2mean=float(2 ** np.log2(kc).mean()),
                             kg_med=float(np.median(kg)), kg_p90=float(np.percentile(kg, 90)),
                             kg_max=float(kg.max())))
    return pd.DataFrame(rows)


def run(thr="0.05"):
    t = pd.read_csv(os.path.join(RES, "lm_layer_thresholds_pc.csv")).merge(weight_stats(), on=["block", "matrix"])
    pt = pd.read_csv(os.path.join(RES, "lm_layer_thresholds.csv"))
    m = t.merge(pt[["block", "matrix", f"b_rtn_{thr}", f"censor_rtn_{thr}"]], on=["block", "matrix"])

    # 1. level drop: observed vs 0.63*(log2 kappa - log2 kappa_eff)
    both = m[(m[f"censor_pc_{thr}"] == "ok") & (m[f"censor_rtn_{thr}"] == "ok")]
    drop_obs = both[f"b_rtn_{thr}"] - both[f"b_pc_{thr}"]
    drop_pred = 0.63 * (np.log2(both.kappa) - np.log2(both.kappa_eff))
    print(f"level drop (n={len(both)}): observed {drop_obs.mean():.2f}±{drop_obs.std():.2f}, "
          f"predicted {drop_pred.mean():.2f}±{drop_pred.std():.2f}, r={np.corrcoef(drop_obs, drop_pred)[0,1]:.2f}")

    # 2+3. within-regime slopes for every candidate statistic
    ok = t[t[f"censor_pc_{thr}"] == "ok"]
    y = ok[f"b_pc_{thr}"]
    out = []
    for c in ["kappa_eff", "kc_mean", "kc_med", "kc_p90", "kc_log2mean", "kc_max", "kg_med", "kg_p90", "kg_max"]:
        x = np.log2(ok[c])
        sl, _, r2 = ols(x, y)
        lo, hi = boot(x, y)
        out.append(dict(stat=c, slope=sl, lo=lo, hi=hi, r2=r2, log2_range=float(np.ptp(np.log2(t[c])))))
        print(f"  b_pc ~ log2 {c:11s}: slope {sl:+.3f} [{lo:+.3f},{hi:+.3f}] R2={r2:.3f} range={out[-1]['log2_range']:.2f}")
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(RES, "lm_pc_stats.csv"), index=False)

    # noise floor: reliability of b_pc across adjacent thresholds
    ok2 = t[(t["censor_pc_0.02"] == "ok") & (t["censor_pc_0.05"] == "ok")]
    print(f"reliability corr(b_pc@0.02, b_pc@0.05) = {np.corrcoef(ok2['b_pc_0.02'], ok2['b_pc_0.05'])[0,1]:.3f}")
    return df


if __name__ == "__main__":
    run()
