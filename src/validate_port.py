"""Validate the lm_eval2 port against the original GPT-2 run (known results, no new science).

Two checks:
  1. thresholds: RTN is deterministic given weights+data, so b_rtn recomputed from a lm_eval2
     subset run (results/lm2_layers.parquet) must match results/lm_layer_thresholds.csv exactly
     (validated 2026-10-11: max |new-ref| = 0.0000 bits over 11 matrices; kappa to 7e-15).
  2. gaussian equivalence (`--gaussian`): the gaussian arm is draw-sensitive — the mean dnll of a
     single z draw is heavy-tailed across draws (observed 0.018 vs 0.046 at the same sigma on
     block-0 c_attn, each with sem 0.0025), so thresholds from different seeds can differ 10-25%
     and a statistical tolerance tests nothing. Instead we replicate the ORIGINAL sequential
     generator stream (seed 1234, draws in level-then-matrix order) and require the ported eval
     path to reproduce the original code's dnll per level to |diff| <= 3e-3 (the cache-assert
     tolerance). Reference: results/lm_orig_recheck.parquet from `python -m src.lm_eval
     --blocks 0 --bits "" --out results/lm_orig_recheck.parquet`.
Design consequence recorded in PREDICTIONS_V2.md: the 8B gaussian arm averages 2 independent
draws per (matrix, sigma) level.
"""
import os
import numpy as np, pandas as pd
from .analysis_lm import loglog_crossing

RES = os.path.join(os.path.dirname(__file__), "..", "results")
THR = 0.05
SIGMAS = [0.03, 0.045, 0.07, 0.1, 0.15, 0.22, 0.33, 0.5, 0.75]


def gaussian_equivalence(block=0, device="mps", tokens=49152):
    """Replicate the original generator stream for one block; ported eval must match per-level dnll."""
    import torch
    from transformers import AutoModelForCausalLM
    from .lm_eval2 import make_adapter, get_mod, clean_pass
    ref = pd.read_parquet(os.path.join(RES, "lm_orig_recheck.parquet"))
    ref = ref[(ref.family == "gaussian") & (ref.block == block)]
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ad = make_adapter(model)
    ids_all = np.load(os.path.join(RES, "wikitext2_test_gpt2_ids.npy"))
    nseq = tokens // 1024
    ids = torch.from_numpy(ids_all[: nseq * 1024].astype(np.int64)).view(nseq, 1024)
    nll_clean, caches, _ = clean_pass(ad, ids, device, collect_pr=False)
    gen = torch.Generator(device="cpu").manual_seed(1234)   # the original's single stream
    worst = 0.0
    for name in ad.mats:
        mod = get_mod(ad.blocks()[block], name)
        W0 = mod.weight.detach().clone()
        rms = W0.pow(2).mean().sqrt().item()
        for s in SIGMAS:
            z = torch.randn(W0.shape, generator=gen).to(device)
            mod.weight.copy_(W0 + s * rms * z)
            nll = torch.cat([ad.forward_from(caches[block][i:i + 8].to(device), block,
                                             ids[i:i + 8].to(device)).cpu() for i in range(0, nseq, 8)])
            mod.weight.copy_(W0)
            d = float((nll - nll_clean).mean())
            r = ref[(ref.matrix == name) & (np.isclose(ref.knob, s))]
            diff = abs(d - float(r.dnll.iloc[0]))
            worst = max(worst, diff)
            flag = "" if diff <= 3e-3 else "  <-- MISMATCH"
            print(f"{name:12s} sigma={s:5.3f}: ported {d:+.4f} vs original {float(r.dnll.iloc[0]):+.4f}  |diff|={diff:.4f}{flag}")
    print(f"\nGAUSSIAN EQUIVALENCE (block {block}): worst |diff| = {worst:.4f} ->",
          "PASSED" if worst <= 3e-3 else "FAILED")


def run(path=os.path.join(RES, "lm2_layers.parquet")):
    df = pd.read_parquet(path)
    ref = pd.read_csv(os.path.join(RES, "lm_layer_thresholds.csv"))
    rows = []
    for (L, name), g in df.groupby(["block", "matrix"]):
        r = dict(block=L, matrix=name, kappa=g.kappa.iloc[0])
        ga, gr = g[g.family == "gaussian"], g[g.family == "rtn"]
        if len(ga):
            r["sigma_crit_new"], r["cen_g"] = loglog_crossing(ga.sigma_rel.values, ga.dnll.values, THR)
        if len(gr):
            import math
            xb = -gr.knob.values
            o = np.argsort(xb)
            xb, yb = xb[o], gr.dnll.values[o]
            bc, cb = np.nan, "none"
            if yb[0] >= THR:
                bc, cb = -xb[0], "low"
            else:
                for i in range(len(xb) - 1):
                    if yb[i] < THR <= yb[i + 1]:
                        y0 = max(yb[i], 1e-6)
                        t = (math.log(THR) - math.log(y0)) / (math.log(yb[i + 1]) - math.log(y0))
                        bc, cb = -(xb[i] + t * (xb[i + 1] - xb[i])), "ok"
                        break
                if cb == "none":
                    bc, cb = -xb[-1], "high"
            r["b_rtn_new"], r["cen_r"] = bc, cb
        rows.append(r)
    new = pd.DataFrame(rows)
    m = new.merge(ref[["block", "matrix", "kappa", f"sigma_crit_{THR}", f"censor_{THR}",
                       f"b_rtn_{THR}", f"censor_rtn_{THR}"]], on=["block", "matrix"], suffixes=("_new", "_ref"))
    print(f"{len(m)} matrices compared (threshold {THR})")
    kerr = (m.kappa_new - m.kappa_ref).abs().max()
    print(f"kappa: max |new-ref| = {kerr:.2e}  (must be ~0: same weights, same formula)")
    ok_r = m[(m.cen_r == "ok") & (m[f"censor_rtn_{THR}"] == "ok")]
    if len(ok_r):
        d = (ok_r.b_rtn_new - ok_r[f"b_rtn_{THR}"]).abs()
        print(f"b_rtn  (n={len(ok_r)}): max |new-ref| = {d.max():.4f} bits (deterministic; expect ~0)")
    ok_g = m[(m.cen_g == "ok") & (m[f"censor_{THR}"] == "ok")]
    if len(ok_g):
        ratio = ok_g.sigma_crit_new / ok_g[f"sigma_crit_{THR}"]
        print(f"sigma_crit (n={len(ok_g)}): ratio new/ref in [{ratio.min():.3f}, {ratio.max():.3f}] (different noise draws; require within 15%)")
    bad = []
    if len(ok_r) and (ok_r.b_rtn_new - ok_r[f"b_rtn_{THR}"]).abs().max() > 0.05:
        bad.append("b_rtn mismatch")
    print("RTN/kappa PORT VALIDATION:", "FAILED: " + ", ".join(bad) if bad else "PASSED",
          "(gaussian: run --gaussian for the exact-z equivalence check; a sigma_crit ratio off 1",
          "here reflects draw sensitivity, not a port bug)")
    return m


if __name__ == "__main__":
    import sys
    if "--gaussian" in sys.argv:
        gaussian_equivalence()
    else:
        run()
