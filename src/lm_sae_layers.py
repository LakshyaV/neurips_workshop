"""Validate the superposition proxy F_hat = d_in / PR against a per-layer SAE feature count.
For each residual-stream SAE layer L (jbloom GPT2-Small-SAEs-Reformatted, blocks.L.hook_resid_pre), run the fp32 model on
~50k tokens (128-token BOS contexts), encode the residual entering block L, and count live features (firing on >= 20 tokens),
also the participation ratio of the SAE activation covariance. F_SAE(L) = live / d_model. Compare with F_hat of the matrices
that read that residual (attn.c_attn and mlp.c_fc of block L, after LayerNorm) and with their Gaussian-noise thresholds."""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from transformers import GPT2LMHeadModel
from .lm_sae import SAE, resid_pre

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps"); ap.add_argument("--tokens", type=int, default=49152)
    ap.add_argument("--out", default=os.path.join(RES, "lm_sae_layers.csv"))
    args = ap.parse_args(); device = args.device
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).to(device).eval()
    for prm in model.parameters(): prm.requires_grad_(False)
    ids_all = np.load(os.path.join(RES, "wikitext2_test_gpt2_ids.npy")); n_seq = args.tokens // 127
    body = torch.from_numpy(ids_all[: n_seq * 127].astype(np.int64)).view(n_seq, 127)
    ids = torch.cat([torch.full((n_seq, 1), 50256, dtype=torch.long), body], dim=1)
    thr = pd.read_csv(os.path.join(RES, "lm_layer_thresholds.csv"))
    rows = []
    for L in range(12):
        path = os.path.join(RES, "sae") if L == 6 else os.path.join(RES, "sae", f"L{L}")
        if not os.path.exists(os.path.join(path, "sae_weights.safetensors")):
            print(f"layer {L}: no SAE weights, skipped", flush=True); continue
        sae = SAE(path, device, center=True, sub_bdec=True)
        x = resid_pre(model, ids, device, layer=L)
        ve, l0 = sae.recon_ve(x[:20000])
        n_act = torch.zeros(sae.W_enc.shape[1], device=device); s1 = torch.zeros_like(n_act); s2 = torch.zeros_like(n_act)
        for i in range(0, x.shape[0], 4096):
            a = sae.encode(x[i:i + 4096]); n_act += (a > 0).float().sum(0); s1 += a.sum(0); s2 += (a ** 2).sum(0)
        N = x.shape[0]; live = int((n_act >= 20).sum()); var = (s2 / N - (s1 / N) ** 2).clamp_min(0)
        pr_act = float(var.sum() ** 2 / (var ** 2).sum())                         # participation ratio of SAE feature variances
        row = dict(layer=L, tokens=N, sae_ve=float(ve), sae_l0=float(l0), live_features=live, F_sae=live / 768, pr_features=pr_act)
        for name in ["attn.c_attn", "mlp.c_fc"]:
            t = thr[(thr.block == L) & (thr.matrix == name)]
            if len(t):
                row[f"F_hat_{name}"] = float(t.F_hat.iloc[0]); row[f"PR_{name}"] = float(t.PR.iloc[0])
                row[f"sigma_crit_{name}"] = float(t["sigma_crit_0.05"].iloc[0]); row[f"censor_{name}"] = t["censor_0.05"].iloc[0]
                row[f"kappa_{name}"] = float(t.kappa.iloc[0])
        rows.append(row); print(f"layer {L:2d}: VE {ve:.3f} L0 {l0:.1f} live {live} F_sae {live/768:.1f} PR_feat {pr_act:.0f} | F_hat c_attn {row.get('F_hat_attn.c_attn', float('nan')):.1f} c_fc {row.get('F_hat_mlp.c_fc', float('nan')):.1f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    d = pd.DataFrame(rows)
    for name in ["attn.c_attn", "mlp.c_fc"]:
        ok = d.dropna(subset=[f"F_hat_{name}"])
        if len(ok) >= 4:
            r1 = np.corrcoef(np.log2(ok.F_sae), np.log2(ok[f"F_hat_{name}"]))[0, 1]
            r2 = np.corrcoef(np.log2(ok.pr_features), np.log2(ok[f"F_hat_{name}"]))[0, 1]
            okc = ok[ok[f"censor_{name}"] == "ok"]
            r3 = np.corrcoef(np.log2(okc.F_sae), np.log2(okc[f"sigma_crit_{name}"]))[0, 1] if len(okc) >= 4 else float("nan")
            print(f"{name}: corr(log2 F_sae, log2 F_hat) = {r1:.2f}; corr(log2 PR_feat, log2 F_hat) = {r2:.2f}; corr(log2 F_sae, log2 sigma_crit) = {r3:.2f} (n={len(okc)})")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
