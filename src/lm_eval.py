"""§5A LM bridge: per-weight-matrix damage threshold in GPT-2 small on a continuous noise axis.
For each Conv1D matrix (48 = 12 blocks x {c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj}):
  - kappa = max|W| / rms(W)
  - superposition proxy F_hat = d_in / PR(input-activation covariance), PR = (Σλ)^2 / Σλ^2
  - sweep additive Gaussian noise W + σ·rms(W)·z, σ on a log grid; ΔNLL vs σ; σ_crit at ΔNLL = {0.02, 0.05, 0.10}
  - RTN quantization at b in {3,4,5,6} (per-tensor symmetric max-abs) as discrete points on the same axis
Prefix caching: blocks 0..L-1 are unchanged when damaging block L, so the residual stream entering block L is cached.
Embeddings (wte, tied to lm_head) are never damaged."""
import argparse, os, sys, time, json, math
import numpy as np, pandas as pd, torch
from transformers import GPT2LMHeadModel

RES = os.path.join(os.path.dirname(__file__), "..", "results")
MATS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]


def get_mod(block, name):
    m = block
    for part in name.split("."):
        m = getattr(m, part)
    return m


@torch.no_grad()
def nll_from(model, h_in, L, targets):
    """Run blocks L.. then ln_f + lm_head; return per-sequence mean NLL [B]. h_in:[B,T,d] residual stream into block L."""
    h = h_in
    for blk in model.transformer.h[L:]:
        out = blk(h)
        h = out[0] if isinstance(out, tuple) else out
    h = model.transformer.ln_f(h)
    logits = model.lm_head(h)
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    nll = -lp.gather(-1, targets[:, 1:, None])[..., 0]
    return nll.mean(dim=1)


@torch.no_grad()
def clean_pass(model, ids, device, n_blocks, collect_pr=True):
    """Cache residual stream into each block for all sequences; collect input covariances for PR."""
    B, T = ids.shape
    caches = [[] for _ in range(n_blocks)]
    covs, counts = {}, {}
    hooks = []
    if collect_pr:
        for L, blk in enumerate(model.transformer.h):
            for name in MATS:
                mod = get_mod(blk, name)
                key = (L, name)
                def mk(key):
                    def hook(m, inp, out):
                        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                        covs[key] = covs.get(key, 0) + x.T @ x
                        counts[key] = counts.get(key, 0) + x.shape[0]
                    return hook
                hooks.append(mod.register_forward_hook(mk(key)))
    for L, blk in enumerate(model.transformer.h):
        def mk_pre(L):
            def pre(m, args, kwargs):
                caches[L].append(args[0].detach())
            return pre
        hooks.append(blk.register_forward_pre_hook(mk_pre(L), with_kwargs=True))
    nll_clean = []
    for i in range(0, B, 8):
        x = ids[i:i + 8].to(device)
        out = model(x)
        lp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
        nll_clean.append((-lp.gather(-1, x[:, 1:, None])[..., 0]).mean(dim=1).cpu())
    for h in hooks:
        h.remove()
    caches = [torch.cat(c, dim=0) for c in caches]
    pr = {}
    if collect_pr:
        for key, C in covs.items():
            C = (C / counts[key]).float()
            pr[key] = float(torch.trace(C) ** 2 / (C ** 2).sum())      # (Σλ)^2 / Σλ^2 = tr(C)^2 / ||C||_F^2
    return torch.cat(nll_clean), caches, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--tokens", type=int, default=49152)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sigmas", default="0.03,0.045,0.07,0.1,0.15,0.22,0.33,0.5,0.75")
    ap.add_argument("--bits", default="3,4,5,6")
    ap.add_argument("--families", default="gaussian,rtn", help="subset of gaussian,rtn,rtn_pc")
    ap.add_argument("--blocks", default=None, help="comma list of block indices (default all)")
    ap.add_argument("--out", default=os.path.join(RES, "lm_layers.parquet"))
    args = ap.parse_args()
    device = args.device
    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    ids_all = np.load(os.path.join(RES, "wikitext2_test_gpt2_ids.npy"))
    nseq = args.tokens // args.seqlen
    ids = torch.from_numpy(ids_all[: nseq * args.seqlen].astype(np.int64)).view(nseq, args.seqlen)
    n_blocks = len(model.transformer.h)
    t0 = time.time()
    nll_clean, caches, pr = clean_pass(model, ids, device, n_blocks)
    print(f"clean NLL {nll_clean.mean():.4f} over {nseq*args.seqlen} tokens; clean pass {time.time()-t0:.1f}s", flush=True)
    # sanity: manual block iteration reproduces the full-model NLL
    chk = torch.cat([nll_from(model, caches[0][i:i + 8].to(device), 0, ids[i:i + 8].to(device)).cpu() for i in range(0, nseq, 8)])
    assert torch.allclose(chk, nll_clean, atol=2e-3), (chk.mean(), nll_clean.mean())
    print("prefix-cache path verified against full forward", flush=True)

    sigmas = [float(x) for x in args.sigmas.split(",")]
    bits = [int(x) for x in args.bits.split(",")] if args.bits else []
    blocks = [int(x) for x in args.blocks.split(",")] if args.blocks else list(range(n_blocks))
    rows = []
    gen = torch.Generator(device="cpu").manual_seed(1234)
    for L in blocks:
        blk = model.transformer.h[L]
        hL = caches[L]
        for name in MATS:
            mod = get_mod(blk, name)
            W0 = mod.weight.detach().clone()
            rms = W0.pow(2).mean().sqrt().item(); amax = W0.abs().max().item()
            kappa = amax / rms
            d_in = W0.shape[0]
            amax_c = W0.abs().amax(dim=0, keepdim=True)                      # per output channel (Conv1D: [in, out])
            n2_c = W0.pow(2).sum(dim=0, keepdim=True); rms_c = (n2_c / W0.shape[0]).sqrt()
            kappa_eff = float(((amax_c / rms_c) ** 2 * n2_c).sum().sqrt() / n2_c.sum().sqrt())   # norm-weighted per-channel κ
            base = dict(block=L, matrix=name, d_in=d_in, d_out=W0.shape[1], kappa=kappa, kappa_eff=kappa_eff, rms=rms, amax=amax,
                        PR=pr[(L, name)], F_hat=d_in / pr[(L, name)], n_tokens=nseq * args.seqlen)
            fams = args.families.split(",")
            levels = ([("gaussian", s) for s in sigmas] if "gaussian" in fams else []) + \
                     ([("rtn", b) for b in bits] if "rtn" in fams else []) + ([("rtn_pc", b) for b in bits] if "rtn_pc" in fams else [])
            for fam, knob in levels:
                if fam == "gaussian":
                    z = torch.randn(W0.shape, generator=gen).to(device)
                    Wd = W0 + knob * rms * z
                elif fam == "rtn":
                    qmax = 2 ** (knob - 1) - 1; s = amax / qmax
                    Wd = torch.clamp(torch.round(W0 / s), -qmax, qmax) * s
                else:
                    qmax = 2 ** (knob - 1) - 1; s = (amax_c / qmax).clamp_min(1e-12)
                    Wd = torch.clamp(torch.round(W0 / s), -qmax, qmax) * s
                sig_rel = ((Wd - W0).pow(2).mean().sqrt() / rms).item()
                mod.weight.copy_(Wd)
                t1 = time.time()
                nll = torch.cat([nll_from(model, hL[i:i + 8].to(device), L, ids[i:i + 8].to(device)).cpu() for i in range(0, nseq, 8)])
                mod.weight.copy_(W0)
                d = (nll - nll_clean)
                rows.append(dict(base, family=fam, knob=float(knob), sigma_rel=sig_rel, dnll=float(d.mean()),
                                 dnll_sem=float(d.std() / math.sqrt(len(d))), nll=float(nll.mean()), eval_s=time.time() - t1))
            print(f"block {L:2d} {name:12s} κ={kappa:5.1f} PR={pr[(L,name)]:6.1f} F̂={d_in/pr[(L,name)]:6.2f} | "
                  + " ".join(f"{r['knob']:.3g}:{r['dnll']:.3f}" for r in rows[-len(levels):]) + f"  ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_parquet(args.out, index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
