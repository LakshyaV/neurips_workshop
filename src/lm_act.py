"""Activation-quantization arm: for each of the 48 GPT-2 matrices, fake-quantize the INPUT activations to that matrix
(per-tensor symmetric RTN with the absmax of the calibration set, b in BITS) and measure ΔNLL; record the activation
outlier ratio kappa_act = max|x| / rms(x) over the calibration tokens. Same prefix caching as lm_eval."""
import argparse, os, sys, math, time
import numpy as np, pandas as pd, torch
from transformers import GPT2LMHeadModel
from .lm_eval import get_mod, MATS, nll_from, clean_pass

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BITS = [3, 4, 5, 6, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps"); ap.add_argument("--tokens", type=int, default=49152)
    ap.add_argument("--out", default=os.path.join(RES, "lm_act.parquet"))
    args = ap.parse_args(); device = args.device
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).to(device).eval()
    for prm in model.parameters(): prm.requires_grad_(False)
    ids_all = np.load(os.path.join(RES, "wikitext2_test_gpt2_ids.npy")); nseq = args.tokens // 1024
    ids = torch.from_numpy(ids_all[: nseq * 1024].astype(np.int64)).view(nseq, 1024)
    n_blocks = len(model.transformer.h)
    nll_clean, caches, pr = clean_pass(model, ids, device, n_blocks, collect_pr=False)
    print(f"clean NLL {nll_clean.mean():.4f}", flush=True)
    # activation statistics per matrix input (absmax and rms over calibration tokens = the same 49k tokens)
    stats = {}
    hooks = []
    for L, blk in enumerate(model.transformer.h):
        for name in MATS:
            mod = get_mod(blk, name)
            def mk(key):
                def hook(m, inp, out):
                    x = inp[0].detach().float()
                    s = stats.setdefault(key, dict(amax=0.0, sumsq=0.0, n=0))
                    s["amax"] = max(s["amax"], x.abs().max().item()); s["sumsq"] += (x ** 2).sum().item(); s["n"] += x.numel()
                return hook
            hooks.append(mod.register_forward_hook(mk((L, name))))
    with torch.no_grad():
        for i in range(0, nseq, 8): model(ids[i:i + 8].to(device))
    for h in hooks: h.remove()
    rows = []
    t0 = time.time()
    for L in range(n_blocks):
        blk = model.transformer.h[L]; hL = caches[L]
        for name in MATS:
            mod = get_mod(blk, name); st = stats[(L, name)]
            rms = math.sqrt(st["sumsq"] / st["n"]); amax = st["amax"]; kappa_act = amax / rms
            W = mod.weight; kappa_w = (W.abs().max() / W.pow(2).mean().sqrt()).item()
            base = dict(block=L, matrix=name, kappa_act=kappa_act, kappa_w=kappa_w, amax_act=amax, rms_act=rms)
            for b in BITS:
                qmax = 2 ** (b - 1) - 1; s = amax / qmax
                def pre(m, inp, s=s, qmax=qmax):
                    x = inp[0]
                    return (torch.clamp(torch.round(x / s), -qmax, qmax) * s,) + tuple(inp[1:])
                h = mod.register_forward_pre_hook(pre)
                nll = torch.cat([nll_from(model, hL[i:i + 8].to(device), L, ids[i:i + 8].to(device)).cpu() for i in range(0, nseq, 8)])
                h.remove()
                d = nll - nll_clean
                rows.append(dict(base, bits=b, dnll=float(d.mean()), dnll_sem=float(d.std() / math.sqrt(len(d)))))
            print(f"block {L:2d} {name:12s} κ_act={kappa_act:6.1f} κ_w={kappa_w:5.1f} | " + " ".join(f"{r['bits']}:{r['dnll']:.3f}" for r in rows[-len(BITS):]) + f" ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_parquet(args.out, index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
