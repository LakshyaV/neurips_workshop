"""Architecture-generic activation-quantization arm (port of lm_act.py onto lm_eval2 adapters).

For each matrix, fake-quantize its INPUT activations (per-tensor symmetric RTN, absmax scale
from the same calibration tokens) and measure dNLL; record kappa_act = max|x|/rms(x).
Works on GPT-2 and Llama-class models; same prefix caching and built-in cache assert.
"""
import argparse, os, math, time
import numpy as np, pandas as pd, torch

from .lm_eval2 import make_adapter, get_mod, clean_pass, canon

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BITS = [3, 4, 5, 6, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tokens", type=int, default=49152)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--ids", default=os.path.join(RES, "wikitext2_test_gpt2_ids.npy"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--out", default=os.path.join(RES, "lm2_act.parquet"))
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ad = make_adapter(model)
    blocks = ad.blocks()
    ids_all = np.load(args.ids)
    nseq = args.tokens // args.seqlen
    ids = torch.from_numpy(ids_all[: nseq * args.seqlen].astype(np.int64)).view(nseq, args.seqlen)
    nll_clean, caches, _ = clean_pass(ad, ids, args.device, collect_pr=False, batch=args.batch)
    print(f"clean NLL {nll_clean.mean():.4f}", flush=True)

    # input-activation absmax/rms per matrix over the calibration tokens
    stats, hooks = {}, []
    for L, blk in enumerate(blocks):
        for name in ad.mats:
            def mk(key):
                def hook(m, inp, out):
                    x = inp[0].detach().float()
                    s = stats.setdefault(key, dict(amax=0.0, sumsq=0.0, n=0))
                    s["amax"] = max(s["amax"], x.abs().max().item())
                    s["sumsq"] += (x ** 2).sum().item()
                    s["n"] += x.numel()
                return hook
            hooks.append(get_mod(blk, name).register_forward_hook(mk((L, name))))
    with torch.no_grad():
        for i in range(0, nseq, args.batch):
            model(ids[i:i + args.batch].to(args.device))
    for h in hooks:
        h.remove()

    which = [int(x) for x in args.blocks.split(",")] if args.blocks else list(range(len(blocks)))
    rows, t0 = [], time.time()
    for L in which:
        blk = blocks[L]
        hL = caches[L]
        for name in ad.mats:
            mod = get_mod(blk, name)
            st = stats[(L, name)]
            rms = math.sqrt(st["sumsq"] / st["n"])
            amax = st["amax"]
            Wc = canon(mod.weight.detach(), ad.conv1d).float()
            base = dict(block=L, matrix=name, kappa_act=amax / rms, amax_act=amax, rms_act=rms,
                        kappa_w=float(Wc.abs().max() / Wc.pow(2).mean().sqrt()),
                        n_tokens=nseq * args.seqlen)
            for b in BITS:
                qmax = 2 ** (b - 1) - 1
                s = amax / qmax

                def pre(m, inp, s=s, qmax=qmax):
                    x = inp[0]
                    return (torch.clamp(torch.round(x / s), -qmax, qmax) * s,) + tuple(inp[1:])
                h = mod.register_forward_pre_hook(pre)
                nll = torch.cat([ad.forward_from(hL[i:i + args.batch].to(args.device), L,
                                                 ids[i:i + args.batch].to(args.device)).cpu()
                                 for i in range(0, nseq, args.batch)])
                h.remove()
                d = nll - nll_clean
                rows.append(dict(base, bits=b, dnll=float(d.mean()),
                                 dnll_sem=float(d.std() / math.sqrt(len(d)))))
            print(f"block {L:2d} {name:20s} κ_act={base['kappa_act']:7.1f} κ_w={base['kappa_w']:6.1f} | "
                  + " ".join(f"{r['bits']}:{r['dnll']:.3f}" for r in rows[-len(BITS):])
                  + f" ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_parquet(args.out, index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
