"""Architecture-generic per-token margins arm (port of lm_tokens.py onto lm_eval2 adapters).

All matrices are damaged JOINTLY and, per token, we record the clean log-prob of the correct
token, the clean logit margin m_t = logit[correct] - max_other, and under each condition the
log-prob and whether the argmax is still correct. Families: rtn_pc (v1's quantizer, kept for
cross-validation), rtn_gw (group-wise g=128, what P9a is scored on), gaussian (universality
control). Column names keep v1's scheme (lp_rtn{b}/ok_rtn{b} for rtn_pc, lp_g{s}/ok_g{s} for
gaussian) so analysis_tokens.py works unchanged; rtn_gw adds lp_gw{b}/ok_gw{b}.

Differences from v1, by design:
  - clean copies of every damaged matrix are held in memory (~2x weights; fine on GPT-2 and
    on the rented 80 GB GPU for the 8B model, NOT on a 24 GB laptop with an 8B model);
  - gaussian draws use per-(block, matrix, sigma) hashed seeds (subset-independent) instead of
    v1's single sequential stream, so gaussian arms are compared at the threshold level only;
    rtn_pc is deterministic and must reproduce v1 per token (validated on GPT-2).
"""
import argparse, os
import numpy as np, pandas as pd, torch

from .lm_eval2 import make_adapter, get_mod, canon, damage

RES = os.path.join(os.path.dirname(__file__), "..", "results")


@torch.no_grad()
def per_token(model, ids, device, batch=4):
    """logp_correct [N], margin [N], argmax_correct [N] over all next-token positions."""
    lp_all, mg_all, ok_all = [], [], []
    for i in range(0, ids.shape[0], batch):
        x = ids[i:i + batch].to(device)
        logits = model(x).logits[:, :-1].float()
        tgt = x[:, 1:]
        lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt[..., None])[..., 0]
        lc = logits.gather(-1, tgt[..., None])[..., 0]
        lo = logits.clone()
        lo.scatter_(-1, tgt[..., None], -1e9)
        margin = lc - lo.max(dim=-1).values
        lp_all.append(lp.reshape(-1).cpu())
        mg_all.append(margin.reshape(-1).cpu())
        ok_all.append((margin > 0).reshape(-1).cpu())
    return (torch.cat(lp_all).numpy(), torch.cat(mg_all).numpy(),
            torch.cat(ok_all).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tokens", type=int, default=49152)
    ap.add_argument("--offset", type=int, default=147456)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--ids", default=os.path.join(RES, "wikitext2_test_gpt2_ids.npy"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--bits", default="2,3,4,5,6,8")
    ap.add_argument("--sigmas", default="0.03,0.05,0.08,0.12,0.18,0.27,0.4")
    ap.add_argument("--families", default="rtn_pc,gaussian",
                    help="subset of rtn_pc,rtn_gw,gaussian (all damage is joint over matrices)")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(RES, "lm2_tokens.parquet"))
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
    ids = torch.from_numpy(ids_all[args.offset: args.offset + nseq * args.seqlen]
                           .astype(np.int64)).view(nseq, args.seqlen)
    freq = np.bincount(ids_all, minlength=model.config.vocab_size)
    tgt_ids = ids[:, 1:].reshape(-1).numpy()

    keys = [(L, name) for L in range(len(blocks)) for name in ad.mats]
    mods = [get_mod(blocks[L], name) for L, name in keys]
    W0 = [m.weight.detach().clone() for m in mods]

    lp0, m0, ok0 = per_token(model, ids, args.device, args.batch)
    print(f"clean: NLL {-lp0.mean():.4f}, top-1 acc {ok0.mean():.3f}, "
          f"median margin (correct) {np.median(m0[ok0]):.2f}", flush=True)
    d = dict(token=tgt_ids, freq=freq[tgt_ids], lp_fp32=lp0, margin_fp32=m0, ok_fp32=ok0)

    fams = args.families.split(",")
    bits = [int(x) for x in args.bits.split(",")] if args.bits else []
    sigmas = [float(x) for x in args.sigmas.split(",")] if args.sigmas else []
    col = {"rtn_pc": "rtn", "rtn_gw": "gw", "gaussian": "g"}
    levels = sum([[(f, b) for b in bits] for f in ("rtn_pc", "rtn_gw") if f in fams], []) + \
             ([("gaussian", s) for s in sigmas] if "gaussian" in fams else [])
    for fam, knob in levels:
        for (L, name), mod, w in zip(keys, mods, W0):
            gen = torch.Generator(device="cpu").manual_seed(
                hash((L, name, fam, round(knob * 1e6))) & 0x7FFFFFFF)
            Wd = damage(canon(w, ad.conv1d), fam, knob, gen, args.group)
            mod.weight.copy_(Wd if ad.conv1d else Wd.t())
        lp, mg, ok = per_token(model, ids, args.device, args.batch)
        for mod, w in zip(mods, W0):
            mod.weight.copy_(w)
        tag = f"{col[fam]}{knob:g}" if fam == "gaussian" else f"{col[fam]}{int(knob)}"
        d[f"lp_{tag}"], d[f"ok_{tag}"] = lp, ok
        print(f"{fam} {knob:g}: dNLL {-(lp.mean() - lp0.mean()):+.4f}, top-1 acc {ok.mean():.3f}, "
              f"of clean-correct still correct {ok[ok0].mean():.3f}", flush=True)
        pd.DataFrame(d).to_parquet(args.out, index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
