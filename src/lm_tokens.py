"""Per-token critical bit-width in GPT-2 small: does a token's fp32 logit margin predict the precision at which its
prediction breaks?  All 48 matrices quantized jointly (per-channel RTN, b in BITS) or perturbed jointly with Gaussian
noise (sigma_rel in SIGMAS, universality control). Held-out WikiText-2 tokens.
Per token t we record: fp32 log-prob of the correct token, fp32 logit margin m_t = logit[correct] - max_other,
and under each condition the log-prob and whether the argmax is still the correct token."""
import argparse, os, sys, math
import numpy as np, pandas as pd, torch
from transformers import GPT2LMHeadModel
from .lm_eval import get_mod, MATS
from .lm_mixed import rtn_pc

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BITS = [2, 3, 4, 5, 6, 8]
SIGMAS = [0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.4]


@torch.no_grad()
def per_token(model, ids, device):
    """returns logp_correct [N], margin [N], argmax_correct [N] over all next-token positions"""
    lp_all, mg_all, ok_all = [], [], []
    for i in range(0, ids.shape[0], 4):
        x = ids[i:i + 4].to(device)
        logits = model(x).logits[:, :-1].float()
        tgt = x[:, 1:]
        lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt[..., None])[..., 0]
        lc = logits.gather(-1, tgt[..., None])[..., 0]
        lo = logits.clone(); lo.scatter_(-1, tgt[..., None], -1e9)
        margin = lc - lo.max(dim=-1).values
        lp_all.append(lp.reshape(-1).cpu()); mg_all.append(margin.reshape(-1).cpu()); ok_all.append((margin > 0).reshape(-1).cpu())
    return torch.cat(lp_all).numpy(), torch.cat(mg_all).numpy(), torch.cat(ok_all).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps"); ap.add_argument("--tokens", type=int, default=49152); ap.add_argument("--offset", type=int, default=147456)
    ap.add_argument("--out", default=os.path.join(RES, "lm_tokens.parquet"))
    args = ap.parse_args(); device = args.device
    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).to(device).eval()
    for prm in model.parameters(): prm.requires_grad_(False)
    ids_all = np.load(os.path.join(RES, "wikitext2_test_gpt2_ids.npy")); nseq = args.tokens // 1024
    ids = torch.from_numpy(ids_all[args.offset: args.offset + nseq * 1024].astype(np.int64)).view(nseq, 1024)
    # unigram frequency of each target token within the full test corpus (for the rarity confound)
    freq = np.bincount(ids_all, minlength=50257); tgt_ids = ids[:, 1:].reshape(-1).numpy()
    mods = [get_mod(model.transformer.h[L], name) for L in range(12) for name in MATS]; W0 = [m.weight.detach().clone() for m in mods]
    lp0, m0, ok0 = per_token(model, ids, device)
    print(f"fp32: NLL {-lp0.mean():.4f}, top-1 acc {ok0.mean():.3f}, median margin (correct tokens) {np.median(m0[ok0]):.2f}", flush=True)
    d = dict(token=tgt_ids, freq=freq[tgt_ids], lp_fp32=lp0, margin_fp32=m0, ok_fp32=ok0)
    gen = torch.Generator(device="cpu").manual_seed(7)
    for b in BITS:
        for m, w in zip(mods, W0): m.weight.copy_(rtn_pc(w, b))
        lp, mg, ok = per_token(model, ids, device)
        for m, w in zip(mods, W0): m.weight.copy_(w)
        d[f"lp_rtn{b}"], d[f"ok_rtn{b}"] = lp, ok
        print(f"rtn_pc {b} bits: dNLL {-(lp.mean() - lp0.mean()):+.4f}, top-1 acc {ok.mean():.3f}, of fp32-correct still correct {ok[ok0].mean():.3f}", flush=True)
    for s in SIGMAS:
        for m, w in zip(mods, W0):
            z = torch.randn(w.shape, generator=gen).to(device); m.weight.copy_(w + s * w.pow(2).mean().sqrt() * z)
        lp, mg, ok = per_token(model, ids, device)
        for m, w in zip(mods, W0): m.weight.copy_(w)
        d[f"lp_g{s}"], d[f"ok_g{s}"] = lp, ok
        print(f"gauss {s}: dNLL {-(lp.mean() - lp0.mean()):+.4f}, top-1 acc {ok.mean():.3f}, of fp32-correct still correct {ok[ok0].mean():.3f}", flush=True)
    pd.DataFrame(d).to_parquet(args.out, index=False); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
