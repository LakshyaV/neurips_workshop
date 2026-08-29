"""§5E: per-SAE-feature critical bit-width in GPT-2 small.
SAE: jbloom/GPT2-Small-SAEs-Reformatted, blocks.6.hook_resid_pre (24,576 features, d_in 768; trained on 128-token
contexts with BOS, on TransformerLens activations = HF residual stream with the per-position mean over d_model removed).
We damage blocks 0-5 (24 matrices, jointly; per-channel RTN at BITS, Gaussian noise at SIGMAS), read the residual stream
entering block 6, run the SAE, and accumulate per-feature statistics paired with the fp32 activations."""
import argparse, os, sys, math, json
import numpy as np, pandas as pd, torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel
from .lm_eval import get_mod, MATS
from .lm_mixed import rtn_pc

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BITS = [2, 3, 4, 5, 6, 8]
SIGMAS = [0.05, 0.1, 0.15, 0.2]
LAYER = 6


class SAE:
    def __init__(self, path, device, center=True, sub_bdec=True):
        w = load_file(os.path.join(path, "sae_weights.safetensors"))
        self.W_enc, self.b_enc, self.W_dec, self.b_dec = (w[k].to(device) for k in ["W_enc", "b_enc", "W_dec", "b_dec"])
        self.center, self.sub_bdec = center, sub_bdec
        self.dec_norm = self.W_dec.norm(dim=1)

    def prep(self, x):
        if self.center: x = x - x.mean(dim=-1, keepdim=True)
        return x

    def encode(self, x):
        x = self.prep(x)
        if self.sub_bdec: x = x - self.b_dec
        return torch.relu(x @ self.W_enc + self.b_enc)

    def recon_ve(self, x):
        x = self.prep(x); a = self.encode(x if not self.center else x + 0)      # prep is idempotent after centering
        a = torch.relu(((x - self.b_dec) if self.sub_bdec else x) @ self.W_enc + self.b_enc)
        r = a @ self.W_dec + self.b_dec
        return 1 - ((r - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum(), (a > 0).float().sum(-1).mean()


@torch.no_grad()
def resid_pre(model, ids, device, layer=LAYER, bs=32):
    out = []
    for i in range(0, ids.shape[0], bs):
        x = ids[i:i + bs].to(device)
        hs = model(x, output_hidden_states=True).hidden_states[layer]
        out.append(hs[:, 1:].reshape(-1, hs.shape[-1]))               # drop the BOS position
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps"); ap.add_argument("--tokens", type=int, default=196608); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sae", default=os.path.join(RES, "sae")); ap.add_argument("--out", default=os.path.join(RES, "lm_sae.parquet"))
    ap.add_argument("--center", type=int, default=1); ap.add_argument("--sub_bdec", type=int, default=1)
    args = ap.parse_args(); device = args.device
    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).to(device).eval()
    for prm in model.parameters(): prm.requires_grad_(False)
    ids_all = np.load(os.path.join(RES, "wikitext2_test_gpt2_ids.npy"))
    n_seq = args.tokens // 127
    body = torch.from_numpy(ids_all[args.offset: args.offset + n_seq * 127].astype(np.int64)).view(n_seq, 127)
    ids = torch.cat([torch.full((n_seq, 1), 50256, dtype=torch.long), body], dim=1)     # BOS + 127 tokens
    sae = SAE(args.sae, device, center=bool(args.center), sub_bdec=bool(args.sub_bdec))
    mods = [get_mod(model.transformer.h[L], name) for L in range(LAYER) for name in MATS]
    W0 = [m.weight.detach().clone() for m in mods]
    x0 = resid_pre(model, ids, device)                                   # [N, 768] fp32 residual into block 6
    ve, l0 = sae.recon_ve(x0[:20000]); print(f"fp32 residual: {x0.shape[0]} tokens; SAE variance explained {ve:.3f}, L0 {l0:.1f}", flush=True)
    F = sae.W_enc.shape[1]
    stats = {}

    def run(name, a0_fn):
        # accumulate per-feature paired statistics in chunks
        acc = dict(n_act0=torch.zeros(F, device=device), sum_a0=torch.zeros(F, device=device), sum_a0sq=torch.zeros(F, device=device),
                   sum_err=torch.zeros(F, device=device), sum_asq=torch.zeros(F, device=device), n_joint=torch.zeros(F, device=device), n_actq=torch.zeros(F, device=device))
        xq = resid_pre(model, ids, device)
        for i in range(0, x0.shape[0], 4096):
            a0 = sae.encode(x0[i:i + 4096]); a = sae.encode(xq[i:i + 4096])
            acc["n_act0"] += (a0 > 0).float().sum(0); acc["sum_a0"] += a0.sum(0); acc["sum_a0sq"] += (a0 ** 2).sum(0)
            acc["sum_err"] += ((a - a0) ** 2).sum(0); acc["sum_asq"] += (a ** 2).sum(0)
            acc["n_joint"] += ((a0 > 0) & (a > 0)).float().sum(0); acc["n_actq"] += (a > 0).float().sum(0)
        ve, l0 = sae.recon_ve(xq[:20000])
        stats[name] = {k: v.cpu().numpy() for k, v in acc.items()}
        rho = 1 - acc["sum_err"] / acc["sum_a0sq"].clamp_min(1e-12); live = acc["n_act0"] > 20
        print(f"  {name:12s}: SAE VE on damaged resid {ve:.3f}, L0 {l0:.1f}; features with rho>=0.5: {(rho[live] >= 0.5).float().mean():.3f} of {int(live.sum())} live", flush=True)

    run("fp32", None)
    for b in BITS:
        for m, w in zip(mods, W0): m.weight.copy_(rtn_pc(w, b))
        run(f"rtn{b}", None)
        for m, w in zip(mods, W0): m.weight.copy_(w)
    gen = torch.Generator(device="cpu").manual_seed(7)
    for s in SIGMAS:
        for m, w in zip(mods, W0):
            z = torch.randn(w.shape, generator=gen).to(device); m.weight.copy_(w + s * w.pow(2).mean().sqrt() * z)
        run(f"g{s}", None)
        for m, w in zip(mods, W0): m.weight.copy_(w)
    rows = dict(feature=np.arange(F), dec_norm=sae.dec_norm.cpu().numpy(), n_tokens=x0.shape[0])
    for name, acc in stats.items():
        for k, v in acc.items(): rows[f"{k}_{name}"] = v
    pd.DataFrame(rows).to_parquet(args.out, index=False); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
