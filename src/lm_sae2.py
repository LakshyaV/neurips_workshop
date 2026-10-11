"""Architecture-generic SAE-feature erasure arm (port of lm_sae.py onto lm_eval2 adapters).

Damage every matrix BELOW the SAE read point jointly, read the residual stream at the read
point, run the SAE on paired clean/damaged activations, and accumulate the v1 per-feature
statistics (n_act0, sum_a0, sum_a0sq, sum_err, sum_asq, n_joint, n_actq per condition).

Two SAE formats:
  - v1 / jbloom GPT-2 (`sae_weights.safetensors`: W_enc/b_enc/W_dec/b_dec), ReLU,
    center the residual per position and subtract b_dec before encoding (v1 defaults);
  - Llama Scope (`final.safetensors`: encoder./decoder. weight+bias, transposed), JumpReLU
    with the scalar threshold from hyperparams.json, NO centering, NO b_dec subtraction,
    input scaled by sqrt(d_model)/dataset_average_activation_norm["in"] (SAELens
    "expected_average_only_in"); everything is accumulated in that scaled space.

Read point: HF hidden_states[read_hs] (= residual entering block read_hs), BOS position
dropped. v1 GPT-2: read_hs 6, damage blocks 0-5, ctx 128. Llama Scope L15R: read_hs 16
(blocks.15.hook_resid_post), damage layers 0-15, ctx 1024. Sequences are BOS + (ctx-1)
body tokens, as in SAE training. Joint damage holds clean copies (~2x weight memory below
the read point).
"""
import argparse, json, os
import numpy as np, pandas as pd, torch

from .lm_eval2 import make_adapter, get_mod, canon, damage, stable_seed

RES = os.path.join(os.path.dirname(__file__), "..", "results")


class SAE2:
    def __init__(self, path, device, fmt):
        from safetensors.torch import load_file
        self.fmt = fmt
        if fmt == "v1":
            w = load_file(os.path.join(path, "sae_weights.safetensors"))
            self.W_enc, self.b_enc = w["W_enc"].float().to(device), w["b_enc"].float().to(device)
            self.W_dec, self.b_dec = w["W_dec"].float().to(device), w["b_dec"].float().to(device)
            self.center, self.sub_bdec, self.scale, self.thresh = True, True, 1.0, None
        elif fmt == "llamascope":
            hp = json.load(open(os.path.join(path, "hyperparams.json")))
            w = load_file(os.path.join(path, "checkpoints", "final.safetensors"))
            self.W_enc = w["encoder.weight"].float().t().contiguous().to(device)   # [d_in, F]
            self.b_enc = w["encoder.bias"].float().to(device)
            self.W_dec = w["decoder.weight"].float().t().contiguous().to(device)   # [F, d_in]
            self.b_dec = w["decoder.bias"].float().to(device)
            self.center, self.sub_bdec = False, False
            self.scale = float(hp["d_model"] ** 0.5 / hp["dataset_average_activation_norm"]["in"])
            self.thresh = float(hp["jump_relu_threshold"])
        else:
            raise ValueError(fmt)
        self.dec_norm = self.W_dec.norm(dim=1)

    def prep(self, x):
        x = x.float() * self.scale
        if self.center:
            x = x - x.mean(dim=-1, keepdim=True)
        return x

    def encode_prepped(self, x):
        pre = ((x - self.b_dec) if self.sub_bdec else x) @ self.W_enc + self.b_enc
        a = torch.relu(pre)
        if self.thresh is not None:
            a = a * (pre > self.thresh)
        return a

    def encode(self, x):
        return self.encode_prepped(self.prep(x))

    def recon_ve(self, x):
        x = self.prep(x)
        a = self.encode_prepped(x)
        r = a @ self.W_dec + self.b_dec
        return (1 - ((r - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum(),
                (a > 0).float().sum(-1).mean())


@torch.no_grad()
def resid_at(model, ids, device, read_hs, bs=8):
    out = []
    for i in range(0, ids.shape[0], bs):
        x = ids[i:i + bs].to(device)
        hs = model(x, output_hidden_states=True).hidden_states[read_hs]
        out.append(hs[:, 1:].reshape(-1, hs.shape[-1]).float())        # drop the BOS position
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tokens", type=int, default=196608, help="body tokens (excl. BOS)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=128, help="context incl. BOS (v1: 128, Llama Scope: 1024)")
    ap.add_argument("--ids", default=os.path.join(RES, "wikitext2_test_gpt2_ids.npy"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sae", default=os.path.join(RES, "sae"))
    ap.add_argument("--sae-format", default="v1", choices=["v1", "llamascope"])
    ap.add_argument("--read-hs", type=int, default=6,
                    help="hidden_states index read (v1 GPT-2: 6; Llama Scope L15R: 16)")
    ap.add_argument("--bits", default="2,3,4,5,6,8")
    ap.add_argument("--sigmas", default="0.05,0.1,0.15,0.2")
    ap.add_argument("--families", default="rtn_pc,gaussian",
                    help="subset of rtn_pc,rtn_gw,gaussian (joint, blocks below read point)")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(RES, "lm2_sae.parquet"))
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ad = make_adapter(model)
    device = args.device

    body = args.ctx - 1
    ids_all = np.load(args.ids)
    n_seq = args.tokens // body
    bodyt = torch.from_numpy(ids_all[args.offset: args.offset + n_seq * body]
                             .astype(np.int64)).view(n_seq, body)
    bos = model.config.bos_token_id
    ids = torch.cat([torch.full((n_seq, 1), bos, dtype=torch.long), bodyt], dim=1)

    sae = SAE2(args.sae, device, args.sae_format)
    keys = [(L, name) for L in range(args.read_hs) for name in ad.mats]
    mods = [get_mod(ad.blocks()[L], name) for L, name in keys]
    W0 = [m.weight.detach().clone() for m in mods]

    x0 = resid_at(model, ids, device, args.read_hs, args.batch)
    ve, l0 = sae.recon_ve(x0[:20000])
    print(f"clean residual: {x0.shape[0]} tokens; SAE VE {ve:.3f}, L0 {l0:.1f}", flush=True)
    F = sae.W_enc.shape[1]
    stats = {}

    def run(name):
        acc = dict(n_act0=torch.zeros(F, device=device), sum_a0=torch.zeros(F, device=device),
                   sum_a0sq=torch.zeros(F, device=device), sum_err=torch.zeros(F, device=device),
                   sum_asq=torch.zeros(F, device=device), n_joint=torch.zeros(F, device=device),
                   n_actq=torch.zeros(F, device=device))
        xq = resid_at(model, ids, device, args.read_hs, args.batch)
        for i in range(0, x0.shape[0], 4096):
            a0, a = sae.encode(x0[i:i + 4096]), sae.encode(xq[i:i + 4096])
            acc["n_act0"] += (a0 > 0).float().sum(0)
            acc["sum_a0"] += a0.sum(0)
            acc["sum_a0sq"] += (a0 ** 2).sum(0)
            acc["sum_err"] += ((a - a0) ** 2).sum(0)
            acc["sum_asq"] += (a ** 2).sum(0)
            acc["n_joint"] += ((a0 > 0) & (a > 0)).float().sum(0)
            acc["n_actq"] += (a > 0).float().sum(0)
        ve, l0 = sae.recon_ve(xq[:20000])
        stats[name] = {k: v.cpu().numpy() for k, v in acc.items()}
        rho = 1 - acc["sum_err"] / acc["sum_a0sq"].clamp_min(1e-12)
        live = acc["n_act0"] > 20
        print(f"  {name:12s}: VE on damaged resid {ve:.3f}, L0 {l0:.1f}; "
              f"features with rho>=0.5: {(rho[live] >= 0.5).float().mean():.3f} "
              f"of {int(live.sum())} live", flush=True)

    run("fp32")
    fams = args.families.split(",")
    bits = [int(x) for x in args.bits.split(",")] if args.bits else []
    sigmas = [float(x) for x in args.sigmas.split(",")] if args.sigmas else []
    col = {"rtn_pc": "rtn", "rtn_gw": "gw", "gaussian": "g"}
    levels = sum([[(f, b) for b in bits] for f in ("rtn_pc", "rtn_gw") if f in fams], []) + \
             ([("gaussian", s) for s in sigmas] if "gaussian" in fams else [])
    for fam, knob in levels:
        for (L, name), mod, w in zip(keys, mods, W0):
            gen = torch.Generator(device="cpu").manual_seed(
                stable_seed(L, name, fam, round(knob * 1e6)))
            Wd = damage(canon(w, ad.conv1d), fam, knob, gen, args.group)
            mod.weight.copy_(Wd if ad.conv1d else Wd.t())
        run(f"{col[fam]}{knob:g}" if fam == "gaussian" else f"{col[fam]}{int(knob)}")
        for mod, w in zip(mods, W0):
            mod.weight.copy_(w)
        rows = dict(feature=np.arange(F), dec_norm=sae.dec_norm.cpu().numpy(), n_tokens=x0.shape[0])
        for name, acc in stats.items():
            for k, v in acc.items():
                rows[f"{k}_{name}"] = v
        pd.DataFrame(rows).to_parquet(args.out, index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
