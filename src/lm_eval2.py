"""Architecture-generic port of lm_eval.py (v2 Phase 2 harness).

Adds to the original GPT-2-only harness:
  - a Llama-class adapter (q/k/v/o + gate/up/down nn.Linear, RoPE block iteration);
  - group-wise RTN (`rtn_gw`, groups of g=128 along the input dimension), the modern default;
  - tail incoherence statistics per matrix (kc_max, kg_p90, kg_max) next to kappa / kappa_eff;
  - `--stats-only` (the kappa-only spike: weight statistics, NO damage evaluations);
  - `--dtype bf16` for 7B-class models ("full-precision statistics" = the released bf16 weights);
  - `--mechanics-check` on a random-init Llama config (no download, no science).

Validation protocol (no new results): the prefix-cache path must reproduce the full forward
(built-in assert, as in lm_eval.py); on GPT-2 the RTN arm is deterministic given weights+data,
so its dnll curve must match the original run -- src/validate_port.py compares recomputed
thresholds against results/lm_layer_thresholds.csv.

Gaussian noise is seeded per (block, matrix, level) so results do not depend on which subset of
blocks is run; draws therefore differ from the original lm_eval.py run, and gaussian arms are
compared at the threshold level, not the sample level.
"""
import argparse, os, math, time
import numpy as np, pandas as pd, torch

RES = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------- adapters
class GPT2Adapter:
    """Conv1D weights [d_in, d_out]; canonical orientation is the stored one."""
    mats = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    conv1d = True

    def __init__(self, model):
        self.model = model

    def blocks(self):
        return self.model.transformer.h

    @torch.no_grad()
    def forward_from(self, h, L, ids):
        m = self.model
        for blk in m.transformer.h[L:]:
            out = blk(h)
            h = out[0] if isinstance(out, tuple) else out
        h = m.transformer.ln_f(h)
        logits = m.lm_head(h)
        return nll_of(logits, ids)


class LlamaAdapter:
    """nn.Linear weights [d_out, d_in]; canonical [d_in, d_out] view is the transpose."""
    mats = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    conv1d = False

    def __init__(self, model):
        self.model = model

    def blocks(self):
        return self.model.model.layers

    def _pos(self, h):
        T = h.shape[1]
        pos_ids = torch.arange(T, device=h.device)[None]
        return self.model.model.rotary_emb(h, pos_ids)

    @torch.no_grad()
    def forward_from(self, h, L, ids):
        m = self.model
        pe = self._pos(h)
        for blk in m.model.layers[L:]:
            out = blk(h, attention_mask=None, position_embeddings=pe)
            h = out[0] if isinstance(out, tuple) else out
        h = m.model.norm(h)
        logits = m.lm_head(h)
        return nll_of(logits, ids)


def make_adapter(model):
    kind = model.config.model_type
    if kind == "gpt2":
        return GPT2Adapter(model)
    if kind in ("llama", "mistral", "qwen2"):
        return LlamaAdapter(model)
    raise ValueError(f"no adapter for model_type={kind}")


def get_mod(block, name):
    m = block
    for part in name.split("."):
        m = getattr(m, part)
    return m


def nll_of(logits, ids):
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return (-lp.gather(-1, ids[:, 1:, None])[..., 0]).mean(dim=1)


# ------------------------------------------------------- stats + quantizers
def canon(W, conv1d):
    """Canonical [d_in, d_out] view."""
    return W if conv1d else W.t()


def weight_stats(Wc, group=128):
    """Wc canonical [d_in, d_out]."""
    rms = Wc.pow(2).mean().sqrt().item()
    amax = Wc.abs().max().item()
    amax_c = Wc.abs().amax(dim=0)
    n2_c = Wc.pow(2).sum(dim=0)
    rms_c = (n2_c / Wc.shape[0]).sqrt()
    kc = amax_c / rms_c
    kappa_eff = float((kc.pow(2) * n2_c).sum().sqrt() / n2_c.sum().sqrt())
    ng = Wc.shape[0] // group
    kg_stats = {}
    if ng >= 1:
        Wg = Wc[: ng * group].reshape(ng, group, -1)
        kg = (Wg.abs().amax(dim=1) / Wg.pow(2).mean(dim=1).sqrt()).flatten()
        kg_stats = dict(kg_med=float(kg.median()), kg_p90=float(torch.quantile(kg, 0.90)),
                        kg_max=float(kg.max()))
    return dict(rms=rms, amax=amax, kappa=amax / rms, kappa_eff=kappa_eff,
                kc_max=float(kc.max()), **kg_stats)


def damage(Wc, fam, knob, gen, group=128):
    """Return damaged canonical weight. fam in {gaussian, rtn, rtn_pc, rtn_gw}."""
    if fam == "gaussian":
        rms = Wc.pow(2).mean().sqrt()
        z = torch.randn(Wc.shape, generator=gen, device="cpu").to(Wc.device, Wc.dtype)
        return Wc + knob * rms * z
    qmax = 2 ** (int(knob) - 1) - 1
    if fam == "rtn":
        s = Wc.abs().max() / qmax
        return torch.clamp(torch.round(Wc / s), -qmax, qmax) * s
    if fam == "rtn_pc":
        s = (Wc.abs().amax(dim=0, keepdim=True) / qmax).clamp_min(1e-12)
        return torch.clamp(torch.round(Wc / s), -qmax, qmax) * s
    if fam == "rtn_gw":
        d_in = Wc.shape[0]
        ng = max(d_in // group, 1)
        g = d_in // ng
        assert ng * g == d_in, f"d_in={d_in} not divisible into groups of {group}"
        Wg = Wc.reshape(ng, g, -1)
        s = (Wg.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)
        return (torch.clamp(torch.round(Wg / s), -qmax, qmax) * s).reshape(Wc.shape)
    raise ValueError(fam)


# ----------------------------------------------------------------- caching
@torch.no_grad()
def clean_pass(adapter, ids, device, collect_pr=True, batch=8):
    model, blocks = adapter.model, adapter.blocks()
    caches = [[] for _ in blocks]
    covs, counts = {}, {}
    hs = []
    if collect_pr:
        for L, blk in enumerate(blocks):
            for name in adapter.mats:
                key = (L, name)

                def mk(key):
                    def hook(m, inp, out):
                        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                        covs[key] = covs.get(key, 0) + x.T @ x
                        counts[key] = counts.get(key, 0) + x.shape[0]
                    return hook
                hs.append(get_mod(blk, name).register_forward_hook(mk(key)))
    for L, blk in enumerate(blocks):
        def mk_pre(L):
            def pre(m, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                caches[L].append(h.detach())
            return pre
        hs.append(blk.register_forward_pre_hook(mk_pre(L), with_kwargs=True))
    nll_clean = []
    for i in range(0, ids.shape[0], batch):
        x = ids[i:i + batch].to(device)
        out = model(x)
        nll_clean.append(nll_of(out.logits, x).cpu())
    for h in hs:
        h.remove()
    caches = [torch.cat(c, dim=0) for c in caches]
    pr = {}
    for key, C in covs.items():
        C = (C / counts[key]).float()
        pr[key] = float(torch.trace(C) ** 2 / (C ** 2).sum())
    return torch.cat(nll_clean), caches, pr


def mechanics_check(device="cpu"):
    """Random-init tiny Llama + random ids: the prefix-cache path must equal the full forward."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=512,
                      max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = torch.randint(0, 512, (4, 128))
    ad = make_adapter(model)
    nll_clean, caches, pr = clean_pass(ad, ids, device)
    for L in range(cfg.num_hidden_layers):
        chk = ad.forward_from(caches[L].to(device), L, ids.to(device)).cpu()
        assert torch.allclose(chk, nll_clean, atol=2e-3), (L, chk.mean(), nll_clean.mean())
    # a damage eval must run end to end on every family
    blk = ad.blocks()[1]
    mod = get_mod(blk, "mlp.gate_proj")
    W0 = mod.weight.detach().clone()
    gen = torch.Generator(device="cpu").manual_seed(7)
    for fam, knob in [("gaussian", 0.1), ("rtn", 4), ("rtn_pc", 4), ("rtn_gw", 4)]:
        Wd = damage(canon(W0, ad.conv1d), fam, knob, gen)
        mod.weight.copy_(Wd if ad.conv1d else Wd.t())
        ad.forward_from(caches[1].to(device), 1, ids.to(device))
        mod.weight.copy_(W0)
    print("mechanics-check PASSED: llama prefix cache == full forward; all 4 damage families ran")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--tokens", type=int, default=49152)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ids", default=os.path.join(RES, "wikitext2_test_gpt2_ids.npy"),
                    help="npy of token ids for the model's own tokenizer")
    ap.add_argument("--sigmas", default="0.03,0.045,0.07,0.1,0.15,0.22,0.33,0.5,0.75")
    ap.add_argument("--bits", default="3,4,5,6")
    ap.add_argument("--families", default="gaussian,rtn", help="subset of gaussian,rtn,rtn_pc,rtn_gw")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--stats-only", action="store_true",
                    help="kappa-only spike: weight statistics for every matrix, no damage evals")
    ap.add_argument("--mechanics-check", action="store_true")
    ap.add_argument("--out", default=os.path.join(RES, "lm2_layers.parquet"))
    args = ap.parse_args()
    if args.mechanics_check:
        mechanics_check()
        return
    from transformers import AutoModelForCausalLM
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ad = make_adapter(model)
    blocks = ad.blocks()
    torch.manual_seed(0)

    if args.stats_only:
        rows = []
        for L, blk in enumerate(blocks):
            for name in ad.mats:
                Wc = canon(get_mod(blk, name).weight.detach(), ad.conv1d)
                rows.append(dict(block=L, matrix=name, d_in=Wc.shape[0], d_out=Wc.shape[1],
                                 **weight_stats(Wc.float(), args.group)))
        out = args.out.replace(".parquet", "_stats.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"stats-only: wrote {len(rows)} rows to {out}")
        return

    ids_all = np.load(args.ids)
    nseq = args.tokens // args.seqlen
    ids = torch.from_numpy(ids_all[: nseq * args.seqlen].astype(np.int64)).view(nseq, args.seqlen)
    t0 = time.time()
    nll_clean, caches, pr = clean_pass(ad, ids, args.device, batch=args.batch)
    print(f"clean NLL {nll_clean.mean():.4f} over {nseq*args.seqlen} tokens; clean pass {time.time()-t0:.1f}s", flush=True)
    chk = torch.cat([ad.forward_from(caches[0][i:i + args.batch].to(args.device), 0,
                                     ids[i:i + args.batch].to(args.device)).cpu()
                     for i in range(0, nseq, args.batch)])
    assert torch.allclose(chk, nll_clean, atol=2e-3), (chk.mean(), nll_clean.mean())
    print("prefix-cache path verified against full forward", flush=True)

    sigmas = [float(x) for x in args.sigmas.split(",")]
    bits = [int(x) for x in args.bits.split(",")] if args.bits else []
    fams = args.families.split(",")
    which = [int(x) for x in args.blocks.split(",")] if args.blocks else list(range(len(blocks)))
    levels = ([("gaussian", s) for s in sigmas] if "gaussian" in fams else []) + \
             sum([[(f, b) for b in bits] for f in ("rtn", "rtn_pc", "rtn_gw") if f in fams], [])
    rows = []
    for L in which:
        blk = blocks[L]
        hL = caches[L]
        for name in ad.mats:
            mod = get_mod(blk, name)
            W0 = mod.weight.detach().clone()
            Wc0 = canon(W0, ad.conv1d)
            st = weight_stats(Wc0.float(), args.group)
            base = dict(block=L, matrix=name, d_in=Wc0.shape[0], d_out=Wc0.shape[1],
                        PR=pr[(L, name)], F_hat=Wc0.shape[0] / pr[(L, name)],
                        n_tokens=nseq * args.seqlen, **st)
            for fam, knob in levels:
                gen = torch.Generator(device="cpu").manual_seed((hash((L, name, fam, round(knob * 1e6))) & 0x7FFFFFFF))
                Wd = damage(Wc0, fam, knob, gen, args.group)
                sig_rel = ((Wd - Wc0).float().pow(2).mean().sqrt() / st["rms"]).item()
                mod.weight.copy_(Wd if ad.conv1d else Wd.t())
                t1 = time.time()
                nll = torch.cat([ad.forward_from(hL[i:i + args.batch].to(args.device), L,
                                                 ids[i:i + args.batch].to(args.device)).cpu()
                                 for i in range(0, nseq, args.batch)])
                mod.weight.copy_(W0)
                d = nll - nll_clean
                rows.append(dict(base, family=fam, knob=float(knob), sigma_rel=sig_rel,
                                 dnll=float(d.mean()), dnll_sem=float(d.std() / math.sqrt(len(d))),
                                 nll=float(nll.mean()), eval_s=time.time() - t1))
            print(f"block {L:2d} {name:20s} κ={st['kappa']:6.1f} κ_eff={st['kappa_eff']:5.1f} "
                  f"κg_max={st.get('kg_max', float('nan')):6.1f} | "
                  + " ".join(f"{r['knob']:.3g}:{r['dnll']:.3f}" for r in rows[-len(levels):])
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_parquet(args.out, index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
