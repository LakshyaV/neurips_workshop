"""QuaRot-style R1 residual-stream rotation for Llama-class models (v2 Phase 2 rotation arm).

Two steps, both exact at full precision:
  1. fold_norms: fold every RMSNorm gain gamma into the following linear layers
     (input_layernorm -> q/k/v, post_attention_layernorm -> gate/up, final norm -> lm_head),
     leaving unit-gain norms. RMSNorm with unit gain commutes with any orthogonal Q
     (||xQ|| = ||x||), which step 2 needs.
  2. rotate_r1: pick an orthogonal Q (randomized Hadamard, d a power of two) and fuse it:
       embed_tokens.weight <- E Q            (residual stream becomes xQ)
       W <- W Q   for residual-consuming inputs (q,k,v,gate,up): out = x W^T must be unchanged
                   with x' = xQ, so W'^T = Q^T W^T, i.e. stored [out,in] weight W' = W Q;
       W <- Q^T W for residual-writing outputs (o_proj, down_proj): out' = out Q on the out-axis
                   of the stored [out,in] weight.
       lm_head.weight <- lm_head.weight @ Q  (consumes the rotated, norm-folded residual)
     The model's function is unchanged (exactness test: max |logits_before - logits_after|),
     while every residual-facing weight matrix is rotated, which is what changes kappa.

What this is for: measure kappa (and kappa_g) before/after rotation, re-run RTN cliffs on the
rotated model, and test bits saved against (1-e) * log2(kappa_before / kappa_after) (compensated)
vs log2(kappa_before / kappa_after) (naive). R2/R3/R4 online rotations from QuaRot are NOT
implemented: the weight-law test only needs the residual-stream (R1) weight rotation.

Run `python -m src.rotate --exactness` for the no-download random-init test.
"""
import argparse
import torch


def hadamard(d, device="cpu", dtype=torch.float64):
    """Sylvester Hadamard matrix, d a power of two, scaled orthogonal (H H^T = I)."""
    assert d & (d - 1) == 0, f"d={d} is not a power of two"
    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / torch.sqrt(torch.tensor(float(d), dtype=dtype, device=device))


def random_hadamard(d, seed=0, device="cpu", dtype=torch.float64):
    """Randomized Hadamard: H diag(s), s random signs — orthogonal, outlier-spreading."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    s = (torch.randint(0, 2, (d,), generator=gen, dtype=torch.int64) * 2 - 1).to(dtype).to(device)
    return hadamard(d, device, dtype) * s[None, :]


@torch.no_grad()
def fold_norms(model):
    """Fold RMSNorm gains into adjacent linears; norms become unit-gain. Llama-class only."""
    for layer in model.model.layers:
        g = layer.input_layernorm.weight.data.double()
        for name in ("q_proj", "k_proj", "v_proj"):
            w = getattr(layer.self_attn, name).weight
            w.data = (w.data.double() * g[None, :]).to(w.dtype)      # [out,in] * gamma on in-axis
        layer.input_layernorm.weight.data.fill_(1.0)
        g = layer.post_attention_layernorm.weight.data.double()
        for name in ("gate_proj", "up_proj"):
            w = getattr(layer.mlp, name).weight
            w.data = (w.data.double() * g[None, :]).to(w.dtype)
        layer.post_attention_layernorm.weight.data.fill_(1.0)
    g = model.model.norm.weight.data.double()
    if model.config.tie_word_embeddings:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
        model.config.tie_word_embeddings = False
    model.lm_head.weight.data = (model.lm_head.weight.data.double() * g[None, :]).to(model.lm_head.weight.dtype)
    model.model.norm.weight.data.fill_(1.0)


@torch.no_grad()
def rotate_r1(model, Q):
    """Fuse orthogonal Q (double precision, [d, d]) into all residual-facing weights."""
    Q = Q.double()
    emb = model.model.embed_tokens.weight
    emb.data = (emb.data.double() @ Q).to(emb.dtype)
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj"):
            w = getattr(layer.self_attn, name).weight
            w.data = (w.data.double() @ Q).to(w.dtype)               # consumes residual: W' = W Q
        w = layer.self_attn.o_proj.weight
        w.data = (Q.t() @ w.data.double()).to(w.dtype)               # writes residual: out' = out Q  =>  W' = Q^T W
        for name in ("gate_proj", "up_proj"):
            w = getattr(layer.mlp, name).weight
            w.data = (w.data.double() @ Q).to(w.dtype)
        w = layer.mlp.down_proj.weight
        w.data = (Q.t() @ w.data.double()).to(w.dtype)
    lm = model.lm_head.weight
    lm.data = (lm.data.double() @ Q).to(lm.dtype)


RESIDUAL_MATS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def kappa_report(model, group=128):
    """Per-matrix kappa and kappa_g,max, for before/after comparisons."""
    rows = []
    for L, layer in enumerate(model.model.layers):
        for name in RESIDUAL_MATS:
            m = layer
            for part in name.split("."):
                m = getattr(m, part)
            Wc = m.weight.detach().float().t()                        # canonical [in, out]
            rms = Wc.pow(2).mean().sqrt()
            ng = max(Wc.shape[0] // group, 1)
            g = Wc.shape[0] // ng
            Wg = Wc[: ng * g].reshape(ng, g, -1)
            kg = (Wg.abs().amax(dim=1) / Wg.pow(2).mean(dim=1).sqrt())
            rows.append(dict(block=L, matrix=name, kappa=float(Wc.abs().max() / rms),
                             kg_max=float(kg.max())))
    return rows


def exactness_test():
    """Random-init tiny Llama: fold + rotate must leave the logits unchanged (fp32 tolerance)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=512,
                      max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # make the norm gains non-trivial so folding is actually exercised
    for layer in model.model.layers:
        layer.input_layernorm.weight.data.uniform_(0.5, 1.5)
        layer.post_attention_layernorm.weight.data.uniform_(0.5, 1.5)
    model.model.norm.weight.data.uniform_(0.5, 1.5)
    ids = torch.randint(0, 512, (2, 64))
    with torch.no_grad():
        ref = model(ids).logits
    fold_norms(model)
    with torch.no_grad():
        fold_diff = (model(ids).logits - ref).abs().max().item()
    k0 = kappa_report(model)
    Q = random_hadamard(cfg.hidden_size, seed=1)
    rotate_r1(model, Q)
    with torch.no_grad():
        rot_diff = (model(ids).logits - ref).abs().max().item()
    k1 = kappa_report(model)
    import numpy as np
    dk = np.log2([a["kappa"] for a in k0]) - np.log2([b["kappa"] for b in k1])
    print(f"fold-only  max |dlogits| = {fold_diff:.2e}")
    print(f"fold+R1    max |dlogits| = {rot_diff:.2e}")
    print(f"log2 kappa reduction across {len(dk)} matrices: mean {dk.mean():+.2f}, range [{dk.min():+.2f}, {dk.max():+.2f}]")
    assert fold_diff < 1e-3 and rot_diff < 1e-3, "rotation is not function-preserving"
    print("EXACTNESS TEST PASSED (random-init model; kappa shifts here are meaningless)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exactness", action="store_true")
    args = ap.parse_args()
    if args.exactness:
        exactness_test()
