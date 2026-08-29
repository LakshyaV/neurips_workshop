"""§5C: κ-guided mixed-precision allocation in GPT-2 small.
Allocations of per-matrix bit-widths b_i ∈ {3,4,5,6,8} at equal parameter-weighted average bits, evaluated end-to-end
(all 48 matrices quantized simultaneously with per-tensor RTN) on HELD-OUT tokens (different from the sweep tokens):
  uniform  : every matrix at b
  kappa    : b_i = c + log2 κ_i (the corrected law), c chosen to meet the budget, rounded to available widths
  oracle   : greedy on the measured single-matrix ΔNLL(b) table from the sweep tokens (sensitivity-based)
  random   : random permutations of the κ allocation (same multiset of bit-widths)"""
import argparse, os, sys, math, itertools
import numpy as np, pandas as pd, torch
from transformers import GPT2LMHeadModel
from .lm_eval import get_mod, MATS

RES = os.path.join(os.path.dirname(__file__), "..", "results")
WIDTHS = [3, 4, 5, 6, 8]


def rtn(W, bits):
    qmax = 2 ** (bits - 1) - 1; s = W.abs().max() / qmax
    return torch.clamp(torch.round(W / s), -qmax, qmax) * s


def rtn_pc(W, bits):
    """per output-channel (Conv1D weight [in, out]: scale per column) symmetric RTN"""
    qmax = 2 ** (bits - 1) - 1; s = (W.abs().amax(dim=0, keepdim=True) / qmax).clamp_min(1e-12)
    return torch.clamp(torch.round(W / s), -qmax, qmax) * s


def kappa_eff(W):
    amax_c = W.abs().amax(dim=0, keepdim=True); n2_c = W.pow(2).sum(dim=0, keepdim=True); rms_c = (n2_c / W.shape[0]).sqrt()
    return float(((amax_c / rms_c) ** 2 * n2_c).sum().sqrt() / n2_c.sum().sqrt())


@torch.no_grad()
def eval_nll(model, ids, device):
    out = []
    for i in range(0, ids.shape[0], 8):
        x = ids[i:i + 8].to(device)
        lp = torch.log_softmax(model(x).logits[:, :-1].float(), dim=-1)
        out.append((-lp.gather(-1, x[:, 1:, None])[..., 0]).mean(dim=1).cpu())
    return torch.cat(out)


def round_to_budget(cont, params, budget):
    """Round continuous bit targets to WIDTHS, then greedily adjust to meet the param-weighted budget."""
    b = np.array([min(WIDTHS, key=lambda w: abs(w - c)) for c in cont], dtype=float)
    def avg(b): return float((b * params).sum() / params.sum())
    # decrease the largest-slack matrices while over budget; increase while under
    while avg(b) > budget + 1e-9:
        cand = [i for i in range(len(b)) if b[i] > WIDTHS[0]]
        i = max(cand, key=lambda i: (b[i] - cont[i]) * params[i])            # most over its continuous target
        b[i] = WIDTHS[WIDTHS.index(int(b[i])) - 1]
    while True:
        cand = [i for i in range(len(b)) if b[i] < WIDTHS[-1]]
        if not cand: break
        i = max(cand, key=lambda i: (cont[i] - b[i]))
        nb = WIDTHS[WIDTHS.index(int(b[i])) + 1]
        trial = b.copy(); trial[i] = nb
        if avg(trial) <= budget + 1e-9: b = trial
        else: break
    return b


def oracle_alloc(dtab, params, budget):
    """Greedy knapsack on measured single-matrix ΔNLL: start at 3 bits, upgrade best ΔNLL-reduction per bit·param."""
    n = len(params); b = np.full(n, 3.0)
    def cost(i, w): return dtab[i].get(w, dtab[i][max(dtab[i])])          # ΔNLL at width w (8 bits ≈ 6-bit value if missing)
    def avg(b): return float((b * params).sum() / params.sum())
    while True:
        best, bi, bw = 0.0, None, None
        for i in range(n):
            if b[i] >= WIDTHS[-1]: continue
            nw = WIDTHS[WIDTHS.index(int(b[i])) + 1]
            gain = (cost(i, b[i]) - cost(i, nw)) / ((nw - b[i]) * params[i])
            trial = b.copy(); trial[i] = nw
            if avg(trial) <= budget + 1e-9 and gain > best:
                best, bi, bw = gain, i, nw
        if bi is None: break
        b[bi] = bw
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps"); ap.add_argument("--tokens", type=int, default=49152)
    ap.add_argument("--offset", type=int, default=98304, help="token offset for the held-out evaluation")
    ap.add_argument("--budgets", default="3.5,4.0,4.5,5.0"); ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--quant", default="pc", help="pc (per-channel) or pt (per-tensor)")
    ap.add_argument("--sweep", default=os.path.join(RES, "lm_layers_pc.parquet"))
    ap.add_argument("--out", default=os.path.join(RES, "lm_mixed.parquet"))
    args = ap.parse_args()
    device = args.device
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).to(device).eval()
    for prm in model.parameters(): prm.requires_grad_(False)
    ids_all = np.load(os.path.join(RES, "wikitext2_test_gpt2_ids.npy"))
    nseq = args.tokens // 1024
    ids = torch.from_numpy(ids_all[args.offset: args.offset + nseq * 1024].astype(np.int64)).view(nseq, 1024)
    sweep = pd.read_parquet(args.sweep)
    noise = pd.read_parquet(os.path.join(RES, "lm_layers.parquet"))
    thr = pd.read_csv(os.path.join(RES, "lm_layer_thresholds.csv"))
    Q = rtn_pc if args.quant == "pc" else rtn
    fam = "rtn_pc" if args.quant == "pc" else "rtn"
    mats = [(L, name) for L in range(len(model.transformer.h)) for name in MATS]
    mods = [get_mod(model.transformer.h[L], name) for L, name in mats]
    W0 = [m.weight.detach().clone() for m in mods]
    params = np.array([w.numel() for w in W0], dtype=float)
    kappa = np.array([(kappa_eff(w) if args.quant == "pc" else (w.abs().max() / w.pow(2).mean().sqrt()).item()) for w in W0])
    sig = np.array([float(thr[(thr.block == L) & (thr.matrix == name)]["sigma_crit_0.05"].iloc[0]) for L, name in mats])
    law = 1 + np.log2(1 + kappa / (math.sqrt(12) * sig))          # per-matrix b_crit from the corrected law (needs the noise sweep)
    dtab = []
    for L, name in mats:
        g = sweep[(sweep.block == L) & (sweep.matrix == name) & (sweep.family == fam)]
        d = {int(k): float(v) for k, v in zip(g.knob, g.dnll)}
        d[8] = 0.0
        dtab.append(d)
    print("kappa_eff range", kappa.min().round(2), kappa.max().round(2), "| law b_crit range", law.min().round(2), law.max().round(2), flush=True)
    nll0 = eval_nll(model, ids, device).mean().item()
    print(f"held-out clean NLL {nll0:.4f} on {nseq*1024} tokens (offset {args.offset})", flush=True)
    rows = []
    rng = np.random.default_rng(0)

    def run(name, b, budget):
        for m, w0, bi in zip(mods, W0, b): m.weight.copy_(Q(w0, int(bi)))
        nll = eval_nll(model, ids, device)
        for m, w0 in zip(mods, W0): m.weight.copy_(w0)
        d = nll.mean().item() - nll0
        avgb = float((b * params).sum() / params.sum())
        rows.append(dict(alloc=name, budget=budget, avg_bits=avgb, dnll=d, sem=float(nll.std() / math.sqrt(len(nll))),
                         bits=",".join(str(int(x)) for x in b)))
        print(f"  {name:10s} budget {budget:.2f} avg_bits {avgb:.3f}  ΔNLL {d:+.4f}", flush=True)
        return d

    for budget in [float(x) for x in args.budgets.split(",")]:
        if float(budget).is_integer() and int(budget) in WIDTHS:
            run("uniform", np.full(len(mats), float(budget)), budget)
        # kappa rule: b_i = c + log2 kappa_i
        lo, hi = -10.0, 20.0
        for _ in range(60):
            c = 0.5 * (lo + hi); cont = c + np.log2(kappa)
            if float((np.clip(cont, WIDTHS[0], WIDTHS[-1]) * params).sum() / params.sum()) > budget: hi = c
            else: lo = c
        cont = np.clip(lo + np.log2(kappa), WIDTHS[0], WIDTHS[-1])
        bk = round_to_budget(cont, params, budget)
        run("kappa", bk, budget)
        # law rule: shift the per-matrix law prediction by a common offset to meet the budget
        lo2, hi2 = -10.0, 10.0
        for _ in range(60):
            c = 0.5 * (lo2 + hi2)
            if float((np.clip(law + c, WIDTHS[0], WIDTHS[-1]) * params).sum() / params.sum()) > budget: hi2 = c
            else: lo2 = c
        run("law", round_to_budget(np.clip(law + lo2, WIDTHS[0], WIDTHS[-1]), params, budget), budget)
        run("oracle", oracle_alloc(dtab, params, budget), budget)
        for r in range(args.n_random):
            perm = rng.permutation(len(bk))
            run(f"random{r}", bk[perm], budget)
        pd.DataFrame(rows).to_parquet(args.out, index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
