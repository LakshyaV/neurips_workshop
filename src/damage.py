"""E3: post-hoc damage on the fp32 (b=inf) arms of trained checkpoints. All damage families are
applied to the same weights; the learned bias is kept (PTQ-style). Writes results/E3.parquet."""
import argparse, glob, os, sys, math, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.quant import (fake_quant, qmax_from_bits, damage_gaussian, damage_rank_truncate, damage_column_dropout,
                       MODE_UNIFORM, MODE_DITHER, MODE_PERCH, MODE_SR)
from src.train import eval_rho_grouped
from src.metrics import feature_dims, kappa

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FAMILIES = {
    "ptq_uniform":   [2, 3, 4, 5, 6, 7, 8],
    "ptq_dither":    [2, 3, 4, 5, 6, 7, 8],
    "ptq_perchannel": [2, 3, 4, 5, 6, 7, 8],
    "ptq_stochastic": [2, 3, 4, 5, 6, 7, 8],
    "gaussian":      list(np.logspace(-2.5, 0, 16)),
    "rank":          None,                                   # 1..m
    "dropout":       [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5],
}
DRAWS = {"ptq_uniform": 1, "ptq_dither": 2, "ptq_perchannel": 1, "ptq_stochastic": 2, "gaussian": 2, "rank": 1, "dropout": 2}


def apply(family, W, knob, draw):
    K, m, n = W.shape
    gen = torch.Generator(device="cpu").manual_seed(10_000 + draw)
    if family.startswith("ptq"):
        qmax = qmax_from_bits(torch.full((K, 1, 1), float(knob))).to(W.device)
        mode = {"ptq_uniform": MODE_UNIFORM, "ptq_dither": MODE_DITHER, "ptq_perchannel": MODE_PERCH,
                "ptq_stochastic": MODE_SR}[family]
        v = torch.rand(K, m, n, generator=gen).to(W.device)
        _, Wd, _ = fake_quant(W, qmax, torch.full((K, 1, 1), mode, dtype=torch.long, device=W.device),
                              dither_v=v, sr_u=v, return_effective=True)
        return Wd
    if family == "gaussian":
        return damage_gaussian(W, float(knob), gen)
    if family == "rank":
        return damage_rank_truncate(W, int(knob)).to(W.device)
    if family == "dropout":
        return damage_column_dropout(W, float(knob), gen)
    raise ValueError(family)


def run_ckpt(path, device, families, out_rows):
    ck = torch.load(path)
    specs, m, n = ck["specs"], ck["m"], ck["n"]
    idx = [k for k, s in enumerate(specs) if math.isinf(float(s["bits"])) and int(s["mode"]) == MODE_UNIFORM
           and float(s.get("alpha", 0.0)) == 0.0 and math.isinf(float(s.get("kclip", float("inf"))))]
    W = ck["W"][idx].to(device); b = ck["b"][idx].to(device)
    sp = [specs[k] for k in idx]
    K0 = len(sp)
    keys = sorted(set((float(s["p"]), int(s["seed"])) for s in sp))
    p_G = torch.tensor([k[0] for k in keys], dtype=torch.float32)
    gidx0 = torch.tensor([keys.index((float(s["p"]), int(s["seed"]))) for s in sp], device=device)
    kap0 = kappa(W).cpu().numpy()
    eval_samples = int(max(40000, 100.0 / float(p_G.min())))
    tag = os.path.basename(path).replace(".pt", "")
    for fam in families:
        knobs = FAMILIES[fam] if FAMILIES[fam] is not None else list(range(1, m + 1))
        t0 = time.time()
        # stack all (knob, draw) levels along K and evaluate in one batched pass (chunked by memory)
        levels = [(kn, d) for kn in knobs for d in range(DRAWS[fam])]
        Wd_all = torch.cat([apply(fam, W, kn, d) for kn, d in levels], dim=0)
        b_all = b.repeat(len(levels), 1)
        gidx_all = gidx0.repeat(len(levels))
        Ktot = Wd_all.shape[0]
        chunk = int(max(64, min(1024, 2 ** 27 // (Ktot * n))))
        rho, nu = eval_rho_grouped(Wd_all, b_all, p_G, gidx_all, eval_samples, chunk=chunk)
        sig_abs = (Wd_all - W.repeat(len(levels), 1, 1)).pow(2).mean(dim=(1, 2)).sqrt()
        rms0 = W.pow(2).mean(dim=(1, 2)).sqrt().repeat(len(levels))
        D = feature_dims(Wd_all)
        rho_np, D_np, wn = rho.cpu().numpy(), D.cpu().numpy(), Wd_all.norm(dim=1).cpu().numpy()
        sig_np, rms_np = sig_abs.cpu().numpy(), rms0.cpu().numpy()
        nret = (rho >= 0.5).sum(1).cpu().numpy()
        for li, (kn, d) in enumerate(levels):
            for j, s in enumerate(sp):
                k = li * K0 + j
                out_rows.append(pd.DataFrame(dict(
                    run_id=f"E3_{fam}_m{m}_n{n}_p{s['p']}_k{kn}_d{d}_s{s['seed']}", exp="E3", group=tag,
                    m=m, n=n, p=float(s["p"]), alpha=0.0, bits=(float(kn) if fam.startswith("ptq") else float("nan")),
                    mode=-1, mode_name=fam, damage_type=fam, knob=float(kn), draw=d, seed=int(s["seed"]),
                    kclip=float("inf"), feature_idx=np.arange(n), importance=1.0, rho_i=rho_np[k], D_i=D_np[k],
                    w_norm=wn[k], kappa=float(kap0[j]), sigma_eps_measured=float(sig_np[k]),
                    sigma_rel=float(sig_np[k] / max(rms_np[k], 1e-12)), rms_w=float(rms_np[k]),
                    n_ret=int(nret[k]), eval_samples=eval_samples, device=str(device))))
        print(f"  {tag} {fam}: {len(levels)} levels x {K0} models in {time.time()-t0:.0f}s; "
              f"N_ret range {nret.min()}..{nret.max()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_glob", default=os.path.join(RES, "ckpt", "E1_*.pt"))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=os.path.join(RES, "E3.parquet"))
    args = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(args.ckpt_glob)):
        print(f"=== {path}", flush=True)
        run_ckpt(path, args.device, args.families.split(","), rows)
        pd.concat(rows, ignore_index=True).to_parquet(args.out, index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
