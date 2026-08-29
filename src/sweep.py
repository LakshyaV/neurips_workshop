"""Experiment registry + runner. Each experiment writes results/<exp>.parquet (per-feature rows),
results/<exp>_trainlog.parquet and checkpoints results/ckpt/<exp>_m{m}_n{n}[_tag].pt."""
import argparse, itertools, os, sys, time, math
import pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.train import train_group
from src.quant import MODE_UNIFORM, MODE_DITHER, MODE_SR, MODE_PERCH

INF = float("inf")
BITS_MAIN = [2, 3, 4, 5, 6, 7, 8, INF]
P_MAIN = [0.002, 0.005, 0.02, 0.05]
SEEDS = [0, 1, 2, 3]
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def specs_product(ps, bitss, seeds, modes=(MODE_UNIFORM,), alphas=(0.0,), kclips=(INF,)):
    out = []
    for p, b, s, mo, a, kc in itertools.product(ps, bitss, seeds, modes, alphas, kclips):
        if math.isinf(b) and mo != MODE_UNIFORM:
            continue                                   # fp32 arm is mode-independent: keep one copy
        out.append(dict(p=p, bits=float(b), seed=s, mode=mo, alpha=a, kclip=kc))
    return out


def groups_for(exp, args):
    """Yield (tag, m, n, specs, steps) for an experiment."""
    if exp == "E0":      # fp32 replication: uniform importance, n=400, m=30, sparsity sweep
        ps = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]
        yield "E0_m30_n400", 30, 400, specs_product(ps, [INF], [0, 1]), 10000
    elif exp == "pilot":  # sizing gate: the pilot config + the largest group, kappa trajectory
        yield "pilot_m10_n80", 10, 80, specs_product([0.02], BITS_MAIN, [0]), args.steps
        yield "pilot_m16_n2048", 16, 2048, specs_product(P_MAIN, BITS_MAIN, SEEDS), args.steps
    elif exp == "E1":
        for m in [10, 16]:
            for r in [4, 8, 16, 32, 64, 128]:
                yield f"E1_m{m}_n{m*r}", m, m * r, specs_product(P_MAIN, BITS_MAIN, SEEDS), args.steps
    elif exp == "E2":    # lattice controls (QAT): dithered + stochastic rounding + per-channel, m=16 (reduced)
        for r in [8, 32, 128]:
            yield (f"E2_m16_n{16*r}", 16, 16 * r,
                   specs_product([0.005, 0.02], BITS_MAIN, [0, 1], modes=(MODE_DITHER, MODE_SR, MODE_PERCH, MODE_UNIFORM)),
                   args.steps)
    elif exp == "E5":    # power-law importance, prefix fraction
        for r in [8, 32]:
            yield (f"E5_m16_n{16*r}", 16, 16 * r,
                   specs_product([0.005, 0.02], BITS_MAIN, SEEDS, alphas=(0.5, 1.0, 2.0)), args.steps)
    elif exp == "E6":    # kappa manipulation at fixed (m, n, p): weight clipping at c*rms
        for r in [16, 64]:
            yield (f"E6_m16_n{16*r}", 16, 16 * r,
                   specs_product([0.005, 0.02], BITS_MAIN, SEEDS, kclips=(2.0, 3.0, 4.0, INF)), args.steps)
    else:
        raise ValueError(exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--only", default=None, help="substring filter on group tag")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(os.path.join(RES, "ckpt"), exist_ok=True)
    all_rows, all_logs = [], []
    t0 = time.time()
    for tag, m, n, specs, steps in groups_for(args.exp, args):
        if args.only and args.only not in tag:
            continue
        B = args.batch if n < 1024 else min(args.batch, 512)
        print(f"=== {tag}: K={len(specs)} models, m={m}, n={n}, steps={steps}, B={B}", flush=True)
        tg = time.time()
        rows, log, _ = train_group(specs, m, n, steps=steps, B=B, device=args.device, exp=args.exp,
                                   log_every=args.log_every, ckpt_path=os.path.join(RES, "ckpt", tag + ".pt"))
        rows["group"] = tag; log["group"] = tag
        all_rows.append(rows); all_logs.append(log)
        # write incrementally so partial sweeps are usable
        pd.concat(all_rows, ignore_index=True).to_parquet(os.path.join(RES, f"{args.exp}.parquet"), index=False)
        pd.concat(all_logs, ignore_index=True).to_parquet(os.path.join(RES, f"{args.exp}_trainlog.parquet"), index=False)
        print(f"=== {tag} done in {time.time()-tg:.0f}s (total {time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
