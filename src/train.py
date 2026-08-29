"""Batched QAT trainer for K toy models with per-model (p, bits, mode, importance, seed, kclip)."""
import math
import time
import numpy as np
import pandas as pd
import torch
from .model import BatchedTMS
from .quant import qmax_from_bits, check_quantizer_health, MODE_SR, MODE_DITHER, MODE_NAMES
from .metrics import feature_dims, kappa, nu0


def importance_vec(n, alpha):
    i = torch.arange(n, dtype=torch.float32)
    return (i + 1.0) ** (-alpha) if alpha > 0 else torch.ones(n)


def seed_device(seed, device):
    torch.manual_seed(seed)
    if str(device).startswith("mps"):
        torch.mps.manual_seed(seed)


@torch.no_grad()
def eval_rho_grouped(Wq, b, p_G, gidx, n_samples, chunk=1024, seed=12345):
    """Monte-Carlo rho on fresh samples; models sharing an x-group index (gidx) share samples."""
    device = Wq.device
    K, m, n = Wq.shape
    G = p_G.numel()
    p_G3 = p_G.view(G, 1, 1).to(device)
    seed_device(seed, device)
    sse = torch.zeros(K, n, device=device)
    done = 0
    while done < n_samples:
        B = min(chunk, n_samples - done)
        r1 = torch.rand(G, B, n, device=device)
        r2 = torch.rand(G, B, n, device=device)
        x = ((r1 < p_G3).to(r2.dtype) * r2)[gidx]
        xh = BatchedTMS.forward_with(Wq, b, x)
        sse += ((x - xh) ** 2).sum(dim=1)
        done += B
    nu = sse / n_samples
    p_K = p_G[gidx.cpu()].view(K, 1).to(device)
    rho = 1.0 - nu / nu0(p_K)
    return rho, nu


def train_group(specs, m, n, steps=4000, B=1024, lr=1e-3, warmup=200, device="mps",
                log_every=100, eval_samples=None, x_seed_base=0, exp="E1", verbose=True,
                ckpt_path=None, damage_type="qat"):
    """specs: list of dict(p, seed, bits, mode, alpha, kclip). Returns (rows_df, log_df, ckpt_dict)."""
    K = len(specs)
    p_K = torch.tensor([s["p"] for s in specs], dtype=torch.float32)
    seeds = [int(s["seed"]) for s in specs]
    bits = torch.tensor([float(s["bits"]) for s in specs], dtype=torch.float32)
    qmax = qmax_from_bits(bits).view(K, 1, 1).to(device)
    mode = torch.tensor([int(s["mode"]) for s in specs], dtype=torch.int64).view(K, 1, 1).to(device)
    alphas = [float(s.get("alpha", 0.0)) for s in specs]
    imp = torch.stack([importance_vec(n, a) for a in alphas]).view(K, 1, n).to(device)
    kclip = torch.tensor([float(s.get("kclip", float("inf"))) for s in specs]).view(K, 1, 1).to(device)
    has_clip = bool(torch.isfinite(kclip).any().item())

    keys = sorted(set((float(s["p"]), int(s["seed"])) for s in specs))
    gidx = torch.tensor([keys.index((float(s["p"]), int(s["seed"]))) for s in specs], device=device)
    G = len(keys)
    p_G = torch.tensor([k[0] for k in keys], dtype=torch.float32)
    p_G3 = p_G.view(G, 1, 1).to(device)

    model = BatchedTMS(K, n, m, init_seeds=seeds, device=device)
    gen_cpu = torch.Generator(device="cpu").manual_seed(999 + x_seed_base)
    need_dither = bool((mode == MODE_DITHER).any().item())
    need_sr = bool((mode == MODE_SR).any().item())
    dither_v = torch.rand(K, m, n, generator=gen_cpu).to(device) if need_dither else None

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        t = (step - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * t))

    seed_device(1000 + x_seed_base, device)      # x stream: differs by seed via slice index in keys
    logs = []
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        r1 = torch.rand(G, B, n, device=device)
        r2 = torch.rand(G, B, n, device=device)
        x = ((r1 < p_G3).to(r2.dtype) * r2)[gidx]
        sr_u = torch.rand(K, m, n, device=device) if need_sr else None
        y = model(x, qmax, mode, dither_v, sr_u)
        loss_k = (imp * (x - y) ** 2).sum(-1).mean(1)
        loss = loss_k.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if has_clip:
            with torch.no_grad():
                rms = model.W.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
                lim = kclip * rms
                model.W.copy_(torch.maximum(torch.minimum(model.W, lim), -lim))
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                kap = kappa(model.W).cpu().numpy()
            lk = loss_k.detach().cpu().numpy()
            for k in range(K):
                logs.append(dict(exp=exp, m=m, n=n, k=k, p=specs[k]["p"], seed=specs[k]["seed"],
                                 bits=float(specs[k]["bits"]), mode=int(specs[k]["mode"]),
                                 step=step, loss=float(lk[k]), kappa=float(kap[k])))
            if verbose and (step % (log_every * 10) == 0 or step == steps - 1):
                print(f"  [{exp} m={m} n={n} K={K}] step {step:5d} loss {float(loss):.4f} "
                      f"kappa(mean) {kap.mean():.2f}  {time.time()-t0:.0f}s", flush=True)
    train_time = time.time() - t0

    # ------------------------------------------------------------------ evaluation
    with torch.no_grad():
        sr_u_eval = (torch.rand(K, m, n, generator=torch.Generator(device="cpu").manual_seed(4242)).to(device)
                     if need_sr else None)
        _, Wq, delta = model.effective_weights(qmax, mode, dither_v, sr_u_eval)
        bad = check_quantizer_health(Wq, qmax)
        if bad:
            raise RuntimeError(f"dead quantizer arms (<=1 distinct value): models {bad}")
        if eval_samples is None:
            eval_samples = int(max(40000, 400.0 / float(p_G.min())))
        chunk = int(max(128, min(1024, 2 ** 28 // (K * n))))
        rho, nu = eval_rho_grouped(Wq, model.b, p_G, gidx, eval_samples, chunk=chunk)
        D = feature_dims(Wq)
        sumD = D.sum(dim=1)
        if bool((sumD > m * 1.02 + 1e-3).any()):
            raise RuntimeError(f"sum_i D_i > m: {sumD.max().item():.3f} > {m}")
        kap_lat = kappa(model.W)
        kap_eff = kappa(Wq)
        W = model.W.detach()
        sig_eps = (Wq - W).pow(2).mean(dim=(1, 2)).sqrt()
        amax = W.abs().amax(dim=(1, 2))
        rms_w = W.pow(2).mean(dim=(1, 2)).sqrt()
        wnorm = Wq.norm(dim=1)                                   # [K,n]
        nret = (rho >= 0.5).sum(1)

    # ------------------------------------------------------------------ rows (one per model x feature)
    rho_np, D_np, wn_np, imp_np = rho.cpu().numpy(), D.cpu().numpy(), wnorm.cpu().numpy(), imp.view(K, n).cpu().numpy()
    kl, ke, se, am, rw, dl = (t.cpu().numpy() for t in (kap_lat, kap_eff, sig_eps, amax, rms_w, delta.view(K)))
    lk = loss_k.detach().cpu().numpy()
    frames = []
    for k, s in enumerate(specs):
        run_id = (f"{exp}_m{m}_n{n}_p{s['p']}_b{s['bits']}_mode{int(s['mode'])}_a{float(s.get('alpha', 0.0))}"
                  f"_kc{float(s.get('kclip', float('inf')))}_s{int(s['seed'])}")
        frames.append(pd.DataFrame(dict(
            run_id=run_id, exp=exp, m=m, n=n, p=float(s["p"]), alpha=float(s.get("alpha", 0.0)),
            bits=float(s["bits"]), mode=int(s["mode"]), mode_name=MODE_NAMES[int(s["mode"])],
            damage_type=damage_type, knob=float(s["bits"]), seed=int(s["seed"]),
            kclip=float(s.get("kclip", float("inf"))),
            feature_idx=np.arange(n), importance=imp_np[k], rho_i=rho_np[k], D_i=D_np[k], w_norm=wn_np[k],
            kappa=float(kl[k]), kappa_eff=float(ke[k]), sigma_eps_measured=float(se[k]),
            delta=float(dl[k]), amax=float(am[k]), rms_w=float(rw[k]), loss_final=float(lk[k]),
            n_ret=int(nret[k]), steps=steps, batch=B, eval_samples=eval_samples, device=str(device))))
    rows = pd.concat(frames, ignore_index=True)
    log_df = pd.DataFrame(logs)
    ckpt = dict(W=model.W.detach().cpu(), b=model.b.detach().cpu(), Wq=Wq.cpu(), specs=specs, m=m, n=n,
                dither_v=(dither_v.cpu() if dither_v is not None else None), keys=keys, exp=exp)
    if ckpt_path is not None:
        torch.save(ckpt, ckpt_path)
    if verbose:
        print(f"  done: train {train_time:.0f}s, eval {eval_samples} samples; "
              f"N_ret by model: {nret.cpu().numpy().tolist()[:16]}...", flush=True)
    return rows, log_df, ckpt
