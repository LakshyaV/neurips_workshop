"""Regenerate per-feature result rows from saved checkpoints (eval only), for groups whose parquet rows were lost."""
import glob, os, sys, math
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.train import eval_rho_grouped
from src.metrics import feature_dims, kappa
from src.quant import MODE_NAMES
RES = os.path.join(os.path.dirname(__file__), "..", "results")

def regen(path, device="mps"):
    ck = torch.load(path); specs, m, n, keys = ck["specs"], ck["m"], ck["n"], ck["keys"]; K = len(specs)
    W, b, Wq = ck["W"].to(device), ck["b"].to(device), ck["Wq"].to(device)
    p_G = torch.tensor([k[0] for k in keys], dtype=torch.float32)
    gidx = torch.tensor([keys.index((float(s["p"]), int(s["seed"]))) for s in specs], device=device)
    ev = int(max(40000, 400.0 / float(p_G.min()))); chunk = int(max(128, min(1024, 2 ** 28 // (K * n))))
    rho, nu = eval_rho_grouped(Wq, b, p_G, gidx, ev, chunk=chunk)
    D = feature_dims(Wq); kl, ke = kappa(W), kappa(Wq); sig = (Wq - W).pow(2).mean(dim=(1, 2)).sqrt()
    amax = W.abs().amax(dim=(1, 2)); rms = W.pow(2).mean(dim=(1, 2)).sqrt(); wn = Wq.norm(dim=1); nret = (rho >= 0.5).sum(1)
    bits = torch.tensor([float(s["bits"]) for s in specs]); qm = torch.where(torch.isinf(bits), bits, 2.0 ** (bits - 1) - 1)
    delta = torch.where(torch.isinf(qm), torch.zeros_like(qm), amax.cpu() / torch.where(torch.isinf(qm), torch.ones_like(qm), qm))
    frames = []
    for k, s in enumerate(specs):
        run_id = (f"{ck['exp']}_m{m}_n{n}_p{s['p']}_b{s['bits']}_mode{int(s['mode'])}_a{float(s.get('alpha', 0.0))}"
                  f"_kc{float(s.get('kclip', float('inf')))}_s{int(s['seed'])}")
        i = torch.arange(n, dtype=torch.float32); a = float(s.get("alpha", 0.0)); imp = ((i + 1) ** (-a)) if a > 0 else torch.ones(n)
        frames.append(pd.DataFrame(dict(run_id=run_id, exp=ck["exp"], m=m, n=n, p=float(s["p"]), alpha=a, bits=float(s["bits"]), mode=int(s["mode"]),
            mode_name=MODE_NAMES[int(s["mode"])], damage_type="qat", knob=float(s["bits"]), seed=int(s["seed"]), kclip=float(s.get("kclip", float("inf"))),
            feature_idx=np.arange(n), importance=imp.numpy(), rho_i=rho[k].cpu().numpy(), D_i=D[k].cpu().numpy(), w_norm=wn[k].cpu().numpy(),
            kappa=float(kl[k]), kappa_eff=float(ke[k]), sigma_eps_measured=float(sig[k]), delta=float(delta[k]), amax=float(amax[k]), rms_w=float(rms[k]),
            loss_final=float("nan"), n_ret=int(nret[k]), steps=6000, batch=(1024 if n < 1024 else 512), eval_samples=ev, device=device, group=os.path.basename(path)[:-3])))
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    out = sys.argv[1]; paths = sorted(glob.glob(sys.argv[2]))
    frames = []
    for p in paths:
        print("regen", p, flush=True); frames.append(regen(p))
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    print("ALL DONE", flush=True)
