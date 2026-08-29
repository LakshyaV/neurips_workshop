"""Pre-quantization damage forecasting: how well do fp32 quantities predict what quantization breaks? (ROC AUC)"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
RES = os.path.join(os.path.dirname(__file__), "..", "results")
rows = []
d = pd.read_parquet(os.path.join(RES, "lm_tokens.parquet")); c = d[d.ok_fp32]
for b in [4, 5, 6]:
    y = (~c[f"ok_rtn{b}"].astype(bool)).astype(int)
    rows.append(dict(level="token", bits=b, event="prediction flips", rate=y.mean(), auc_predictor=roc_auc_score(y, -c.margin_fp32), predictor="fp32 logit margin", auc_baseline=roc_auc_score(y, -np.log(c.freq.clip(lower=1))), baseline="token rarity", n=len(y)))
f = pd.read_csv(os.path.join(RES, "lm_sae_features.csv"))
for b in [3, 4, 5]:
    y = (f[f"rho_rtn{b}"] < 0.5).astype(int)
    rows.append(dict(level="SAE feature", bits=b, event="feature erased", rate=y.mean(), auc_predictor=roc_auc_score(y, -np.log2(f.mean_act)), predictor="fp32 magnitude", auc_baseline=roc_auc_score(y, -np.log2(f.freq)), baseline="feature frequency", n=len(y)))
t = pd.read_csv(os.path.join(RES, "E5_per_feature_bits.csv")); y = (t.b_i > 4).astype(int)
rows.append(dict(level="toy feature", bits=4, event="needs > 4 bits", rate=y.mean(), auc_predictor=roc_auc_score(y, -(t.rho_inf - 0.5)), predictor="fp32 margin", auc_baseline=np.nan, baseline="", n=len(y)))
a = pd.read_csv(os.path.join(RES, "lm_act_thresholds.csv")); oka = a[a.censor != "none"]; y = (oka.b_act > 5).astype(int)
rows.append(dict(level="matrix (activations)", bits=5, event="needs > 5 activation bits", rate=y.mean(), auc_predictor=roc_auc_score(y, oka.kappa_act), predictor="activation outlier ratio", auc_baseline=roc_auc_score(y, oka.kappa_w), baseline="weight outlier ratio", n=len(y)))
l = pd.read_csv(os.path.join(RES, "lm_layer_thresholds.csv")); ok = l[l["censor_rtn_0.05"] != "none"]; y = (ok["b_rtn_0.05"] > 4).astype(int)
rows.append(dict(level="matrix (weights)", bits=4, event="needs > 4 weight bits", rate=y.mean(), auc_predictor=roc_auc_score(y, ok.kappa), predictor="weight outlier ratio", auc_baseline=roc_auc_score(y, ok.F_hat), baseline="superposition proxy", n=len(y)))
out = pd.DataFrame(rows); out.to_csv(os.path.join(RES, "forecast_auc.csv"), index=False); print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
