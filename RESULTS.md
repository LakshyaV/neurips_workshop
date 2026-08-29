# RESULTS.md — what happened (including what did not)

Preregistration: `PREDICTIONS.md` (commit 6a2afc9, 11:46). Dated analysis changes: `DEVIATIONS.md` (D1–D3).
All numbers below are from `results/*.parquet` via `src/analysis*.py`; figures in `figs/`.

## Headline

**The preregistered law is rejected.** The critical bit-width does not grow with the superposition ratio F.
Under round-to-nearest quantization the bit-width at which the represented features lose 5 % of their explained
variance is ≈ 3.8–4.0 bits for every configuration tested, across F = 4 → 56 (m = 10 and 16, p = 0.002 → 0.05),
both under quantization-aware training (E1) and post-training quantization of the same models (E3).
What sets the threshold is (a) a universal tolerance to *relative weight perturbation* — every noise-like damage family
crosses the 5 % line at the same σ_rel — and (b) the outlier ratio of the live weights, κ_live, exactly as the
corrected derivation (DEVIATIONS.md D2) predicts. Scoring of every preregistered prediction is at the end.

## E0 — fp32 replication (gate 1) ✅
n = 400, m = 30, uniform importance, 10 sparsity levels × 2 seeds, 10k steps. D* = m/Σ‖W_i‖² = 1.00 (S=0), 0.60
(S=0.5), 0.50 (S=0.7), 0.37 (S=0.8), 0.21 (S=0.9) …; D_i histograms cluster at ½ (antipodal pairs, S = 0.5–0.7) and
≈ 0.4 (S = 0.8). Σ_i D_i ≤ m holds for every model. Port cross-check vs numpy reference: max |ΔW| = 1e-7 after 300
STE steps (CPU and MPS). `figs/fig0_e0_replication.png`.

**What E0 also showed (→ D1):** under uniform importance the model never *drops* features at high F; it represents all
live features with nearly identical ρ (p = 0.02: 365 live columns, ρ ∈ [0.35, 0.49]; p = 0.01: 400 live, ρ ≈ 0.56).
The preregistered hard count N_ret = #{ρ ≥ 0.5} is therefore a cliff (0 → 399 between adjacent p) and, where it
straddles 0.5, seed-noisy (74 vs 109 at p = 0.1 with 138 vs 135 live columns).

## Gate 2 — sizing pilot, m = 16, n = 2048, 8000 steps, 4 seeds (QAT)
| p | live k | k/n | F = k/m | κ (fp32) | κ_live | b_crit (R_soft 95 %) |
|---|---|---|---|---|---|---|
| 0.002 | 894 | 0.44 | 55.9 | 4.76 | 3.14 | 3.38 ± 0.08 |
| 0.005 | 405 | 0.20 | 25.3 | 6.25 | 2.78 | 3.42 ± 0.02 |
| 0.02 | 138 | 0.07 | 8.6 | 10.3 | 2.66 | 3.54 ± 0.03 |
| 0.05 | 83 | 0.04 | 5.2 | 13.5 | 2.71 | 3.41 ± 0.11 |

κ plateaus (< 1 % change over the last 2000 steps); R(b) monotone for all 16 (p, seed) curves; ΣD_i = m.
Preregistered prediction for these cells: 5.2–5.4 bits with a 1.5-bit spread. Measured: flat at 3.4.

## E3 on the pilot checkpoint (post-hoc damage, paired, R_live)
σ_rel at which R_live = 0.95 (mean over 4 seeds × draws), by family and p:

| family | p=0.002 (F=56) | 0.005 (F=25) | 0.02 (F=8.6) | 0.05 (F=5.2) | slope of log2 σ_crit vs log2 F |
|---|---|---|---|---|---|
| RTN (per-tensor) | 0.168 | 0.157 | 0.127 | 0.127 | +0.13 [+0.13, +0.14] |
| RTN per-channel | 0.161 | 0.134 | 0.126 | 0.131 | +0.09 |
| dithered | 0.194 | 0.198 | 0.215 | 0.204 | −0.03 |
| stochastic rounding | 0.177 | 0.185 | 0.154 | 0.159 | +0.07 |
| Gaussian noise | 0.204 | 0.220 | 0.231 | 0.222 | −0.04 |
| feature dropout | 0.462 | 0.411 | 0.313 | 0.283 | +0.21 |
| rank truncation | < σ_rel(s = m−1) at every F: removing one of 16 directions already costs > 5 % | | | | — |

Preregistered (noise-floor) prediction: −½ for the noise-like families. Observed: all within ±0.13 of zero.
On the bits axis (R_live = 0.95): RTN **3.84 / 3.81 / 3.83 / 3.85** (slope −0.004 [−0.015, +0.009]);
per-channel 3.14–3.44; dithered 4.02 → 5.33 (worse, and κ-dependent: dither turns exact zeros of the dead columns
into noise); stochastic rounding 4.2–4.6. The corrected law, b = 1 + log2(1 + κ_live/(√12·σ_crit)) with the measured
κ_live and σ_crit(RTN), gives 3.68 / 3.61 / 3.82 / 3.85 — within 0.2 bits, no free parameter.
`figs/fig3_pilot_anydamage.png`.

## E3 on the E1 checkpoints, m = 10, n = 40 … 320 (F = 4 → 32)
RTN b_crit (R_live 95 %) = 3.74–4.03 for all 16 (n, p) cells, no trend in F or p; σ_crit(RTN) = 0.11–0.13; Gaussian
0.11–0.19; per-channel 0.10–0.14; SR 0.10–0.14; dropout 0.22–0.46; rank: catastrophic at s = m−1 (σ_rel ≈ 0.31).
Even the below-capacity n/m = 4 cells (all features live, ρ high) cross at 3.8 bits — the numpy pilot's "quantization
does nothing below capacity" was an artefact of the hard count.

## E1 — main sweep (QAT)
%E1%

## E4 — translation vs tilt
%E4%

## E2 — lattice controls under QAT
%E2%

## E5 — importance ordering (power-law importance)
%E5%

## E6 — κ manipulation at fixed F
%E6%

## §5A — GPT-2 small, per-matrix damage thresholds
GPT-2 small (124M), 48 weight matrices (12 blocks × {c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj}), embeddings excluded,
49,152 WikiText-2 test tokens (clean NLL 3.4595), one matrix damaged at a time. Continuous axis: Gaussian noise
W + σ_rel·rms(W)·z on 9 log-spaced σ_rel ∈ [0.03, 0.75]; discrete axis: RTN at 3–6 bits. Threshold ΔNLL = 0.05 nats
(0.02 / 0.10 sensitivity). Per matrix: κ = max|W|/rms(W) ∈ [8.8, 186] and F̂ = d_in/PR ∈ [15, 278].
`figs/fig7_lm_layers.png`, `results/lm_layer_thresholds.csv`.

| test | slope | 95 % CI (bootstrap over matrices) | R² | n |
|---|---|---|---|---|
| log2 σ_crit vs log2 F̂ (preregistered: −½) | **+0.14** | [−0.13, +0.35] | 0.04 | 42 |
| log2 σ_crit vs log2 F̂, ΔNLL = 0.02 / 0.10 | +0.14 / −0.05 | [−0.12, 0.31] / [−0.62, 0.25] | 0.03 / 0.00 | 48 / 30 |
| measured RTN b_crit vs log2 κ (P2: slope 1) | **0.63** | [0.49, 0.76] | 0.53 | 34 |
| measured RTN b_crit vs log2 F̂ | 0.28 | [−0.02, 0.61] | 0.14 | 34 |
| joint: b_crit = 2.40 + 0.72·log2 κ − 0.12·log2 F̂ | | | | 34 |
| b_crit,equiv = 1 + log2(1 + κ/(√12 σ_crit)) vs log2 κ | 0.73 | [0.55, 0.83] | 0.67 | 42 |
| measured RTN b_crit vs b_crit,equiv (noise → bits conversion) | 0.92 | | 0.68 | 34 |
| log2 σ_crit vs log2 κ | 0.46 | [0.30, 0.65] | 0.35 | 34 |

Reading: the per-matrix damage threshold carries no information about the superposition proxy (R² ≤ 0.04 at every
threshold), while κ predicts the *independently measured* RTN bit-width with R² = 0.53 and the noise-equivalent
conversion from the toy analysis reproduces each matrix's RTN cliff to a mean error of −0.23 bits. The κ slope is
0.6–0.7 rather than 1 because outlier-heavy matrices also tolerate more relative noise (σ_crit ∝ κ^0.46): the
outliers inflate the quantization step, but a matrix whose energy is concentrated in a few large entries is less
perturbed, per unit of relative noise, in the directions that matter. The most fragile matrices are the blocks-1–3
mlp.c_proj (κ = 158–186, > 0.05 nats even at 6 bits) and block-0 attn.c_proj (σ_crit = 0.11); attention output
projections are the most noise-tolerant (σ_crit ≈ 0.7) yet not the easiest to quantize — which is exactly the
κ-vs-σ_crit distinction the corrected law makes.

## Scoring PREDICTIONS.md
%SCORE%

## Things that went wrong and were fixed (audit trail)
- The soft count clipped at 0 has a Monte-Carlo clipping bias when reference and damaged models are evaluated with
  different sample counts (first pilot-E3 run gave R = 1.16 > 1). Fixed by evaluating an undamaged copy inside every
  damage stack (identical samples) and using the unclipped explained variance of the fp32-live features (D3).
- The [K, n, n] Gram matrix for D_i on stacked damage levels (512 models × 2048²) exhausted MPS memory and stalled every
  GPU job; chunked over models.
- Three concurrent MPS jobs stall each other; at most two were run at a time afterwards.
- `tms_numpy_pilot.py` was not shipped with the plan; a minimal numpy reference with manual gradients was written and
  the port cross-checked against it.
