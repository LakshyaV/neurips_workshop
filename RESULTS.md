# RESULTS.md — what happened (including what did not)

Preregistration: `PREDICTIONS.md` (commit 6a2afc9, 11:46). Dated analysis changes: `DEVIATIONS.md` (D1–D3).
All numbers below are from `results/*.parquet` via `src/analysis*.py`; figures in `figs/`.

## Headline

**The preregistered law is rejected.** The critical bit-width does not grow with the superposition ratio F.
Under round-to-nearest quantization the bit-width at which the represented features lose 5 % of their explained
variance is ≈ 3.8–4.0 bits for every configuration tested, across F = 4 → 56 (m = 10 and 16, p = 0.002 → 0.05),
both under quantization-aware training (E1) and post-training quantization of the same models (E3).
What sets the threshold is (a) a tolerance to *relative weight perturbation* that is independent of F and shared within
~2× by every noise-like damage family (σ_rel,crit ≈ 0.10–0.23 in the toy; it is **not** a universal constant: in GPT-2 it
varies 7× across matrices and grows as κ^0.46), and (b) the outlier ratio κ, entering as log2 κ with a slope of 0.6–0.7 —
for which the quantizer-free noise arm's σ* ∝ κ^0.46 predicts 0.54 [0.35, 0.71] against the independently measured
0.63 [0.49, 0.75] — consistent within error, prediction 0.1 low (see §5A). The per-feature
margin law (E5, §5D, §5E) is **exploratory** — found after the preregistered analyses — and is labelled as such.
Scoring of every preregistered prediction is at the end.

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


## E3 — full (11 of 12 E1 groups, 44 cells, F = 4 → 56, both m)
σ_crit at R_live = 0.95, slope of log2 σ_crit vs log2 F (preregistered −½ for noise-like families):
RTN **+0.05** [0.05, 0.06] · per-channel +0.04 · dithered +0.02 · stochastic +0.03 · Gaussian **+0.00** [−0.00, 0.01] ·
dropout +0.24 · rank truncation: no finite threshold (removing one of m directions costs > 5 % everywhere).
On the bits axis, post-training: RTN b_crit slope +0.03 [0.02, 0.04] (mean 3.87, m = 10: 3.74–4.03; m = 16: 3.1–3.4 …
per cell in `results/E3_bcrit_bits.csv`), per-channel −0.03, dithered −0.04, stochastic +0.07.
σ_crit is ~1.2–1.3× larger at m = 16 than at m = 10 for every noise-like family (√(16/10) = 1.26): the dominant damage
is the perturbation of each feature's own direction, whose relative size scales as σ_rel/√m — not cross-talk, which is
what both the preregistered derivation and D2 assumed. `figs/fig3_anydamage_null.png`.

## §5A′ — GPT-2, per-channel RTN
With per-channel (per output column) scaling the pathology disappears and every matrix crosses ΔNLL = 0.05 at nearly the
same width: b_crit,pc = 2.70 ± 0.33 bits (n = 43; range 2.0–3.4), uncorrelated with κ_eff (slope −0.01, R² = 0.00),
per-tensor κ (R² = 0.00) or F̂ (R² = 0.00). Per-channel scaling equalises the layers. `figs/fig7b_lm_perchannel.png`.

## §5A″ — Validating the superposition proxy with per-layer SAE feature counts (added after the third review)
`src/lm_sae_layers.py`, `results/lm_sae_layers.csv`. For every block L, the public residual SAE (jbloom, blocks.L.hook_resid_pre)
is run on 49k tokens; F_SAE(L) = live features (≥ 20 firings) / 768. SAE variance explained 0.97 (L0) → 0.75 (L11).

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F_SAE | 5.8 | 3.3 | 4.3 | 6.9 | 10.7 | 14.7 | 17.9 | 20.1 | 21.9 | 23.3 | 20.9 | 22.4 |

- The preregistered proxy F̂ = d_in/PR **anticorrelates** with F_SAE (pooled r = −0.31; mlp.c_fc inputs r = −0.79): it was not
  measuring superposition. The earlier "F̂ predicts nothing" null is therefore replaced, not just weakened.
- Against the real count (matrices reading each block's residual: c_attn + c_fc, 24 uncensored): log2 σ_crit vs log2 F_SAE
  slope **+0.21 [0.03, 0.49]** (block-bootstrap CI), R² 0.16; joint fit log2 σ = −3.45 + 0.24·log2 κ + 0.24·log2 F_SAE.
- **Depth confound (D5):** F_SAE is monotone in depth; depth alone explains R² 0.32 of σ_crit; joint fit gives −0.33·log2 F_SAE +
  0.168·block; within blocks 0–5 the slope is −0.18, within 4–11 it is +0.91. With 12 blocks the F-dependence is **not identifiable
  separately from depth**: no negative dependence is observed and the sign is undetermined. Reported as inconclusive.

## §5C — mixed precision on GPT-2 (held-out 49k tokens, all 48 matrices quantized jointly, per-channel RTN)
ΔNLL (nats) at equal parameter-weighted average bits; κ-rule = b_i = c + log2 κ_eff,i; law = per-matrix
1 + log2(1 + κ_eff/(√12 σ_crit,i)) shifted to the budget; oracle = greedy on measured single-matrix sensitivities;
random = the κ-rule's multiset of widths permuted. `results/lm_mixed.csv`, `figs/fig9_mixed_precision.png`.

| avg bits | uniform | κ-rule | law (noise probe) | sensitivity-greedy (single-matrix ΔNLL, additivity assumed) | random ×3 |
|---|---|---|---|---|---|
| 3.5 | — | 3.56 | **1.86** | 2.08 | 3.6–4.1 |
| 4.0 | **0.449** | 0.659 | 0.600 | 0.459 | 0.68–0.88 |
| 4.5 | — | 0.274 | **0.190** | 0.183 | 0.26–0.33 |
| 5.0 | **0.084** | 0.094 | 0.103 | 0.067 | 0.10–0.11 |

Reading: once per-channel scaling has equalised the matrices (2.7 ± 0.3 bits), there is almost nothing left for mixed
precision to exploit — uniform per-channel RTN is within 0.01–0.02 nats of the sensitivity-greedy allocation at 4 and 5 bits (it is not an oracle: it assumes single-matrix damages add, and loses to the noise-probe law at 3.5 bits and to uniform at 4.0),
and the calibration-free κ-rule does not beat uniform. Between integer budgets the noise-probe ("law") allocation
matches the oracle (4.5 bits) or beats it (3.5 bits: the oracle's greedy step is coarse), and both beat random by 2×.
Per-tensor mixed precision is not viable at these budgets at all: joint per-tensor damage is super-additive (all 48
matrices at 6 bits: +4.6 nats vs 1.6 summed; at 8 bits still +0.14). The practical recommendation that follows from the
law is therefore simple: **per-channel scaling removes the κ tax entirely; after it, per-layer bit allocation is
second-order for GPT-2 small** (a statement about this model and quantizer, not about mixed precision in general).

## E1 — main sweep (QAT)
1536 models (m ∈ {10, 16}, n/m ∈ {4 … 128}, p ∈ {0.002, 0.005, 0.02, 0.05}, b ∈ {2 … 8, ∞}, 4 seeds, 6000 steps, uniform
importance). 40 of 48 (m, n, p) cells have an interpolable b_crit in all 4 seeds (the rest are censored at b ≤ 3 or have
N_ret(∞) too small); F spans 4.0 → 56.5, κ (fp32 arm) 2.5 → 13.5, κ_live 2.4 → 3.1. `figs/fig1_bcrit_vs_F_R_live_95.png`,
`figs/fig2_Nret_vs_bits.png`, `results/E1_bcrit_*.csv`.

**P1 (slope of b_crit vs log2 F; preregistered ½, "dead" below 0.15):**

| retention curve / threshold | slope | bootstrap 95 % CI (seeds) | R² | configs |
|---|---|---|---|---|
| R_live ≥ 0.95 (primary) | **−0.11** | [−0.12, −0.09] | 0.04 | 40 |
| R_live ≥ 0.90 | −0.15 | [−0.16, −0.13] | 0.18 | 27 |
| R_live ≥ 0.99 | +0.04 | [−0.01, +0.08] | 0.00 | 39 |
| R_soft (all features) ≥ 0.95 | −0.02 | [−0.04, −0.01] | 0.01 | 38 |
| preregistered hard count N_ret(ρ≥0.5), where defined | −0.27 | [−0.37, −0.15] | 0.07 | 19 |
| hard count, ρ ≥ 0.25 / 0.75 | +0.32 / +0.05 | [0.10, 0.66] / [−0.64, 0.70] | 0.11 / 0.00 | 9 / 6 |
| R_live ≥ 0.95 on the measured-residual axis b_eff | −0.27 | [−0.32, −0.17] | | 16 |

Every definition lands in the "hypothesis dead" band or below; no CI comes near ½. The zero-parameter prediction
1 + log2(1 + κ√(F/3)) vs measurement has slope 0.46, R² = 0.33 — its residual predictive power comes entirely from κ.

**What does vary — and it is κ (P2):** b_crit ranges 3.1 → 5.2 bits across cells, and

| regressor (40 cells) | slope | intercept | R² |
|---|---|---|---|
| log2 F | −0.11 | 4.47 | 0.04 |
| **log2 κ (fp32 arm)** | **0.72** | 2.57 | **0.82** |
| log2 ℓ (live fraction k/n) | −0.35 | 3.62 | 0.83 |
| log2 κ_live | −1.58 | | 0.06 |
| joint | b = 2.77 + 0.71·log2 κ − 0.05·log2 F | | |

b_crit − log2 κ = 1.99 ± 0.31 over all 40 cells (middle panel of fig. 1). In E1, κ and the live fraction are collinear
(κ = κ_live/√ℓ with κ_live ≈ 2.7 universal), so E6 is needed to separate them; it does (below).

**QAT vs PTQ (36 shared cells):** mean b_crit 4.02 (QAT) vs 3.87 (PTQ RTN); the QAT − PTQ difference correlates
0.89 with log2 κ — training through a per-tensor STE grid is *worse* than post-hoc rounding precisely when the tensor
has large outliers / few live columns, and better only for dense tensors. Quantization-aware training does not remove
the κ tax; per-channel scaling does (E3, GPT-2).

## E4 — translation vs tilt
`figs/fig4_translation_vs_tilt.png`. Under uniform importance the ρ profile of the represented features is nearly flat
(all live features share ρ within ±0.1), so quantization lowers the whole profile together (a tilt of the plateau, not a
prefix truncation): the three best features of the fp32 model lose ρ at the same bit-widths as the median feature
(top-3 ρ drops < 0.05 for b ≥ 5, 0.1–0.3 at b = 3, in every cell). "Capacity" (some features kept intact, others dropped)
is therefore the wrong frame for uniform importance; the right frame is uniform degradation of all represented
directions by the same relative perturbation. The importance-ordered case is E5.

## E2 — lattice controls under QAT
QAT with the alternative quantizers (m = 16, n/m ∈ {8, 32, 128}, p ∈ {0.005, 0.02}, b ∈ {2 … 8, ∞}, 2 seeds, 348
models). Paired difference in b_crit (R_live 95 %) against the shared-lattice uniform quantizer of the same (config, seed):

| quantizer | Δb_crit vs uniform (mean ± SEM, n = 12) | slope vs log2 F |
|---|---|---|
| dithered lattice (per-element offset) | **+0.27 ± 0.13** (worse) | −0.11 [−0.25, 0.04] |
| stochastic rounding | **+0.63 ± 0.11** (worse) | +0.00 [−0.23, 0.25] |
| per-channel scale | **−0.54 ± 0.09** (better) | −0.30 [−0.49, −0.09] |
| uniform (reference) | 0 | −0.25 [−0.41, −0.09] |

Same ordering and magnitudes as post-training (E3: dither 0.2–1.5 bits worse, per-channel 0.4–0.5 better). Models do
exploit the shared lattice — but what they exploit is its exact zero (dead columns stay dead), not fine alignment;
stochastic rounding, which never lets a weight settle, is the worst of all. `figs/fig5_lattice.png`.

## E5 — importance ordering (power-law importance)
Power-law importance I_i = (i+1)^−α, α ∈ {0.5, 1, 2}, m = 16, n ∈ {128, 512}, p ∈ {0.005, 0.02}, all bit-widths, 4 seeds
(576 models). Here the model does drop features, from the tail: the hard count is well-behaved (e.g. α = 2, n = 128,
p = 0.005: N_ret = 36.5 / 42 / 44 / 46 / 46 / 46 at b = 3 … 8 vs 46.25 at fp32). `figs/fig6_ordering.png`,
`figs/fig6b_per_feature_bits.png`, `results/E5_model_summary.csv`, `results/E5_per_feature_bits.csv`.

**P3 — ordering is preserved (confirmed).** Prefix fraction (share of the retained set that is the top-|R| by importance):
0.955–0.979 at fp32 and 0.947–0.991 at every b ≥ 3; the largest drop is 0.032 (α = 2, n = 512, p = 0.02, b = 3), against
the preregistered 0.10 bar. Quantization removes features from the tail of the importance ranking, exactly as fp32
capacity does.

**Per-feature critical bits.** For each feature live at fp32, b_i = the smallest b ≥ 3 with ρ_i ≥ 0.5 (9 = never by 8 bits):

| α | n | p | live | b_i top-10 | b_i ranks 20–40 % | b_i tail (> 60 %) | corr(b_i, rank) | corr(b_i, ρ_i^fp32) |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 128 | 0.02 | 102 | 3.00 | 3.03 | 3.93 | 0.51 | −0.56 |
| 0.5 | 512 | 0.005 | 196 | 3.00 | 4.18 | — | 0.65 | −0.70 |
| 1 | 128 | 0.005 | 121 | 3.00 | 3.00 | 4.16 | 0.54 | −0.71 |
| 1 | 128 | 0.02 | 69 | 3.00 | 3.00 | 8.00 | 0.46 | −0.57 |
| 1 | 512 | 0.005 | 110 | 3.00 | 5.27 | — | 0.49 | −0.67 |
| 2 | 128 | 0.005 | 56 | 3.00 | 4.84 | — | 0.75 | −0.93 |
| 2 | 512 | 0.005 | 48 | 3.00 | — | — | 0.73 | −0.96 |

The most important features need 3 bits everywhere; the marginal ones need 4–8+. Pooled over all 4,290 live features,
**b_i = c − 0.52·log2(ρ_i^fp32 − 0.5), R² = 0.68**: a feature's critical bit-width is set by its fp32 margin above the
retention threshold, half a bit per halving of the margin. This is the "½ bit per doubling" exponent of the preregistered
law — but the doubling that matters is a feature's margin, not the layer's superposition ratio. It is the mechanism behind
the tail-loss phenomenon of Hooker et al.: rare/unimportant features sit closest to the threshold and are the first to
be pushed under by any perturbation.

b_crit (95 %) across the 12 (n, p, α) cells has slope −0.77 [−0.81, −0.70] (soft) / −0.85 (hard) in log2 F — negative,
again because steeper importance leaves fewer live columns and a larger κ.

## E6 — κ manipulation at fixed F
Weight clipping at c·rms(W), c ∈ {2, 3, 4, ∞}, at fixed (m = 16, n ∈ {256, 1024}, p ∈ {0.005, 0.02}), 4 seeds each,
QAT. κ of the fp32 arm moves from 7.1 → 2.0 (n = 1024, p = 0.02) while F changes by < 2× (in the opposite direction).
`figs/fig8_kappa.png`, `results/E6_bcrit_by_kclip.csv`.

| cell | κ = 2.0 | 3.0 | 4.0 | unclipped (κ) |
|---|---|---|---|---|
| n=1024, p=0.02 | 3.65 ± 0.13 | 3.94 ± 0.34 | 4.31 ± 0.13 | 4.69 ± 0.23 (κ = 7.1) |
| n=1024, p=0.005 | 3.21 ± 0.14 | 3.89 ± 0.05 | 3.97 ± 0.05 | 3.97 ± 0.09 (κ = 4.2) |
| n=256, p=0.02 | 3.48 ± 0.25 | 4.09 ± 0.20 | 4.11 ± 0.16 | 4.16 ± 0.21 (κ = 3.5) |
| n=256, p=0.005 | censored | 3.32 ± 0.06 | 3.41 ± 0.11 | 3.42 ± 0.09 (κ = 2.8) |

Pooled within-cell slope of b_crit on log2 κ (cell fixed effects, 15 points): **0.65, R² = 0.92**; per-cell slopes
0.58 / 0.71 / 0.83 where κ spans ≥ 0.8 octave. P2's preregistered band was [0.7, 1.3]: the effect is real and causal, its
size is ~⅔ of the naive prediction — the same sub-unit slope as GPT-2 (0.63) and E1 (0.72), because a tensor with fewer
outliers is also less tolerant to relative perturbation (σ_crit ∝ κ^0.46 in GPT-2).

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

**Non-circular test of the sub-unit κ slope.** σ_crit is measured from the Gaussian-noise arm (no quantizer); across
matrices σ_crit ∝ κ^e with e = 0.46 [0.29, 0.65] (ΔNLL 0.05; 0.57 [0.32, 0.89] at 0.02). Substituting into the law predicts
an RTN slope of 1 − e = 0.54 [0.35, 0.71] (0.43 [0.11, 0.68] at 0.02); the slope measured independently from the RTN arm is
0.63 [0.49, 0.75] (0.59 [0.35, 0.80]); paired difference measured − predicted = +0.09 [−0.05, +0.26] (+0.16 [−0.02, +0.37]).
Consistent within error; the prediction's central value is 0.1–0.16 low, so the σ*(κ) account explains part, not all, of
the shortfall from slope 1.

Reading: the per-matrix damage threshold carries no information about the superposition proxy (R² ≤ 0.04 at every
threshold), while κ predicts the *independently measured* RTN bit-width with R² = 0.53 and the noise-equivalent
conversion from the toy analysis reproduces each matrix's RTN cliff to a mean error of −0.23 bits. The κ slope is
0.6–0.7 rather than 1 because outlier-heavy matrices also tolerate more relative noise (σ_crit ∝ κ^0.46): the
outliers inflate the quantization step, but a matrix whose energy is concentrated in a few large entries is less
perturbed, per unit of relative noise, in the directions that matter. The most fragile matrices are the blocks-1–3
mlp.c_proj (κ = 158–186, > 0.05 nats even at 6 bits) and block-0 attn.c_proj (σ_crit = 0.11); attention output
projections are the most noise-tolerant (σ_crit ≈ 0.7) yet not the easiest to quantize — which is exactly the
κ-vs-σ_crit distinction the corrected law makes.

## §5D — GPT-2: which tokens break first, and at how many bits (the per-feature law in a real LM) — *exploratory, not preregistered*
All 48 matrices quantized jointly (per-channel RTN, b ∈ {2, 3, 4, 5, 6, 8}) or perturbed jointly with Gaussian noise
(σ_rel ∈ {0.03 … 0.4}), on 49,104 held-out WikiText-2 tokens (offset 147,456; fp32 NLL 3.425, top-1 accuracy 0.376).
For every token the fp32 model predicts correctly (n = 18,464) we record its fp32 logit margin (correct logit minus the
runner-up) and whether its prediction survives each condition. `src/lm_tokens.py`, `src/analysis_tokens.py`,
`figs/fig10_token_margins.png`, `results/lm_tokens_margin_bins.csv`.

| fp32 margin (log2 bins, logits) | n | mean critical bits | P(flip) 4-bit | 5-bit | 6-bit | 8-bit |
|---|---|---|---|---|---|---|
| < 1/8 | 372 | 6.30 (7 % never) | 0.58 | 0.52 | 0.36 | 0.07 |
| 1/8–1/4 | 752 | 5.86 | 0.58 | 0.42 | 0.27 | 0.02 |
| 1/4–1/2 | 1516 | 5.11 | 0.49 | 0.30 | 0.10 | 0.00 |
| 1/2–1 | 2869 | 4.41 | 0.35 | 0.11 | 0.02 | 0.00 |
| 1–2 | 4383 | 4.00 | 0.20 | 0.02 | 0.00 | 0.00 |
| 2–4 | 4655 | 3.89 | 0.09 | 0.00 | 0.00 | 0.00 |
| 4–8 | 2823 | 3.92 | 0.03 | 0.00 | 0.00 | 0.00 |
| > 8 | 743 | 3.60 | 0.005 | 0.00 | 0.00 | 0.00 |

- Binned mean critical bit-width vs log2 margin: **slope −0.39, R² = 0.90** (per token: −0.34 [−0.35, −0.32], R² 0.23 —
  individual tokens are noisy, the conditional mean is a law).
- Margin, not rarity: b_crit = 5.30 − 0.36·log2 margin − 0.09·log2 frequency. The unigram-frequency coefficient is a
  quarter of the margin's; "quantization deletes rare tokens" is mostly "quantization deletes low-margin predictions",
  and rare tokens happen to have low margins. This is the matched-damage control the plan asked for in §5B.
- Universality: Gaussian weight noise produces the same margin ordering (right panel); at 2–3 bits per-channel RTN
  every token flips regardless of margin (joint collapse, ΔNLL > 4 nats) — the regime where the additivity of
  single-matrix damage breaks down.
- Same exponent family as the toy per-feature law (b_i = c − 0.52·log2 margin_i in variance units): the tail of the
  margin distribution is what a given precision deletes, half a bit or so per doubling of the margin.

## §5E — GPT-2: which *internal features* are erased (sparse-autoencoder features) — *exploratory, not preregistered*
SAE: `jbloom/GPT2-Small-SAEs-Reformatted`, `blocks.6.hook_resid_pre` (24,576 features on the 768-d residual stream entering
block 6; sae_lens convention verified by reconstruction: 84 % variance explained, L0 ≈ 58 on our text, vs. < 0 for the
wrong input conventions). 196,596 held-out WikiText-2 tokens in 128-token BOS-prefixed contexts (the SAE's training
context). Blocks 0–5 (24 matrices) are damaged jointly — per-channel RTN at b ∈ {2, 3, 4, 5, 6, 8} or Gaussian noise at
σ_rel ∈ {0.05, 0.1, 0.15, 0.2} — and every feature's activation pattern is compared with its fp32 pattern on the same tokens:
ρ_f(b) = 1 − Σ_t (a_f^q − a_f^fp32)² / Σ_t (a_f^fp32)², retained if ρ_f ≥ 0.5. 22,499 features fire on ≥ 20 tokens (live).
`src/lm_sae.py`, `src/analysis_sae.py`, `figs/fig11_sae_features.png`, `results/lm_sae_features.csv`.

| condition | 2-bit | 3-bit | 4-bit | 5-bit | 6-bit | 8-bit | σ 0.05 | 0.1 | 0.15 | 0.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| live features retained (ρ_f ≥ 0.5) | 0.000 | 0.261 | 0.844 | 0.986 | 1.000 | 1.000 | 0.995 | 0.901 | 0.618 | 0.293 |
| SAE variance explained on the damaged residual | < 0 | < 0 | 0.740 | 0.816 | 0.830 | 0.839 | 0.826 | 0.771 | 0.662 | 0.105 |

**Which features go first: the weak ones.** Feature strength = rms contribution to the residual stream at fp32
(activation × decoder norm, including zeros), in log2 bins:

| log2 strength | n | mean critical bits | retained at 3-bit | 4-bit | 5-bit | 6-bit |
|---|---|---|---|---|---|---|
| −8 … −7 | 19 | 6.05 | 0.00 | 0.00 | 0.21 | 0.89 |
| −7 … −6 | 171 | 5.49 | 0.00 | 0.04 | 0.56 | 0.96 |
| −6 … −5 | 968 | 4.75 | 0.01 | 0.34 | 0.91 | 1.00 |
| −5 … −4 | 2686 | 4.34 | 0.03 | 0.65 | 0.98 | 1.00 |
| −4 … −3 | 6775 | 4.00 | 0.17 | 0.84 | 0.99 | 1.00 |
| −3 … −2 | 8010 | 3.76 | 0.32 | 0.92 | 1.00 | 1.00 |
| −2 … −1 | 3259 | 3.51 | 0.52 | 0.98 | 1.00 | 1.00 |
| −1 … 0 | 572 | 3.34 | 0.67 | 1.00 | 1.00 | 1.00 |
| > 0 | 38 | 3.2 | 0.76 | 1.00 | 1.00 | 1.00 |

- Per feature (n = 22,499): b_f ~ log2 typical magnitude: **slope −0.37 [−0.38, −0.37], R² = 0.45**; vs log2 strength −0.31 (R² 0.26);
  vs log2 frequency alone +0.13 (R² 0.14). Binned means: **slope −0.34, R² = 0.94**.
- Magnitude, not rarity: b_f = 4.61 − 0.355·log2 magnitude + 0.019·log2 frequency.
- Each extra bit shifts the loss curve by ≈ 1.5–2 log2-units of strength (left panel); Gaussian weight noise produces the
  same family of curves (right panel) — universality holds inside the model.
- Exponent family: toy per-feature −0.52 (variance-unit margin), GPT-2 tokens −0.39 (logit margin), GPT-2 SAE features
  −0.34/−0.37 (residual magnitude): in every case a feature's critical bit-width falls by roughly a third to a half of a bit
  per doubling of its margin/strength.

## §5F — GPT-2: activation quantization (the outlier law on the quantity the folklore is about)
For each of the 48 matrices alone, the *input activations* are fake-quantized with per-tensor symmetric RTN (scale =
absmax over the 49k calibration tokens) at b ∈ {3, 4, 5, 6, 8}; ΔNLL on the same tokens; κ_act = max|x|/rms(x) over the
calibration tokens (range 10–382; mlp.c_proj inputs are the most outlier-heavy). `src/lm_act.py`, `src/analysis_act.py`,
`figs/fig12_activation_kappa.png`, `results/lm_act_thresholds.csv`.

| regressor (42 interpolable matrices) | slope | 95 % CI | R² |
|---|---|---|---|
| **log2 κ_act** | **1.12** | [0.91, 1.39] | **0.73** |
| log2 κ_w (weight outlier ratio of the same matrix) | 0.35 | [−0.01, 0.58] | 0.12 |
| joint: b_act = −0.31 + 1.25·log2 κ_act − 0.18·log2 κ_w | | | |

Mean activation b_crit by matrix type: attn.c_proj 3.4 (κ_act 17), attn.c_attn 4.5 (21), mlp.c_fc 5.5 (42), mlp.c_proj 6.2
(89). Activation outliers set activation bits with the naive slope of 1 (the CI includes 1), and the weight outliers of the
same matrix carry no information about it — the two κ's are separate quantities governing separate quantizers.
(Added after the external review; post-training, per-tensor, one matrix at a time.)

## §5G — Forecasting quantization damage from the fp32 model (ROC AUC)
`src/analysis_forecast.py`, `results/forecast_auc.csv`. Event = what joint per-channel RTN breaks; predictor = an fp32 quantity.

| level | bits | event (rate) | predictor → AUC | baseline → AUC |
|---|---|---|---|---|
| token (n = 18,464 fp32-correct) | 4 / 5 / 6 | prediction flips (0.22 / 0.085 / 0.039) | fp32 logit margin → **0.80 / 0.91 / 0.95** | rarity → 0.60 / 0.56 / 0.57 |
| SAE feature (n = 22,499) | 3 / 4 / 5 | feature erased (0.74 / 0.16 / 0.014) | fp32 magnitude → **0.87 / 0.91 / 0.94** | frequency → 0.23 / 0.36 / 0.51 |
| toy feature (n = 4,290) | 4 | needs > 4 bits (0.086) | fp32 margin → **0.92** | — |
| matrix, activations (n = 48) | 5 | needs > 5 activation bits (0.52) | κ_act → **0.97** | weight κ → 0.64 |
| matrix, weights (n = 48) | 4 | needs > 4 weight bits (0.83) | κ → 0.67 | F̂ → 0.62 |

Reading: which tokens, features and activations a quantized model will lose can be forecast from the fp32 model alone
with AUC 0.8–0.97; rarity/frequency is a poor predictor. The weight-level forecast is weak (small n, 83 % base rate).

## Scoring PREDICTIONS.md
| prediction | preregistered decision rule | outcome |
|---|---|---|
| P1: b_crit vs log2 F slope ½ | [0.40, 0.60] confirmed; [0.15, 0.35] other exponent; < 0.15 dead | **dead**: −0.11 [−0.12, −0.09] (QAT); +0.03 [0.02, 0.04] (PTQ); all thresholds/definitions < 0.15 |
| P1 table of predicted b_crit (2.85 → 6.41 for F = 1 → 256) | | measured 3.1–5.2 with no F trend; predicted-vs-measured slope 0.46, R² 0.33 (κ carries it) |
| P2: κ enters as log2 κ with slope 1 | slope ∈ [0.7, 1.3] confirmed; ≈ 0 no; between partial | **partial, strongly supported**: E6 causal slope 0.65 (R² 0.92); E1 0.72 (R² 0.82); GPT-2 0.63 (R² 0.53) |
| P3: ordering preserved under quantization (power-law importance) | drop in prefix fraction < 0.10 at b ≥ 3 | **confirmed**: prefix-fraction drop ≤ 0.03 at every b ≥ 3 (bar: 0.10); per-feature b_i = c − 0.52·log2(margin_i), R² 0.68 |
| E4: top-3 ρ unchanged (translation) | drop < 0.05 for b ≥ b_crit | **tilt, not translation** under uniform importance: the whole plateau moves |
| E3: noise-like damage slope −½, structured ≈ 0 | | **all noise-like families ≈ 0** (+0.00 … +0.05); dropout +0.24; rank: catastrophic |
| E2: Q vs dithered Q_u | no direction preregistered | shared lattice **better** (PTQ: dither costs 0.2–1.5 bits; QAT: dither +0.27 ± 0.13 bits, stochastic +0.63 ± 0.11, per-channel −0.54 ± 0.09, paired n = 12) — because the lattice contains 0 |
| §5A: log2 σ_crit vs log2 F̂ slope ∈ [−0.65, −0.35] | | **not**: +0.14 [−0.13, 0.35], R² 0.04 |
| §5A: b_crit − log2 κ vs log2 F̂ slope ≈ ½ | | **not**: −0.17 [−0.37, 0.09] |
| (D2, written before E1/E3) σ_crit flat in F; b_crit = const + log2 κ where κ cannot adapt | | **confirmed** (E3 slopes ≈ 0; E6/E1/GPT-2 κ slopes 0.63–0.72, i.e. 0.54 predicted from σ* ∝ κ^0.46) |
| per-feature margin law (E5 per-feature, §5D tokens, §5E SAE) | *not preregistered* | exploratory positive: slopes −0.52 / −0.39 (binned; −0.34 per token, R² 0.23) / −0.37 (R² 0.45 per feature, 0.94 binned) |

## Things that went wrong and were fixed (audit trail)
- The soft count clipped at 0 has a Monte-Carlo clipping bias when reference and damaged models are evaluated with
  different sample counts (first pilot-E3 run gave R = 1.16 > 1). Fixed by evaluating an undamaged copy inside every
  damage stack (identical samples) and using the unclipped explained variance of the fp32-live features (D3).
- The [K, n, n] Gram matrix for D_i on stacked damage levels (512 models × 2048²) exhausted MPS memory and stalled every
  GPU job; chunked over models.
- Three concurrent MPS jobs stall each other; at most two were run at a time afterwards.
- A git history rewrite (to purge a 128 MB model blob that an early `git add -A` had captured) checked out the
  rewritten tree and deleted the tracked raw parquet files and job logs from the working directory. All derived
  artefacts (per-model summaries, b_crit tables, threshold CSVs, figures) survived; the per-feature E1 rows were
  regenerated from the saved checkpoints (`src/regen_rows.py`) and the four smallest E1 groups were retrained.
- `tms_numpy_pilot.py` was not shipped with the plan; a minimal numpy reference with manual gradients was written and
  the port cross-checked against it.
