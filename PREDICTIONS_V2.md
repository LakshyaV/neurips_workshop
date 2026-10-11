# PREDICTIONS_V2 — Phase 2 preregistration (DRAFT, not yet posted)

**Status: DRAFT for author review. Nothing below is binding until posted to OSF with a
timestamp. No damage run on the 7B/8B model may happen before that timestamp.** The κ-only
spike (weight statistics, no evaluations) runs before finalization and its numbers are appended
under "Known at preregistration time". Decision bands are proposals; the author sets them.

## Scope
- Model: **Llama-3.1-8B (base)**, released bf16 weights ("full-precision statistics" throughout;
  the v1 GPT-2 arms remain fp32). Embeddings and lm_head are never damaged.
- Data: held-out WikiText-2 test blocks, Llama tokenizer, 1024-token sequences, ≥49k tokens for
  ΔNLL; larger token budget for per-token margin analyses (target ~200k).
- Thresholds and rules identical to v1: ΔNLL = 0.05 primary, 0.02 replication; log-log
  interpolation for crossings; censoring labels as in v1. Matrix-bootstrap CIs.
- Harness: `src/lm_eval2.py` (architecture-generic port) + `src/rotate.py` (R1 rotation),
  `src/lm_tokens2.py` (P9a), `src/lm_sae2.py` (P9b), `src/lm_act2.py` (P8).
  The κ-only spike runs via `--stream-stats` (statistics streamed from safetensors shards,
  no model instantiation; validated bit-identical to the model path on GPT-2 and a saved
  random-init Llama). All ports validated bit-exact against v1 on GPT-2 (2026-10-11):
  tokens arm 49,104 tokens × 6 bit levels, zero diffs; SAE arm 24,576 features × 7
  deterministic conditions × 7 accumulators, zero diffs (vs v1 code on an identical subset);
  activation arm 60 (matrix, bits) points, zero diffs. Token stream for 8B arms:
  `src/make_ids.py` re-encodes v1's exact text (lossless BPE round trip asserted) with the
  Llama tokenizer → 288,936 tokens.
  Port validation on GPT-2 (known results, 2026-10-11): κ identical to 7e-15; RTN b_crit
  max |new−ref| = 0.0000 bits (11 uncensored matrices, blocks 0/5/11); gaussian arm bit-exact
  (worst |Δdnll| = 0.0000 over 36 (matrix, σ) points) once the original generator stream is
  replicated. Llama mechanics: prefix-cache == full forward (random-init check). Rotation
  exactness: max |Δlogits| = 1.5e-6 (fold + R1, random-init).
- **Gaussian draw sensitivity (methods note):** the mean ΔNLL of a single z draw is
  heavy-tailed across draws (e.g. 0.018 vs 0.046 at the same σ, each ± sem 0.0025, block-0
  c_attn), so single-draw σ_crit carries 10–25% draw noise on some matrices. The 8B gaussian
  arm therefore averages **2 independent draws per (matrix, σ)** (~+2–4 GPU-h, within budget);
  v1's GPT-2 numbers are unaffected as published (their cross-matrix bootstrap absorbs this).
- Compute cap: **$150 / ~40 GPU-hours**; stop and report before exceeding.
- Order of execution: **P6 (rotation) first**, then P4/P5, P7, P8, P9.
- Every later analysis change: dated entry in DEVIATIONS.md (continuing D6, D7, …), written
  before the affected results are inspected.

## Known at preregistration time
- GPT-2 v1 results (published AXIOM submission; tag `axiom-submitted`).
- Per-channel investigation on GPT-2 (v5 Phase 1, exploratory, 9 statistics tried): worst-group
  κ_g,max (g=128) recovers slope 0.58 [0.36, 0.72], R²=0.37, where bulk statistics are null and
  range-restricted. P7 is its confirmatory restatement on a new model and quantizer.
- κ-only spike on Llama-3.1-8B (weight statistics only, appended before posting):
  [SPIKE TABLE HERE: per-matrix κ, κ_eff, κ_g,max distributions, log2 ranges].

## Predictions and decision bands (proposed)

**P4 — per-tensor weight law transfers.** Slope of per-tensor RTN b_crit on log2 κ across
matrices (ΔNLL=0.05, uncensored): **confirm [0.4, 0.9]; dead < 0.2**; between = partial.
(v1: 0.63 [0.49, 0.76] on GPT-2.)

**P5 — compensation transfers.** Gaussian-arm exponent e = d log σ_crit / d log κ in
**[0.2, 0.7]**; the predicted RTN slope (1−e) agrees with the measured P4 slope within the
paired bootstrap CI (as in v1: predicted 0.54 vs measured 0.63, difference CI covering 0).

**P6 — rotation discriminator (runs first; the paper's sharpest new claim).** Apply
norm-folding + R1 randomized-Hadamard rotation; measure per-matrix κ before/after; re-run
per-tensor RTN cliffs on the rotated model. Per matrix, bits saved Δb = b_crit(unrotated) −
b_crit(rotated). Two preregistered predictors: naive = log2(κ_before/κ_after); compensated =
(1−e′)·log2(κ_before/κ_after), with e′ the P5 exponent measured on **unrotated** weights.
Decision rule on the mean paired difference (Δb − predictor), 95% bootstrap CI:
- CI contains 0 for compensated and excludes 0 for naive → **compensated confirmed**;
- the reverse → **naive confirmed** (compensation refuted);
- contains both or excludes both → **inconclusive / both refuted**; report all numbers either way.

**P7 — group-wise regime: the tail statistic is the predictor.** At g=128 group-wise RTN,
b_crit follows log2 κ_g,max with slope in **[0.3, 0.9]**, and κ_g,max beats the bulk statistics
(κ_eff, mean/median κ_c) by ΔR² ≥ 0.1. **Power gate (set from the spike before posting):** if
the across-matrix log2 range of κ_g,max is < 1.0 doubling, the slope test is declared
underpowered in advance and reported descriptively, not scored.

**P8 — activation law transfers.** Activation b_crit (per-tensor RTN on each matrix's inputs)
follows log2 κ_act with slope in **[0.8, 1.4]** (naive prediction 1; v1: 1.12 [0.91, 1.39]).

**P9 — full-precision forecasting transfers.** (a) Logit margins predict 4-bit g=128 token
flips with AUC ≥ **0.75**; (b) Llama Scope SAE feature magnitudes predict erasure at 4 bits
with AUC ≥ **0.8**. SAE chosen and named before any damage run: **L15R-8x**
(`OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x`, folder `Llama3_1-8B-Base-L15R-8x`, hook
`blocks.15.hook_resid_post`, d_model 4096 → 32,768 features, JumpReLU, dataset-wise norm) —
the post-MLP residual after layer 15, i.e. the stream entering layer 16 of 32, the same
half-depth read point as v1 (resid entering block 6 of 12); matrices in layers 0–15 are
damaged, layers 16+ left intact; (c) κ_act flags matrices needing > 5 activation bits with AUC ≥ **0.85**.

## Not covered (declared exploratory if run)
Joint weight+activation quantization; R2/R3/R4 online rotations; GPTQ-style adaptive rounding;
any second model.
