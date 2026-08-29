# DEVIATIONS.md — changes to the preregistered analysis, dated, with the evidence that forced them

## D1 (2026-08-29 12:20, after E0 fp32-only replication, before any quantized result was inspected)

**Evidence (E0, fp32 only, m=30, n=400, uniform importance):** the hard count `N_ret = #{ρ_i ≥ 0.5}` is
degenerate under uniform importance. The model does not drop features in this regime; it represents *all* live
features with nearly identical ρ (p=0.02: 365 live columns, ρ median 0.42, max 0.49 → N_ret = 0;
p=0.01: 400 live columns, ρ median 0.56 → N_ret = 399). Where the ρ distribution does straddle 0.5
(p=0.05–0.1) it is because the marginal represented feature equilibrates at ρ ≈ 0.5, so N_ret is extremely
seed-noisy (p=0.1: 74 vs 109 across two seeds with identical live-column counts 138 vs 135).

**Change:**
1. Primary retention curve becomes the **soft count** `R(b) = Σ_i max(ρ_i(b),0) / Σ_i max(ρ_i(∞),0)`
   (same seed, same x stream). It reduces to `N_ret(b)/N_ret(∞)` under prefix truncation (ρ ∈ {0,1}) and is
   continuous under uniform degradation. `b_crit` = interpolated crossing of R = 0.95 (0.90 / 0.99 sensitivity).
   The preregistered hard-count `b_crit` is still computed and reported wherever it is defined
   (N_ret(∞) ≥ 0.5·n_live), with thresholds 0.25 / 0.5 / 0.75.
2. `F = k_fp32 / m` with `k_fp32` = number of **live columns** of the fp32 arm, `#{i : ‖W_i‖ ≥ 0.5}` (the
   quantity the §2.2 derivation actually uses: "only live columns interfere; live-column norms ≈ 1"). Column
   norms are bimodal (E0: live ≈ 0.97–1.26, dead ≪ 0.5), so the threshold is not delicate; `Σ_i‖W_i‖²/max‖W_i‖²`
   is reported alongside as the soft version.

**What does not change:** the slope prediction (½ per doubling of F) is criterion-independent — the derivation
sets an added-interference variance equal to a fixed fraction of the explainable variance, and any fixed
fraction shifts only the intercept. Decision bands in PREDICTIONS.md apply unchanged to the soft-count slope.
