# DEVIATIONS.md — changes to the preregistered analysis, dated, with the evidence that forced them

## D1 (2026-08-29 11:55, after E0 fp32-only replication, before any quantized result was inspected)

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

## D2 (2026-08-29 12:31, after the gate-2 pilot at m=16, n=2048, before E1/E3 results were analyzed)

**Evidence (pilot, m=16, n=2048, 8000 steps, 4 seeds, QAT):** the models are at capacity for every p
(live fraction k/n = 0.44/0.20/0.07/0.04 at p = 0.002/0.005/0.02/0.05; F = 55.9/25.3/8.6/5.2; κ = 4.8/6.2/10.3/13.5),
κ has plateaued (< 1 % change over the last 2000 steps), R_soft(b) is monotone (0 violations), and
**b_crit(R_soft = 0.95) = 3.38 / 3.42 / 3.54 / 3.41 (seed SD ≤ 0.11) — flat across a 10× range of F and a
2.8× range of κ.** The preregistered law predicts 5.2–5.4 with a 1.5-bit spread. R_soft(b=3) ≈ 0.92 and
R_soft(b=4) ≈ 0.99 at every p. No analysis choice changes this: the hard count, where defined, crosses in the
same place.

**Where the §2.2 derivation goes wrong.** It equates the *added* quantization interference with a fixed
fraction of the explainable variance ν⁰. But at F ≫ 1 the fp32 model already carries interference many times
ν⁰ per feature (the Welch bound gives Σ_j (W_i·W_j)² ≳ F − 1) and survives it through the ReLU with a strongly
negative bias (E0: b_i ≈ −0.5 to −0.65). What the loss responds to is the added cross-talk *relative to the
existing* cross-talk. Per pair, added variance ≈ 2σ_ε² (unit live columns), existing ≈ 1/m, so the relative
increase is 2mσ_ε² = 2‖ε_i‖²/‖W_i‖² — exactly the squared relative perturbation of each feature direction,
σ_rel² = ‖Q(W)−W‖_F²/‖W‖_F² (dead columns contribute zero to both). **F cancels.** Under a per-tensor max-abs
quantizer σ_rel = κ/(√12·qmax), so the corrected prediction is

    σ_rel,crit  independent of F          and          b_crit ≈ 1 + log2(1 + κ / (√12·σ_rel,crit)) = const + log2 κ.

Superposition enters the bit budget only through κ (outliers inflate the per-tensor step), not through F.
The same argument applies to any perturbation, so E3's noise-like families should show slope 0 on the
σ_rel axis, not −½; PTQ quantization (no adaptation) should show the log2 κ dependence; QAT arms can
additionally adapt κ, which would flatten even the κ term (the pilot is consistent with that).

**Revised decision rule for E1/E3 (written before their results):** slope of b_crit vs log2 F in [−0.1, 0.15]
with the CI excluding 0.5 → the preregistered ½-law is **rejected**; the κ-only law is supported if b_crit
tracks log2 κ with slope ≈ 1 where κ is not free to adapt (PTQ, E3; GPT-2, §5A) and if σ_rel,crit is flat in F.
The original PREDICTIONS.md bands are still scored as written.

## D3 (2026-08-29 12:45, after the pilot-checkpoint E3 run, before E1 results were analyzed)

**Evidence:** two artefacts in the D1 soft count surfaced on the post-hoc damage families.
(i) Clipping ρ at 0 gives a positive Monte-Carlo bias on the ~1000–2000 unrepresented features whenever the
damaged and reference models are evaluated with different sample counts (R_soft = 1.16–1.19 > 1 for RTN at
b ≥ 4 in the first pilot-E3 run). (ii) The unclipped Σρ_i counts damage to *unrepresented* features: Gaussian noise
(unlike RTN, whose lattice contains 0) wakes every dead column, and at ℓ = 4 % live columns a 5 % drop in Σρ
is produced by ρ ≈ −0.0015 on each of ~2000 dead features — not a statement about the represented features.

**Change:** primary retention curve = **R_live(b) = Σ_{i ∈ live(fp32)} ρ_i(b) / Σ_{i ∈ live(fp32)} ρ_i(fp32)**,
the explained variance of the features that the fp32 model of the same (config, seed) represents
(‖W_i‖ ≥ 0.5), unclipped, with reference and damaged/quantized models evaluated on identical samples
(paired). This is the soft analogue of the preregistered "features retained" count; damage to features the
model never represented is reported separately (R_soft, all features). Threshold and interpolation rules as in D1.
