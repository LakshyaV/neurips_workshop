# PREDICTIONS.md — preregistration (written before any experiment ran)

Written 2026-08-29 (hour 0:30 of the plan). **This file is never edited after commit.**
Scoring happens in `RESULTS.md`.

## The law

    b_crit = 1 + log2( 1 + κ · sqrt(F/3) ),   F = k_fp32 / m,   κ = max|W| / rms(W)  (measured)

Asymptote: `b_crit ≈ const + ½·log2 F` — one extra bit per 4× features-per-dimension.

| F   | predicted b_crit (κ = 4.5) |
|-----|------|
| 1   | 2.85 |
| 2   | 3.22 |
| 4   | 3.63 |
| 8   | 4.06 |
| 16  | 4.51 |
| 32  | 4.97 |
| 64  | 5.45 |
| 128 | 5.93 |
| 256 | 6.41 |

**Only the slope is the claim.** The intercept is constrained (κ is measured, not fitted) but
carries O(1) constants from the ReLU/bias/tied-weight approximations in §2.2 of the plan.

## Definitions fixed in advance

- `ν_i⁰ = p/3 − p²/4`; `ρ_i = 1 − ν_i/ν_i⁰`; `N_ret = #{i : ρ_i ≥ 0.5}` (primary; 0.25 / 0.75 as sensitivity).
- `b_crit` per (m, n, p, seed) = smallest b at which `N_ret(b) ≥ 0.95 · N_ret(∞)` (same seed, same x stream),
  linearly interpolated between the integer bit-widths bracketing the crossing. 90 % and 99 % variants reported.
- `F = N_ret(∞)/m`, measured from the same config's own fp32 arm.
- `κ` measured on the latent (pre-quantization) weights of the **fp32 arm** of the same (m, n, p, seed).
- Quantizer: per-tensor symmetric max-abs, `qmax = 2^(b−1) − 1`, STE, quantization-aware training (as in the pilot).

## Fit inclusion rules (fixed now)

1. **b = 2 is excluded from every fit.** `qmax = 1` zeroes all `|w| < amax/2`, a different regime and
   (at b ≤ 1.58) a different scale rule. b = 2 curves are plotted but never fitted. Interpolation of a crossing
   between b = 2 and b = 3 is reported as *censored* (`b_crit ≤ 3`).
2. A config contributes a `b_crit` point only if its crossing is bracketed by two fitted bit-widths, i.e.
   `3 ≤ b_crit ≤ 8`. Crossings below 3 or above 8 are censored and plotted as arrows, not fitted.
   (This is the "sizing gate": cells where 3-bit quantization already retains ≥ 95 % have no measurable
   `b_crit` and are controls.)
3. Each (m, n, p) contributes one point = mean over 4 seeds; slope CI = bootstrap over seeds (resample the
   4 seeds with replacement, refit, 1000 reps, report 2.5–97.5 %).
4. Primary fit: OLS of `b_crit` on `log2 F`. Secondary (κ-corrected): OLS of `b_crit − log2 κ` on `log2 F`.
5. Continuous-axis robustness: regress `b_crit` extracted on the axis `b_eff = 1 + log2(1 + amax/(√12·σ̂_ε))`,
   where `σ̂_ε = rms(Q(W) − W)` is the measured residual.

## P1 — the money plot (decision bands)

Slope of `b_crit` vs `log2 F`, with the bootstrap CI:

- slope ∈ [0.40, 0.60] → **confirmed** (½ bit per doubling of F).
- slope ∈ [0.15, 0.35] → a different, stable exponent; report it as the result.
- slope < 0.15 (CI includes 0) → **hypothesis dead**: precision cost independent of superposition.
- CI wider than 0.3 → underpowered; report as inconclusive, do not spin.

## P2 — κ enters as log2 κ

At fixed (m, n, p), moving κ (weight clipping at `c·rms(W)`, c ∈ {2, 3, 4, ∞}) shifts `b_crit` by
`log2(κ₁/κ₂)`. Decision: slope of `b_crit` on `log2 κ` ∈ [0.7, 1.3] → confirmed; ≈ 0 → κ does not enter;
in between → partial.

## P3 — importance ordering (power-law importance `I_i = (i+1)^−α`)

Features are lost in importance order and quantization does **not** degrade the ordering relative to fp32.
Baseline prefix fraction at fp32 is imperfect (pilot: ≈ 0.91). Decision: prefix fraction at `b ≥ 3` within
0.10 of the fp32 value (same seed) → confirmed; a drop > 0.10 → ordering is degraded by quantization.

## E4 — translation vs tilt

Under a capacity mechanism, `ρ` of the 3 best-reconstructed features (ranked at fp32, same seed) changes by
< 0.05 for all `b ≥ b_crit` and the retention profile shifts. If `ρ₀` falls by > 0.15 at `b_crit`, "capacity"
is the wrong frame and the mechanism is uniform degradation.

## E3 — the any-damage null (post-hoc damage on the fp32 checkpoints)

Damage types applied to the same fp32 weights: (a) RTN quantization (PTQ), (b) dithered quantization,
(c) additive Gaussian noise, (d) rank truncation to top-s singular directions, (e) random column dropout.
Common axis: `σ_rel = ‖D(W) − W‖_F / ‖W‖_F` at the 95 %-retention threshold.

Predictions from the noise-floor derivation (it uses only "perturbation adds interference"):
- noise-like damage (a, b, c): `log2 σ_rel,crit` vs `log2 F` slope **−½**.
- structured damage (d, e): threshold set by geometry, not by a noise floor: slope ≈ 0 (dropout at
  capacity kills exactly the dropped fraction; rank truncation removes whole directions).

Reading: if (d, e) also show −½, the law is about capacity under any damage (weaker paper, reported as such).
If only (a–c) do, it is a law about *noise-like* damage, of which lattice quantization is one instance; the
lattice-specific content is then whatever E2 finds.

## E2 — lattice structure

`Q` (shared lattice) vs `Q_u` (per-element dithered lattice) vs stochastic rounding, all QAT at matched Δ.
No direction preregistered. If `N_ret(Q) > N_ret(Q_u)` by more than the seed CI at ≥ 2 bit-widths, models
exploit lattice alignment. If equal, only the step size matters and the law is a pure noise-floor law.

## §5A — LM bridge (GPT-2 small, per weight matrix, PTQ)

Per-matrix continuous damage axis: additive Gaussian noise at `σ_rel · rms(W)`; threshold `σ_crit` where
`ΔNLL = 0.05` nats (0.02 and 0.10 as sensitivity). Superposition proxy `F̂ = d_in / PR(input covariance)`.
- Test 1: `log2 σ_crit` vs `log2 F̂` slope ∈ [−0.65, −0.35] → consistent with the toy law; ≈ 0 → not.
- Test 2: `b_crit = 1 + log2(1 + κ/(√12·σ_crit))` (the RTN bit-width whose residual equals σ_crit),
  regressed on `log2 F̂` after subtracting `log2 κ`: slope ≈ ½.
Embeddings (`wte`, tied to `lm_head`) are excluded from the primary run.

## Things we commit to reporting regardless of outcome

κ trajectories (plateau check), threshold sensitivity (ρ ∈ {0.25, 0.5, 0.75}; retention ∈ {90, 95, 99 %}),
per-channel vs per-tensor, seed CIs, every censored cell, and the E3 null in full.
