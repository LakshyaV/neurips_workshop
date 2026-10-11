# CLAUDE.md — working notes for this repo

## What this is
A one-day preregistered study: does a layer doing more superposition (features per dimension, F) need more bits?
Answer: no — the critical bit-width is set by the outlier ratio κ = max|W|/rms(W) and a universal perturbation
tolerance σ*, not by F. Read in this order: `PREDICTIONS.md` (never edit), `DEVIATIONS.md` (dated changes),
`RESULTS.md` (all numbers + preregistration scoring), `paper/main.tex`.

## Environment
- `source .venv/bin/activate && export PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1`
- Python 3.12 venv (uv), torch 2.13 on Apple MPS, fp32 everywhere. GPT-2 weights are in the HF cache.
- Papers: v1 `main.tex` (12 pp), v2 `main_v2*.tex` (4 pp), v3 `main_v3.tex` (9 pp long form), **v4 `main_v4*.tex` = what was submitted to AXIOM on 2026-08-29** (git tag `axiom-submitted`; keywords variant `main_v4_keywords.tex`; named copy `main_v4_named.tex`). Compile with tectonic. Post-submission edits in the sources are for camera-ready only.

## Layout
- `src/quant.py` batched fake quantizers (per-model tensor `qmax`/`mode`, STE) + post-hoc damage ops
- `src/model.py` batched tied-weight TMS; `src/train.py` batched QAT trainer + evaluation; `src/metrics.py`
- `src/sweep.py` registry: `--exp E0|pilot|E1|E2|E5|E6 [--steps N] [--only <tag>]`; writes `results/<exp>.parquet`
- `src/damage.py` E3 (post-hoc damage on checkpoints, paired clean copy in every stack)
- `src/lm_eval.py` / `src/lm_mixed.py` / `src/analysis_lm.py` GPT-2 per-matrix sweeps and mixed precision
- `src/analysis.py` (E1: summaries, b_crit, seed bootstrap, figs 1/2/4), `src/analysis2.py` (E2/E3/E5/E6)
- `src/regen_rows.py` rebuild per-feature rows from `results/ckpt/*.pt`; `src/merge_e1.py`
- `tests/test_quant.py` (11 tests), `tests/crosscheck_numpy.py` (port vs `tms_numpy_pilot.py`)
- v2 (ICML draft): `paper/main_v5.tex`; `src/lm_eval2.py` architecture-generic harness (GPT-2 +
  Llama-class adapters, `rtn_gw` g=128, tail κ stats, `--stats-only` spike mode, `--mechanics-check`);
  `src/validate_port.py` (GPT-2 rerun vs `results/lm_layer_thresholds.csv`); `src/rotate.py` RMSNorm
  folding + QuaRot-style R1 Hadamard rotation (`--exactness`, no download); `src/analysis_pc_stats.py`
  per-channel κ-statistic investigation. `PREDICTIONS_V2.md` is the Phase 2 prereg DRAFT — binding only
  once posted to OSF, **no Llama-3.1-8B damage runs before its timestamp** (`PREDICTIONS.md` stays never-edit)

## Conventions that matter
- Primary retention metric is `R_live` (explained variance of the features live at fp32, unclipped, paired
  evaluation). The preregistered hard count is reported where defined. See DEVIATIONS D1/D3 before changing this.
- `b_crit` = interpolated crossing of R = 0.95 on b ∈ {3..8}; b = 2 is never fitted.
- κ is measured on the fp32 arm; κ_live = κ·√(live fraction) is what enters the corrected law.
- One row per (model, feature) in the parquet; everything else is a groupby. Do not add a second schema.

## Gotchas (each cost real time)
- Never run more than two MPS jobs at once; a third stalls all of them.
- The [K, n, n] Gram in `feature_dims` must stay chunked (a 512×2048² tensor thrashed the GPU).
- Never `git add -A` without the .gitignore in place: raw parquet/ckpt/logs are ignored on purpose (a history
  rewrite to purge a 128 MB blob deleted tracked results from the working tree once).
- Piping a long job through `tail` hides progress until exit; log to a file instead.
- `Series.sem` is a pandas method — use `df["sem"]`.
- Per-tensor RTN on GPT-2's κ ≈ 160–190 matrices is pathological (non-monotone in bits, super-additive across
  matrices); use per-channel scaling for anything joint.
- Git: always create the branch BEFORE committing; never `git checkout -B main origin/main` while a local commit is not on
  a pushed branch (it orphans the commit and deletes its files from the working tree; recover via `git reflog`).
