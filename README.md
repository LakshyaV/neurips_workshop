# The Critical Bit-Width of Superposition

Does a layer that packs more features per dimension need more bits? This repo tests a preregistered
closed-form law, `b_crit = 1 + log2(1 + κ·√(F/3))` (≈ ½ bit per doubling of features-per-dimension F),
in toy models of superposition (Elhage et al. 2022) trained with quantization-aware fake quantization,
with controls that separate "quantization" from "any damage", and a per-layer bridge to GPT-2 small.

- `PREDICTIONS.md` — preregistration, committed before any experiment ran (never edited).
- `DEVIATIONS.md` — dated changes to the analysis and the evidence that forced them.
- `RESULTS.md` — what happened, including what did not.
- `src/quant.py` batched fake quantizers (uniform / binary / ternary / dithered / stochastic / per-channel) + post-hoc damage ops
- `src/model.py` batched tied-weight TMS; `src/train.py` batched QAT trainer + evaluation; `src/metrics.py`
- `src/sweep.py` experiment registry (E0, pilot, E1, E2, E5, E6); `src/damage.py` (E3, post-hoc)
- `src/lm_eval.py` (§5A per-matrix sweeps), `src/lm_mixed.py` (§5C mixed precision), `src/lm_tokens.py` (§5D per-token critical bits)
- `src/analysis.py`, `src/analysis2.py`, `src/analysis_lm.py`, `src/analysis_tokens.py` — b_crit extraction, seed-bootstrap fits, figures
- `paper/main.tex` — v1 write-up (long form, 2023 style); `paper/main_v2.tex` — **v2: 4-page AXIOM/NeurIPS-2026-workshop submission** (anonymous, official `neurips_2026.sty`, checklist), compile with `tectonic main_v2.tex`
- `tests/test_quant.py` (11 unit tests), `tests/crosscheck_numpy.py` (port vs numpy reference `tms_numpy_pilot.py`)
- `results/*.parquet` one row per (model, feature); `results/ckpt/` weights; `figs/`

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch numpy pandas pyarrow matplotlib scipy transformers datasets tqdm
export PYTORCH_ENABLE_MPS_FALLBACK=1
python tests/test_quant.py && python tests/crosscheck_numpy.py
python -m src.sweep --exp E0            # fp32 replication gate
python -m src.sweep --exp E1 --steps 6000
python -m src.damage                    # E3 on the E1 fp32 checkpoints
python -m src.sweep --exp E2 --steps 6000; python -m src.sweep --exp E5 --steps 6000; python -m src.sweep --exp E6 --steps 6000
python -m src.lm_eval                   # GPT-2 small per-matrix sweep (add --families rtn_pc for per-channel)
python -m src.lm_mixed; python -m src.lm_tokens; python -m src.lm_sae   # mixed precision; per-token and per-SAE-feature critical bits
python -m src.analysis E1; python -m src.analysis2 E3; python -m src.analysis_lm noise; python -m src.analysis_lm pc; python -m src.analysis_lm mixed; python -m src.analysis_tokens; python -m src.analysis_sae; python -m src.analysis2 E2; python -m src.analysis2 E5; python -m src.analysis2 E6
```
