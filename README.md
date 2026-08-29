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
- `src/sweep.py` experiment registry (E0, pilot, E1, E2, E5, E6); `src/damage.py` (E3, post-hoc); `src/lm_eval.py` (§5A)
- `src/analysis.py`, `src/analysis2.py` — b_crit extraction, seed-bootstrap fits, figures
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
python -m src.lm_eval                   # GPT-2 small per-matrix sweep
python -m src.analysis E1; python -m src.analysis2 E3; python -m src.analysis2 E2; python -m src.analysis2 E5; python -m src.analysis2 E6
```
