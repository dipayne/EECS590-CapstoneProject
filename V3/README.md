# V3: Final Version

Due: 2026-05-13.

V3 builds on V2 (the 13-algorithm benchmark on `highway-v0`) by applying
Bayesian hyperparameter optimization to the strongest V2 agent
(DQN + Double + Dueling + PER) and articulating which other post-V2 class
topics were considered but deliberately not implemented.

## What is here

- [decisions.md](decisions.md) — what V3 implemented (Bayesian HPO via
  Optuna/TPE) and what was considered and skipped (multi-agent,
  distributional, constrained, model-based, imitation, hierarchical, offline)
  with my justification for each call.
- [technical-challenges.md](technical-challenges.md) — friction encountered
  during V3 work and how it was handled.
- [citations.md](citations.md) — collaborators, tools, and references.
- `run_optuna.py` — Bayesian HPO study (TPE sampler, 6-dim search space).
- `run_robustness.py` — 5-seed re-training of V2 default vs V3-tuned config
  to confirm the gain is not seed luck.
- `outputs/optuna/` — study database, best params, optimization history,
  parameter importance, parallel coordinate plot, trial table.
- `outputs/robustness/` — per-seed return curves, summary stats, multi-seed
  learning curves, final-return boxplot.

## Headline result

| Config | Final avg-50 return |
|---|---|
| V2 default DQN(D+D+PER), 5-seed mean ± std | **+28.78 ± 0.60** |
| V3 Optuna-tuned best, 5-seed mean ± std | **+27.58 ± 1.82** |
| (Single-seed Optuna trial 9, used to pick the winner) | +29.57 |
| (Single-seed V2 ablation result, seed=42) | +27.60 |

**Bayesian HPO did not produce a robust improvement over V2's defaults.** The
single-seed +29.57 from Optuna's trial 9 was a high draw from a wider
distribution; once both configs are evaluated over 5 matched seeds, V2
default actually has a higher mean and one-third the variance. This is the
honest finding and is discussed in detail in [decisions.md](decisions.md).

See `decisions.md` for the full analysis (parameter importance, winning
hyperparameters, robustness story, and why each skipped algorithm was
skipped).

## Reproducing V3

```bash
# From repo root
python -m pip install -r V2/requirements.txt
python -m pip install -r V3/requirements.txt

# Bayesian HPO (long — ~8h on CPU)
python V3/run_optuna.py --trials 20 --steps 20000

# Multi-seed robustness re-run (~4h on CPU)
python V3/run_robustness.py
```

The Optuna study is SQLite-backed, so re-running `run_optuna.py` resumes
from where it stopped rather than restarting.

## V1 and V2

- [../README.md](../README.md) — project overview and V1 (logistics grid DP).
- [../V2/README.md](../V2/README.md) — V2 RL benchmark, ablation, saliency.
