# V3 Technical Challenges

Friction encountered during V3 work and how it was resolved.

## Broken virtual environment

**Symptom.** The `.venv` in the repo root pointed at
`C:\Users\davis\Anacanda\python.exe` (note the typo — directory no longer
exists on the machine). Any command that activated `.venv` died immediately
with "No Python at ...".

**Resolution.** Bypassed the broken venv and ran V3 work against the system
Python at
`C:\Users\davis\AppData\Local\Programs\Python\Python312\python.exe`, which
already had `torch`, `gymnasium`, and `highway-env` installed. Added
`optuna` to that interpreter via `pip install optuna`. Did not rebuild the
venv — the system interpreter is sufficient for V3.

**Lessons.** `pyvenv.cfg` records an absolute path to the originating
interpreter. If that interpreter is renamed or moved, the entire venv is
silently broken. For deliverables that need to survive machine changes, a
`requirements.txt` plus a system interpreter is more robust than a checked-in
venv reference.

## Optuna matplotlib visualizations marked experimental

**Symptom.** `optuna.visualization.matplotlib.plot_optimization_history`,
`plot_param_importances`, and `plot_parallel_coordinate` all emit
`ExperimentalWarning` at runtime (Optuna 4.8.0, May 2026).

**Resolution.** Wrapped each plot call in `try/except` so a future Optuna
release that breaks the matplotlib interface degrades gracefully (skips the
plot, warns) instead of crashing the whole run. The Plotly variants are
stable, but require an extra dependency I did not want to add.

## PowerShell stderr wrapping for non-error output

**Symptom.** Running Python via PowerShell with `2>&1` wraps every stderr
line (including normal Optuna `INFO`-level log lines) in a
`NativeCommandError` block, which makes the run *look* like it crashed when
it did not, and sets `$?` to `$false` even on exit code 0.

**Resolution.** Avoided `2>&1` on native commands inside PowerShell and read
both streams directly. Knew this was a Windows PowerShell 5.1 quirk going in,
so it didn't burn investigation time, but worth noting for the next person
who tries to script this on Windows.

## Single-seed Optuna gain may be seed-driven

**Symptom.** The Optuna best trial (+29.57) beats the V2 baseline (+27.60)
by 1.97 points (~7%), but the V2 baseline was a single-seed (seed=42) result
from the V2 ablation. A 1.97-point difference is well within the variance I
observed across trials of similar quality (trials 3, 9, 10, 13, 17 all
landed in [+28.2, +29.6]).

**Resolution.** `run_robustness.py` re-trains both the V2 default config and
the V3-tuned config across 5 seeds at 20k steps. If the mean of the tuned
config cleanly clears the V2 mean by more than one standard deviation, the
gain is real. If it doesn't, the V3 conclusion has to be reported honestly
as "tuning produces a config that matches the baseline but does not
significantly improve it" — and that is still useful information about how
well-tuned the V2 default already was.

## DQN buffer alpha/beta not exposed through agent constructor

**Symptom.** The PER replay buffer in V2 supports `alpha` and `beta` (with
beta annealing) as constructor arguments, but `DQNAgent` does not plumb
those through — it instantiates `PrioritizedReplayBuffer(capacity, obs_shape)`
with the buffer's defaults (α=0.6, β_start=0.4, β_end=1.0).

**Resolution.** Decided not to modify the V2 `DQNAgent` for V3 — V3's whole
premise is that V2 is the substrate being tuned, not rewritten. The Optuna
search space therefore excludes PER's own hyperparameters. Noted as future
work in `decisions.md` ("re-introduce PER hyperparameters into the search").
