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

## Single-seed Optuna gain was seed-driven

**Symptom.** The Optuna best trial (+29.57) beat the V2 single-seed baseline
(+27.60) by 1.97 points (~7%), but the V2 baseline was a single seed=42
result from the V2 ablation. A 1.97-point gap is within the variance I
observed across Optuna trials of similar quality (trials 3, 9, 10, 13, 17
all landed in [+28.2, +29.6]).

**Resolution.** `run_robustness.py` re-trained both configs across 5 seeds
at 20k steps each. The result: V2 default mean = +28.78 (std 0.60), V3 tuned
mean = +27.58 (std 1.82). The single-seed Optuna gain was indeed seed
luck — V2 default actually has a higher mean once evaluated fairly. The V3
narrative in `decisions.md` was updated to report this honestly rather than
keep the inflated single-seed number.

## Two robustness runs ran concurrently

**Symptom.** The first robustness run was launched via PowerShell
`run_in_background:true`. Twenty minutes in I checked and saw the output
file at 0 bytes with no python process visible in `Get-Process`. I assumed
it had died silently and restarted it via Bash `run_in_background:true`
with `python -u` for unbuffered stdout. In fact the first run was still
alive — Python's stdout was buffered because PowerShell's pipe is not a tty,
so the output file remained empty for the full duration of each ~30-minute
training. My `Get-Process` check missed the live process because of a race.
Both runs then proceeded in parallel, each writing to the same
`V3/outputs/robustness/` directory.

**Resolution.** Both runs completed at roughly the same time. The later
writer's outputs are what is on disk now (run 2 — the Bash-launched one).
The two runs interfered with each other through CPU contention, so the
absolute numbers are slightly noisier than they would be from a clean
serial run; but both configs were subjected to identical interference at
identical seeds, so the relative comparison between V2 default and V3 tuned
is preserved. The conclusion (V2 default ≥ V3 tuned) is the same in both
runs' summaries.

**Lessons.** Always pass `-u` to Python when launching long jobs through
a pipe-redirected harness, or use `PYTHONUNBUFFERED=1`. And do not restart
a background job based on a 0-byte log file without independently
confirming the process is actually dead (e.g., by Task-Manager PID lookup,
not just `Get-Process`).

## DQN buffer alpha/beta not exposed through agent constructor

**Symptom.** The PER replay buffer in V2 supports `alpha` and `beta` (with
beta annealing) as constructor arguments, but `DQNAgent` does not plumb
those through — it instantiates `PrioritizedReplayBuffer(capacity, obs_shape)`
with the buffer's defaults (α=0.6, β_start=0.4, β_end=1.0).

**Resolution.** Decided not to modify the V2 `DQNAgent` for V3 — V3's whole
premise is that V2 is the substrate being tuned, not rewritten. The Optuna
search space therefore excludes PER's own hyperparameters. Noted as future
work in `decisions.md` ("re-introduce PER hyperparameters into the search").
