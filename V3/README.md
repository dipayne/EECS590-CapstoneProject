# V3 Final Report and README

**Course:** EECS 590, Advanced Topics in EE and CS, Sp26
**Author:** Davis Payne
**Date:** 2026-05-13
**Repository:** https://github.com/dipayne/EECS590-CapstoneProject

This file is both the V3 README and the final report. It describes the
problem, the project's versioning history, the V3 repository structure,
the workflow that produced the V3 results, what I chose to implement and
why, what I considered and chose not to implement, how someone else can
reproduce the V3 work, and a closing reflection.

---

## 1. Problem setting

The capstone project trains a reinforcement learning agent to drive a
single protagonist vehicle in Farama's `highway-v0` environment.

* **Observation.** A 25 dimensional vector describing the ego vehicle and
  its four nearest neighbors: position, velocity, lane, and presence flags.
* **Action space.** Five discrete meta actions: lane left, idle, lane
  right, faster, slower. Each meta action persists for many underlying
  simulator steps, so the agent is already operating on temporally
  extended options.
* **Reward.** Dense per step. A positive term for forward speed and lane
  preference, a large negative term on collision.
* **Number of learning agents.** Exactly one. Surrounding traffic uses
  scripted IDM and MOBIL drivers, which are not learning.

This framing matters for the V3 algorithm choice: the environment is a
single protagonist task with a dense extrinsic reward and a low
dimensional hand engineered observation. Several of the post V2 class
topics target problem structures this environment simply does not have.

---

## 2. Versioning notes

* **V1.** Logistics grid solved with dynamic programming. Covered Bellman
  iteration, value iteration, and policy iteration on a tabular MDP.
* **V2.** Thirteen reinforcement learning algorithms (classical and deep)
  benchmarked on `highway-v0`. Included a DQN ablation that isolated
  Double, Dueling, and Prioritized Experience Replay as separate
  components. Strongest agent: DQN with all three enhancements,
  abbreviated DQN(D+D+PER), achieving +27.60 final avg 50 return at
  seed 42.
* **V3 (this version).** Final deliverable. Applies Bayesian
  hyperparameter optimization (Optuna with the TPE sampler) to
  DQN(D+D+PER), reports the honest multi seed result, and justifies the
  algorithm choice against the other genuine post V2 alternatives for
  this environment (Hyperband and model based RL).

A pointer back to earlier versions:

* [`../README.md`](../README.md) for the project overview and V1.
* [`../V2/README.md`](../V2/README.md) for the V2 benchmark this V3 work
  builds on.

---

## 3. Repository structure

```
V3/
  README.md                 this file (final report and entry point)
  technical-challenges.md   friction encountered during V3 work
  citations.md              collaborators, AI assistants, references
  requirements.txt          V3 specific dependencies (optuna)
  run_optuna.py             20 trial TPE study over 6 hyperparameters
  run_robustness.py         5 seed re-training of V2 default vs V3 best
  outputs/
    optuna/
      study.db                  SQLite store, resumable
      best_params.json          winning HPs and final return
      trials.csv                full per trial table
      optimization_history.png  trial value over time
      param_importances.png     fANOVA importance per HP
      parallel_coordinate.png   trial trajectories across the space
    robustness/
      summary.json              mean/std/min/max per config
      returns_by_seed.json      full per episode return curves
      learning_curves.png       mean and standard deviation bands
      final_avg50_box.png       boxplot of final return across seeds
```

The V2 directory under `../V2/` provides the agent implementation,
environment wrapper, and replay buffer that V3 reuses. V3 does not
reimplement anything from V2; it imports and tunes it.

---

## 4. Workflow

This is the order in which the V3 work was actually produced. It is the
same order another reader should follow when reproducing the results.

1. **Confirm a strong V2 baseline.** The V2 ablation already identified
   DQN(D+D+PER) as the strongest agent. V3 starts there and treats that
   agent as fixed.
2. **Define the search.** Pick six hyperparameters with broad but
   reasonable ranges (see section 5 for the full table) and decide on a
   single seed final avg 50 return as the trial objective.
3. **Run the Optuna study.** `python V3/run_optuna.py --trials 20
   --steps 20000`. The study writes incrementally to
   `V3/outputs/optuna/study.db`, so it is resumable. Total CPU wall on
   the development machine: 8 hours 18 minutes.
4. **Inspect the winner.** Read `best_params.json` and the importance
   plot to understand which hyperparameters drove the result.
5. **Robustness re-run.** `python V3/run_robustness.py`. Trains both the
   V2 default config and the V3 Optuna winner across 5 seeds at 20000
   steps each, to confirm whether the single seed gain survives a
   multi seed evaluation.
6. **Write the report.** Honest reporting of the multi seed result, even
   when it contradicts the single seed headline.

---

## 5. What I implemented in V3

### Bayesian hyperparameter optimization (Optuna, TPE sampler)

**What it is.** A method for searching the hyperparameter space of a
learning algorithm by treating each training run as an expensive
experiment and using probabilistic models to pick the next configuration.
The Tree structured Parzen Estimator (TPE) maintains two density models,
one over good configurations and one over poor configurations, and on
each new trial proposes the configuration that maximizes their ratio.
Compared with grid search or random search, TPE focuses budget on
promising regions after a small warm up phase.

**Why it is the right tool here.** The thing that was clearly
underdeveloped in my V2 work was that every DQN hyperparameter (learning
rate, discount factor, target network update frequency, epsilon decay
schedule) was either a textbook default or a guess. Bayesian HPO is the
directly applicable response to that gap. It is also a method I will use
in every future machine learning project regardless of domain, so the
effort transfers beyond this class.

**What I did.** Wrote `V3/run_optuna.py`, which runs a 20 trial TPE
study on DQN(D+D+PER) over six hyperparameters:

| Hyperparameter | Search range | Type |
|---|---|---|
| `lr` | [1e-5, 5e-3] | log uniform |
| `gamma` | [0.95, 0.999] | uniform |
| `batch_size` | {32, 64, 128, 256} | categorical |
| `target_update_freq` | [100, 1000] | integer |
| `eps_decay_pct` | [0.4, 0.9] | uniform (fraction of n_steps) |
| `train_start` | [200, 2000] | integer |

Each trial trained the agent for 20000 environment steps. Total CPU wall
time: 8 hours 18 minutes. The study is SQLite backed, so it can be
resumed, queried, and re-analyzed without re-training.

**What it found.** The single trial winner (trial 9) reached an avg 50
return of **+29.57**, which appeared to beat the V2 published baseline
(+27.60 at seed 42) by +1.97. The fANOVA parameter importance plot
showed that `gamma` (0.31) and `lr` (0.28) together explain about 60
percent of the variance in final return on `highway-v0`. The winning
configuration pushed `gamma` near its upper bound (0.998) and raised
`lr` by 5.6 times over the V2 default.

**Where the honest story diverges from the headline.** I ran a follow up
robustness study (`V3/run_robustness.py`) that re-trained both the V2
default config and the Optuna winner across five seeds each, again at
20000 steps:

| Config | mean ± std | [min, max] |
|---|---|---|
| V2 default DQN(D+D+PER) | **+28.78 ± 0.60** | [+27.69, +29.32] |
| V3 Optuna tuned best | **+27.58 ± 1.82** | [+24.88, +30.17] |

Bayesian HPO did not robustly improve over the V2 default. The single
+29.57 was a high draw from a noisier distribution. The V2 single seed
+27.60 was conversely a low draw from its own quieter distribution. Once
both configs are evaluated on a common five seed footing, the V2 default
has a higher mean and one third the variance. The reason is that the
Optuna objective was a single seed final avg 50, exactly the quantity
that suffers from seed luck. A more honest objective would have been
median over k ≥ 3 seeds per trial, which would have tripled the study
wall time from 8 to 24 hours.

I am reporting this finding rather than the inflated single seed number
because it is the truthful one. The implication is not that Bayesian HPO
failed. It is that the V2 default hyperparameters were already well
chosen for this task, and HPO confirmed the remaining headroom is small
enough to be drowned out by seed noise. That is a legitimate result in
its own right, and it is the kind of result a more polished publication
would also have to report.

Plots: `V3/outputs/optuna/optimization_history.png`,
`param_importances.png`, `parallel_coordinate.png`,
`V3/outputs/robustness/learning_curves.png`, `final_avg50_box.png`.

---

## 6. What I could have done instead, and why I did not

Most of the post V2 class topics target problem structures my single
agent, dense reward, vector observation environment does not have. They
are not credible alternatives for *this* project and are not discussed
here. The two algorithm families that *are* genuine alternatives for
improving a single agent DQN on `highway-v0` are covered below.

### 6.1 Hyperband (alternative HPO method)

**What it is.** A non Bayesian HPO method built on successive halving:
start many short trials, kill the worst on a small budget, give the
survivors more budget, repeat. Useful when training time scales strongly
with budget and bad configurations can be detected early.

**Why it was a credible alternative.** Hyperband and TPE solve the same
problem from opposite directions. TPE picks better configurations by
modelling success and failure densities. Hyperband picks better
configurations by aggressively killing the bad ones before they finish.
Either is defensible for tuning DQN hyperparameters on `highway-v0`, and
Hyperband can outperform TPE on a fixed compute budget when bad
configurations are easy to spot early.

**Why TPE was the better choice.** Hyperband's whole advantage rests on
being able to detect bad configurations early. On `highway-v0`, the
first several thousand environment steps of any DQN training are
dominated by epsilon greedy exploration, so early returns are nearly
random and look the same across good and bad configurations. A config
whose lr is too high produces identical "still mostly exploring" returns
in steps 0 to 3000 as a config that will end up winning. Killing trials
on that signal would have thrown out winners. TPE does not depend on
early stopping. It just sees the final return of completed trials and
learns from those, which fits this regime cleanly. If I later move to a
multi seed per trial objective (which makes each trial 3 to 5 times more
expensive) and a longer per trial budget, the calculus shifts and
Hyperband, or equivalently Optuna's `HyperbandPruner` layered on the TPE
study, becomes worth its complexity.

### 6.2 Model based RL (PlaNet, Dreamer, TD-MPC, World Model)

**What it is.** Learn a model of environment dynamics, most often a
recurrent latent state space model trained on observation reconstruction
(PlaNet, Dreamer, World Model), or a deterministic encoder dynamics pair
with a value head (TD-MPC). Either plan inside the learned model (CEM,
iLQR, MPPI) or train a policy on imagined trajectories. Drastically more
sample efficient than model free methods on high dimensional
observations.

**Why it was a credible alternative.** The honest finding of my HPO
study is that the V2 DQN is already close to whatever ceiling the
hyperparameter space offers. To push past that ceiling I would need to
change the *algorithm*, not the *hyperparameters*. Model based RL is the
post V2 class of methods that does that. Instead of tuning DQN better,
you learn a model of the environment and either plan with it (TD-MPC) or
train a policy in imagination (Dreamer, PlaNet). On vector observations,
TD-MPC in particular has been shown competitive with model free methods
at a fraction of the sample count.

**Why HPO was the better choice for V3.** Three reasons, in priority
order.

1. **Observation geometry.** PlaNet, Dreamer, and the original World
   Model were designed for pixel observations. The hard part they solve,
   compressing a high dimensional image stream into a usable latent
   state, does not exist in `highway-v0`, where the environment already
   provides a 25 dim hand engineered observation. The data efficiency
   gains shrink dramatically on low dim states while the implementation
   lift (encoder, decoder, dynamics model, reward model, imagination
   policy or MPC planner) stays large. TD-MPC is the one method in the
   family that does not require pixel reconstruction, but it is still
   substantially more implementation than tuning the agent I already
   have.

2. **Result quality versus effort, given the time budget.** V3 was a
   two day deliverable on top of an already working V2 baseline.
   Replacing the agent with a model based method I have not implemented
   before (encoder, dynamics, planner, training loop) risks producing a
   half finished implementation that does not actually beat the V2
   baseline I am trying to improve on, and would leave no time to
   honestly characterise whatever result it did produce. HPO is a
   smaller and faster intervention with a result I can defend end to
   end: search space, trials, winner, multi seed re-run that says the
   winner is within baseline noise.

3. **The result HPO produced is itself informative.** Knowing that V2's
   defaults are already near optimal on `highway-v0` is a useful prior
   to bring into any future model based attempt. Without that prior, a
   V4 model based result of "matches V2 default" would be ambiguous.
   Maybe the model based method is bad, maybe DQN is just hard to beat
   on this task. With the V3 HPO result in hand, that ambiguity
   resolves: DQN really is well tuned, and any model based gain is
   measured against a strong baseline, not a weak one.

---

## 7. What I would do with more time

Listed in the order I would attempt them, drawn from class content.

1. **Re-run Optuna with a multi seed objective** (median over k=3 seeds
   per trial). The obvious correction to the seed luck issue identified
   by the robustness study. Triples wall time but produces a winner
   that does not need a separate robustness re-run.
2. **Add Hyperband pruning to the multi seed Optuna study.** Once per
   trial budget grows, killing bad configurations on the cheap budget
   becomes worth the complexity. A small Optuna API change
   (`HyperbandPruner` in place of the default pruner).
3. **A second, narrower Optuna pass** anchored at trial 9's neighborhood
   (small ranges around `lr = 5.65e-4` and `gamma = 0.998`) to test
   whether that region genuinely contains better configurations or is
   all within V2 default noise.
4. **Re-introduce PER hyperparameters** (`alpha`, `beta` annealing
   schedule) into the search space. Requires a small change to
   `V2/src/agents/deep/dqn.py` so the agent can pass these through to
   the replay buffer, which is currently hardcoded to defaults
   alpha = 0.6 and beta = 0.4.
5. **TD-MPC on top of the tuned config.** This is the natural V4
   experiment. With a well tuned model free baseline in hand, a TD-MPC
   re-implementation has a clean reference to be measured against.

---

## 8. How to use V3

V3 reuses the V2 agent and environment wrapper. Run everything from the
repository root.

### 8.1 Environment

```
python -m pip install -r V2/requirements.txt
python -m pip install -r V3/requirements.txt
```

The V3 requirements file only adds `optuna`. The V2 file pins
`torch`, `gymnasium`, `highway-env`, `numpy`, `matplotlib`, and `tqdm`.

### 8.2 Run the Bayesian HPO study

```
python V3/run_optuna.py --trials 20 --steps 20000
```

Other useful invocations:

```
python V3/run_optuna.py --trials 3  --steps 2000     # smoke test
python V3/run_optuna.py --trials 50 --steps 10000    # longer run, cheaper trials
python V3/run_optuna.py --trials 20 --steps 20000 --study-name my_study
```

The study is SQLite backed at `V3/outputs/optuna/study.db`. Re-running
the same `--study-name` resumes from where the previous invocation
stopped. Artifacts (best params, importance plot, parallel coordinate
plot, optimization history, trials CSV) are written to
`V3/outputs/optuna/` after each invocation.

### 8.3 Run the robustness comparison

```
python V3/run_robustness.py
```

Re-trains both the V2 default config and the V3 Optuna winner across 5
seeds at 20000 steps each. Writes `summary.json`, `returns_by_seed.json`,
`learning_curves.png`, and `final_avg50_box.png` to
`V3/outputs/robustness/`. Wall time on a CPU only laptop is about 4
hours.

### 8.4 Inspect the results

```
cat V3/outputs/optuna/best_params.json
cat V3/outputs/robustness/summary.json
```

For visualization, open the PNGs in `V3/outputs/optuna/` and
`V3/outputs/robustness/`.

### 8.5 Reproducibility notes

* Optuna's TPE is seeded (`SEED = 42` in `run_optuna.py`), but each
  trial also adds `trial.number` so different trials get different RNG
  states.
* `run_robustness.py` seeds both NumPy and PyTorch per training run.
* Single CPU process is assumed. Running two trials in parallel
  introduces CPU contention that perturbs PyTorch's BLAS thread
  scheduling and produces slightly different numerical results despite
  identical seeds. See `technical-challenges.md` for the incident
  report on this.

---

## 9. Reflection

The most honest thing I can say about V3 is that I picked the post V2
topic best matched to my environment, ran it carefully, and reported the
result the analysis actually produced, including the result I would
rather not have reported. The single seed Optuna winner (+29.57) looked
like a clean 7 percent improvement over the V2 baseline. The multi seed
robustness re-run showed that improvement was within noise. I wrote that
finding into the repo rather than burying it because the V3 spec rewards
critical thought and the truthful finding is the one a careful reader
would catch anyway.

Of the two credible alternatives, Hyperband and model based RL, neither
was a better V3 investment than Bayesian HPO via TPE. Hyperband loses
its main advantage on a problem where early returns do not predict final
returns. Model based RL targets a difficulty (high dimensional pixel
observations) that my environment has already resolved. HPO targeted the
actual gap in my existing setup, produced a complete and honest result
in the available time, and gave me a defensible baseline against which
to measure any future model based attempt.

---

## 10. Pointers into the repository

* [`V3/technical-challenges.md`](technical-challenges.md) friction
  encountered during V3 (broken venv, PowerShell stdout buffering, the
  concurrent runs incident).
* [`V3/citations.md`](citations.md) collaborators, AI assistants, and
  references.
* [`V3/run_optuna.py`](run_optuna.py) the Bayesian HPO study.
* [`V3/run_robustness.py`](run_robustness.py) the multi seed comparison.
* [`V3/outputs/optuna/`](outputs/optuna/) Optuna artifacts.
* [`V3/outputs/robustness/`](outputs/robustness/) robustness artifacts.
* [`../V2/README.md`](../V2/README.md) the V2 benchmark this report
  builds on.
* [`../README.md`](../README.md) overall project overview.
