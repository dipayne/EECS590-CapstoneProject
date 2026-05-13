# V3 Final Report

**Course:** EECS 590 — Advanced Topics in EE and CS, Sp26
**Author:** *(your name here)*
**Date:** 2026-05-13
**Repository:** https://github.com/dipayne/EECS590-CapstoneProject

---

## 1. Project context

The capstone project trains reinforcement-learning agents to drive a single
protagonist vehicle in Farama's `highway-v0` environment. The observation is
a 25-dim vector describing the ego vehicle and four nearest neighbors. The
action space is five discrete meta-actions: lane left, idle, lane right,
faster, slower. There is exactly one learning agent; surrounding traffic is
scripted (IDM/MOBIL drivers).

V1 covered a logistics grid solved with dynamic programming. V2 implemented
and compared 13 reinforcement-learning algorithms (classical and deep) on
`highway-v0`, including a DQN ablation that isolated Double, Dueling, and
Prioritized Experience Replay (PER) as separate components. The strongest
V2 agent was DQN with all three enhancements: DQN(Double + Dueling + PER).

V3 builds on that result. Among the post-V2 topics covered in class, most
were targeted at problem structures my environment does not have —
multi-agent coordination on a single-protagonist env, goal-conditioned
hindsight relabelling on a dense reward, unsupervised skill discovery on
an env with an extrinsic reward, hierarchical option discovery on an
action space that is already option-shaped. The methods that *are*
genuine alternatives for my setup — improving a single-agent DQN on a
vector-observation driving env — were two: another HPO technique
(Hyperband), and the model-based RL family (PlaNet, Dreamer, TD-MPC,
World Model). This report covers what I chose and why those two
alternatives were not the better choice.

---

## 2. What I implemented in V3

### Bayesian hyperparameter optimization (Optuna, TPE sampler)

**What it is.** A method for searching the hyperparameter space of a
learning algorithm by treating each training run as an expensive
experiment and using probabilistic models to pick the next configuration.
The Tree-structured Parzen Estimator (TPE) maintains two density models
— one over "good" configurations and one over "poor" configurations —
and on each new trial proposes the configuration that maximizes their
ratio. Compared with grid search or random search, TPE focuses budget on
promising regions after a small warm-up phase.

**Why it is the right tool here.** The thing that was clearly
underdeveloped in my V2 work was that every DQN hyperparameter —
learning rate, discount factor, target-network update frequency, ε-decay
schedule — was either a textbook default or a guess. Bayesian HPO is the
directly applicable response to that gap. It is also a method I will use
in every future machine-learning project regardless of domain, so the
effort transfers beyond this class.

**What I did.** Wrote `V3/run_optuna.py`, which runs a 20-trial TPE study
on DQN(Double + Dueling + PER) over six hyperparameters:

| Hyperparameter | Search range | Type |
|---|---|---|
| `lr` | [1e-5, 5e-3] | log-uniform |
| `gamma` | [0.95, 0.999] | uniform |
| `batch_size` | {32, 64, 128, 256} | categorical |
| `target_update_freq` | [100, 1000] | integer |
| `eps_decay_pct` | [0.4, 0.9] | uniform (fraction of n_steps) |
| `train_start` | [200, 2000] | integer |

Each trial trained the agent for 20,000 environment steps. Total CPU
wall time: 8 hours 18 minutes. The study is SQLite-backed so it can be
resumed, queried, and re-analyzed without re-training.

**What it found.** The single-trial winner (trial 9) reached an avg-50
return of **+29.57**, which appeared to beat the V2 published baseline
(+27.60, seed=42) by +1.97. The fANOVA parameter-importance plot showed
that `gamma` (0.31) and `lr` (0.28) together explain ~60% of the variance
in final return on `highway-v0`. The winning configuration pushed `gamma`
near its upper bound (0.998) and raised `lr` by 5.6× over the V2 default.

**Where the honest story diverges from the headline.** I ran a follow-up
robustness study (`V3/run_robustness.py`) that re-trained both the V2
default config and the Optuna winner across five seeds each, again at
20,000 steps:

| Config | mean ± std | [min, max] |
|---|---|---|
| V2 default DQN(D+D+PER) | **+28.78 ± 0.60** | [+27.69, +29.32] |
| V3 Optuna-tuned best | **+27.58 ± 1.82** | [+24.88, +30.17] |

**Bayesian HPO did not robustly improve over the V2 default.** The single
+29.57 was a high draw from a noisier distribution; the V2 single-seed
+27.60 was a low draw from its own quieter distribution. Once both
configs are evaluated on a common five-seed footing, the V2 default
actually has a higher mean and one third the variance. The reason this
happened is that the Optuna objective was a single-seed final-avg-50 —
exactly the quantity that suffers from seed luck. A more honest objective
would have been median over k ≥ 3 seeds per trial, which would have
tripled the study wall time from 8 to 24 hours.

I am reporting this finding rather than the inflated single-seed number
because it is the truthful one. The implication is not that Bayesian HPO
failed; it is that the V2 default hyperparameters were already well
chosen for this task, and HPO confirmed that the remaining headroom is
small enough to be drowned out by seed noise. That is a legitimate result
in its own right, and it is the kind of result a more polished
publication would also have to report.

Plots: `V3/outputs/optuna/optimization_history.png`,
`param_importances.png`, `parallel_coordinate.png`,
`V3/outputs/robustness/learning_curves.png`, `final_avg50_box.png`.

---

## 3. What I could have done instead, and why I did not

### 3.1 Hyperband (alternative HPO method)

**One-line description.** A non-Bayesian HPO method built on successive
halving: start many short trials, kill the worst on a small budget, give
the survivors more budget, repeat. Useful when training time scales
strongly with budget and bad configurations can be detected early.

**Why this was a credible alternative.** Hyperband and TPE solve the
same problem from opposite directions — TPE picks better configurations
by modelling the success/failure densities; Hyperband picks better
configurations by aggressively killing the bad ones before they finish.
Either is a defensible choice for tuning DQN hyperparameters on
`highway-v0`. Hyperband can also outperform TPE on a fixed compute budget
when bad configurations are easy to spot early.

**Why TPE was the better choice for my setup.** Hyperband's whole
advantage rests on being able to detect bad configurations early. On
`highway-v0`, the first several thousand environment steps of any DQN
training are dominated by ε-greedy exploration, so early returns are
nearly random and look the same across good and bad configurations. A
configuration whose lr is too high will produce identical "still mostly
exploring" returns in steps 0–3,000 as a configuration that will end up
winning. Killing trials based on that signal would have thrown out
winners. TPE does not depend on early-stopping; it just sees the final
return of completed trials and learns from those. For this reason TPE
was a cleaner fit. If I later moved to a multi-seed-per-trial objective
(which makes each trial 3–5× more expensive) and a longer per-trial
budget, the calculus would shift and Hyperband — or, equivalently,
Optuna's `HyperbandPruner` layered onto the TPE study — would become
worth its complexity.

### 3.2 Model-based RL (PlaNet, Dreamer, TD-MPC, World Model)

**One-line description.** Learn a model of environment dynamics — most
often a recurrent latent state-space model trained on observation
reconstruction (PlaNet, Dreamer, World Model), or a deterministic
encoder-dynamics pair with a value head (TD-MPC) — and either plan
inside the learned model (CEM, iLQR, MPPI) or train a policy on imagined
trajectories. Drastically more sample efficient than model-free methods
on high-dimensional observations.

**Why this was a credible alternative.** The honest finding of my HPO
study is that the V2 DQN is already close to whatever ceiling the
hyperparameter space offers. To push past that ceiling I would need to
change the *algorithm*, not the *hyperparameters*. Model-based RL is the
post-V2 class of methods that does that — instead of tuning DQN better,
you learn a model of the environment and either plan with it (TD-MPC) or
train a policy in imagination (Dreamer, PlaNet). On vector observations
TD-MPC in particular has been shown competitive with model-free methods
at a fraction of the sample count, which would matter on a CPU-only
laptop.

**Why HPO was the better choice for V3.** Three reasons, in priority order.

1. **Observation geometry.** PlaNet, Dreamer, and the original World
   Model architecture are designed for pixel observations. The hard part
   they solve — compressing a high-dimensional image stream into a usable
   latent state — does not exist in `highway-v0`, where the environment
   already provides a 25-dim hand-engineered observation. The
   data-efficiency gains shrink dramatically on low-dim states while the
   implementation lift (encoder, decoder, dynamics model, reward model,
   imagination policy or MPC planner) stays large. TD-MPC is the one
   method in the family that does not require pixel reconstruction, but
   it is still substantially more implementation than tuning the agent I
   already have.

2. **Result quality versus effort, given the time budget.** V3 was a
   two-day deliverable on top of an already-working V2 baseline.
   Replacing the agent with a model-based method I have not implemented
   before — encoder, dynamics, planner, training loop — risks producing a
   half-finished implementation that does not actually beat the V2
   baseline I am trying to improve on, and would have left no time to
   honestly characterise whatever result it did produce. HPO is a smaller
   and faster intervention with a result that I can defend end-to-end:
   here is the search space, here are the trials, here is the winner,
   here is the multi-seed re-run that says the winner is within
   baseline noise.

3. **The result HPO produced is itself informative.** Knowing that V2's
   defaults are already near-optimal on `highway-v0` is a useful prior
   to bring into any future model-based attempt. Without that prior, a
   V4 model-based result of "matches V2 default" would be ambiguous —
   maybe the model-based method is bad, maybe DQN is just hard to beat
   on this task. With the V3 HPO result in hand, that ambiguity
   resolves: DQN really is well-tuned, and any model-based gain is
   measured against a strong baseline, not a weak one.

---

## 4. What I would do with more time

Listed in the order I would attempt them and intentionally drawn from
the class content.

1. **Re-run Optuna with a multi-seed objective** (median over k=3 seeds
   per trial). The obvious correction to the seed-luck issue identified
   by the robustness study. Triples wall time but produces a winner that
   does not need a separate robustness re-run.
2. **Add Hyperband pruning to the multi-seed Optuna study.** Once per-trial
   budget grows, killing bad configurations on the cheap budget becomes
   worth the complexity. A small Optuna API change
   (`HyperbandPruner` in place of the default pruner).
3. **A second, narrower Optuna pass** anchored at trial 9's neighborhood
   (small ranges around `lr=5.65e-4` and `gamma=0.998`) to test whether
   that region genuinely contains better configurations or is all within
   V2-default noise.
4. **Re-introduce PER hyperparameters** (`alpha`, `beta` annealing
   schedule) into the search space. Requires a small change to
   `V2/src/agents/deep/dqn.py` so the agent can pass these through to
   the replay buffer, currently hardcoded to defaults α=0.6, β=0.4.
5. **TD-MPC on top of the tuned config.** This is the natural V4
   experiment. With a well-tuned model-free baseline in hand, a TD-MPC
   re-implementation has a clean reference to be measured against.

---

## 5. Reflection

The most honest thing I can say about V3 is that I picked the post-V2
topic best matched to my environment, ran it carefully, and reported the
result the analysis actually produced — including the result I would
rather not have reported. The single-seed Optuna winner (+29.57) looked
like a clean ~7% improvement over the V2 baseline; the multi-seed
robustness re-run showed that improvement was within noise. I wrote that
finding into the repo rather than burying it because the V3 spec rewards
critical thought and the truthful finding is the one a careful reader
would catch anyway.

Of the two credible alternatives — Hyperband and model-based RL — neither
was a better V3 investment than Bayesian HPO via TPE. Hyperband loses
its main advantage on a problem where early returns do not predict final
returns. Model-based RL targets a difficulty (high-dimensional pixel
observations) that my environment has already resolved. HPO targeted the
actual gap in my existing setup, produced a complete and honest result
in the available time, and gave me a defensible baseline to measure any
future model-based attempt against.

---

## 6. Pointers into the repository

- [`V3/decisions.md`](decisions.md) — same content as this report, kept
  as a working document alongside the code.
- [`V3/technical-challenges.md`](technical-challenges.md) — friction
  encountered during V3 (broken venv, PowerShell stdout buffering, the
  concurrent-runs incident).
- [`V3/citations.md`](citations.md) — collaborators, AI assistants, and
  references.
- [`V3/run_optuna.py`](run_optuna.py) — Bayesian HPO study.
- [`V3/run_robustness.py`](run_robustness.py) — multi-seed comparison.
- [`V3/outputs/optuna/`](outputs/optuna/) — Optuna artifacts.
- [`V3/outputs/robustness/`](outputs/robustness/) — robustness artifacts.
- [`V2/README.md`](../V2/README.md) — the V2 benchmark this report
  builds on.
- [`README.md`](../README.md) — overall project overview.
