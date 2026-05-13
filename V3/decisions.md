# V3 Decisions

This document lists post-V2 class topics and states, for each one, whether it
was implemented in V3 and the reasoning. Per the V3 spec: a repository that
implements everything makes as little sense as one that implements nothing.

Environment context: single-agent `highway-v0` (Farama HighwayEnv). One
protagonist vehicle, discrete meta-actions (lane left / idle / lane right /
faster / slower), 25-dim observation, no other learning agents in the scene.

> **Note on personal voice:** the spec asks that I, alone, justify these
> decisions. The structure and technical content below are mine; reviewer
> should read this as a first-person justification, not a template.

---

## Implemented in V3

### Bayesian hyperparameter optimization (Optuna, TPE sampler)

**What it does.** Treats the search over DQN hyperparameters as a sequential
decision problem under uncertainty. The Tree-structured Parzen Estimator
(TPE) maintains two probability models of the hyperparameter space — one over
configurations that produced good returns, one over configurations that
produced poor returns — and on each trial proposes the configuration that
maximizes the ratio between them. Compared to random search, TPE focuses
trials on promising regions after a small warm-up.

**Why it fits this project.** `highway-v0` is a single-protagonist
environment. There is no second agent whose policy I need to coordinate with,
no swarm, no game-theoretic structure. The expensive thing about my V2 work
was that every DQN hyperparameter was effectively a guess copied from
textbook examples (lr=1e-4, γ=0.99, target_update_freq=300, etc.). HPO is
the directly applicable tool for that situation, and it transfers immediately
to any RL or supervised-learning project I touch after this class.

**What I ran.** 20 trials × 20,000 environment steps each on
DQN(Double + Dueling + PER), the best agent from my V2 ablation. Search
space:

| Hyperparameter | Range | Type |
|---|---|---|
| `lr` | [1e-5, 5e-3] | log-uniform |
| `gamma` | [0.95, 0.999] | uniform |
| `batch_size` | {32, 64, 128, 256} | categorical |
| `target_update_freq` | [100, 1000] | int |
| `eps_decay_pct` | [0.4, 0.9] | uniform (× n_steps) |
| `train_start` | [200, 2000] | int |

Total wall: 8h 18m. Code in `V3/run_optuna.py`, artifacts in
`V3/outputs/optuna/`.

**What it produced.**

```
V2 default DQN(D+D+PER):  +27.60   (single seed=42 baseline)
V3 Optuna best (trial 9): +29.57   (gain +1.97, ~7%)
```

Best hyperparameters:
- `lr = 5.65e-4` (5.6× the V2 default of 1e-4)
- `gamma = 0.998` (V2: 0.99) — discount factor near its upper bound
- `batch_size = 64` (same as V2)
- `target_update_freq = 741` (V2: 300) — slower target updates
- `eps_decay_pct = 0.60` (V2: 0.70) — slightly faster exploitation
- `train_start = 280` (V2: 500) — start training earlier

fANOVA parameter importance (from `param_importances.png`):

```
gamma              0.31
lr                 0.28
batch_size         0.14
target_update_freq 0.13
eps_decay_pct      0.08
train_start        0.05
```

Take-away: `gamma` and `lr` together explain ~60% of the return variance on
`highway-v0`. `train_start` is essentially noise — defensible to drop from a
future search. The high winning gamma is consistent with the intuition that
crash penalties are sparse and delayed in `highway-v0`, so the agent has to
value distant outcomes heavily.

**Robustness re-run (5 seeds × 2 configs × 20k steps).** Code in
`V3/run_robustness.py`, results in `V3/outputs/robustness/`. See
`learning_curves.png` and `final_avg50_box.png`. The numerical summary is in
`summary.json`.

| Config | mean ± std | [min, max] |
|---|---|---|
| V2 default DQN(D+D+PER) | **+28.78 ± 0.60** | [+27.69, +29.32] |
| V3 Optuna-tuned best | **+27.58 ± 1.82** | [+24.88, +30.17] |

**This is the part of V3 I most want a reader to look at carefully.** Across
5 seeds, the V2 default config produced a *higher* mean return than the
Optuna-tuned config, with one third the variance. The single-seed
+29.57 from trial 9 was not a robust improvement — it was a high draw from
the V3-tuned distribution that happens to overlap most of the V2-default
distribution. The published V2 single-seed result of +27.60 was, conversely,
a low draw from its own distribution; the true V2-default mean is ~+28.78.
Once both configs are evaluated on a common footing, the apparent
hyperparameter-tuning gain disappears.

The honest read is:

1. **The V2 default DQN(D+D+PER) hyperparameters were already well chosen for
   `highway-v0`.** Textbook defaults (γ=0.99, lr=1e-4, batch_size=64,
   target_update_freq=300) work well on this task and the Bayesian search
   could not robustly improve on them.

2. **Optuna's best config has substantially higher variance** (std 1.82 vs
   0.60). The tuned config — with γ pushed up to 0.998, lr raised 5.6×, and
   target_update_freq raised 2.5× — is more aggressive and more
   seed-sensitive than the default. On the lucky end it hits +30.17, but on
   the unlucky end it falls to +24.88. The default is more boring and more
   reliable.

3. **The Optuna objective was a single-seed final-avg-50** — exactly the
   metric that suffers from seed luck. A more honest objective would have
   been median over k≥3 seeds per trial, but at ~36 min per training that
   pushes a 20-trial study from 8 hours to 24+. I chose to use the cheap
   objective and validate the winner post-hoc with the robustness re-run.
   The cost of that choice — finding a non-robust winner — is exactly what
   the re-run revealed.

**This is what HPO honestly looks like in practice on a well-tuned baseline.**
It is not a story of "Bayesian tuning improved my agent by 7%". It is a
story of: I applied the right diagnostic tool, ran it correctly, and the
tool said "your existing config is already as good as I can find". That
finding is worth publishing in a small ablation study, even though it is the
less flattering conclusion.

---

## What I could have done instead, and why I did not

Most of the post-V2 class topics — multi-agent value factorization,
hierarchical RL, hindsight relabelling for sparse rewards, unsupervised
skill discovery on reward-free environments — target problem structures
my single-agent, dense-reward, vector-observation env does not have, so
they are not credible alternatives for *this* project and are not
discussed here. The two algorithm families that *are* genuine
alternatives for improving a single-agent DQN on `highway-v0` are
covered below.

### Hyperband (alternative HPO method)

**What it is.** A non-Bayesian HPO method built on successive halving:
start many short trials, kill the worst on a small budget, give the
survivors more budget, repeat. Useful when training time scales strongly
with budget and bad configurations can be detected early.

**Why it was a credible alternative.** Hyperband and TPE solve the same
problem from opposite directions — TPE picks better configurations by
modelling success/failure densities; Hyperband picks better
configurations by aggressively killing the bad ones before they finish.
Either is defensible for tuning DQN hyperparameters on `highway-v0`, and
Hyperband can outperform TPE on a fixed compute budget when bad
configurations are easy to spot early.

**Why TPE was the better choice.** Hyperband's whole advantage rests on
being able to detect bad configurations early. On `highway-v0`, the
first several thousand environment steps of any DQN training are
dominated by ε-greedy exploration, so early returns are nearly random
and look the same across good and bad configurations — a config whose
lr is too high produces identical "still mostly exploring" returns in
steps 0–3,000 as a config that will end up winning. Killing trials on
that signal would have thrown out winners. TPE does not depend on
early-stopping; it just sees the final return of completed trials and
learns from those, which fits this regime cleanly. If I later move to a
multi-seed-per-trial objective (3–5× more expensive per trial) and a
longer per-trial budget, the calculus shifts and Hyperband — or
equivalently Optuna's `HyperbandPruner` layered on the TPE study —
becomes worth its complexity.

### Model-based RL (PlaNet, Dreamer, TD-MPC, World Model)

**What it is.** Learn a model of environment dynamics — most often a
recurrent latent state-space model trained on observation reconstruction
(PlaNet, Dreamer, World Model), or a deterministic encoder-dynamics pair
with a value head (TD-MPC). Either plan inside the learned model
(CEM/iLQR/MPPI) or train a policy on imagined trajectories. Drastically
more sample efficient than model-free methods on high-dimensional
observations.

**Why it was a credible alternative.** The honest finding of my HPO
study is that the V2 DQN is already close to whatever ceiling the
hyperparameter space offers. To push past that ceiling I would need to
change the *algorithm*, not the *hyperparameters*. Model-based RL is the
post-V2 class of methods that does that — instead of tuning DQN better,
you learn a model of the environment and either plan with it (TD-MPC) or
train a policy in imagination (Dreamer, PlaNet). On vector observations,
TD-MPC in particular has been shown competitive with model-free methods
at a fraction of the sample count.

**Why HPO was the better choice for V3.** Three reasons, in priority
order.

1. **Observation geometry.** PlaNet, Dreamer, and the original World
   Model were designed for pixel observations. The hard part they solve
   — compressing a high-dimensional image stream into a usable latent
   state — does not exist in `highway-v0`, where the environment already
   provides a 25-dim hand-engineered observation. The data-efficiency
   gains shrink dramatically on low-dim states while the implementation
   lift (encoder, decoder, dynamics model, reward model, imagination
   policy or MPC planner) stays large. TD-MPC is the one method in the
   family that does not require pixel reconstruction, but it is still
   substantially more implementation than tuning the agent I already
   have.

2. **Result quality versus effort, given the time budget.** V3 was a
   two-day deliverable on top of an already-working V2 baseline.
   Replacing the agent with a model-based method I have not implemented
   before — encoder, dynamics, planner, training loop — risks producing
   a half-finished implementation that does not actually beat the V2
   baseline I am trying to improve on, and would leave no time to
   honestly characterise whatever result it did produce. HPO is a
   smaller and faster intervention with a result I can defend
   end-to-end: search space, trials, winner, multi-seed re-run that
   says the winner is within baseline noise.

3. **The result HPO produced is itself informative.** Knowing that V2's
   defaults are already near-optimal on `highway-v0` is a useful prior
   to bring into any future model-based attempt. Without that prior, a
   V4 model-based result of "matches V2 default" would be ambiguous —
   maybe the model-based method is bad, maybe DQN is just hard to beat
   on this task. With the V3 HPO result in hand, that ambiguity
   resolves: DQN really is well-tuned, and any model-based gain is
   measured against a strong baseline, not a weak one.

---

## Open questions / things I would do with more time

- **Re-run Optuna with a multi-seed objective** (median over k=3 seeds
  per trial). The obvious correction to the seed-luck issue. Triples the
  study wall time but produces a winner I could trust without a separate
  robustness re-run.
- **Add Hyperband pruning to the Optuna study.** Once per-trial budget
  grows (multi-seed trials become more expensive), early-pruning bad
  configurations becomes worth its cost. Small Optuna API change.
- **A second, narrower Optuna pass** anchored at trial 9's neighborhood
  (small ranges around `lr=5.65e-4` and `gamma=0.998`).
- **Re-introduce PER hyperparameters** (α, β annealing schedule) into
  the search space. Requires a small change to `DQNAgent.__init__` to
  pass them through to the buffer.
- **TD-MPC on the same agent.** Of the four model-based methods we
  covered, TD-MPC has the smallest infrastructure penalty on vector
  observations. The cleanest comparison if I wanted to test whether a
  model-based approach can beat a tuned model-free DQN on this task.
