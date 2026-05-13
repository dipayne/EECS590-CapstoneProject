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

V3 builds on that result. The instructor asked for a deliberate choice of
post-V2 class content to apply, with clear justification of what was kept
out. This report is that justification.

---

## 2. What I implemented in V3

### Bayesian hyperparameter optimization (Optuna, TPE sampler)

**What it is.** A method for searching the hyperparameter space of a
learning algorithm by treating each training run as an expensive experiment
and using probabilistic models to pick the next configuration. The
Tree-structured Parzen Estimator (TPE) maintains two density models — one
over "good" configurations and one over "poor" configurations — and on each
new trial proposes the configuration that maximizes their ratio. Compared
with grid search or random search, TPE focuses budget on promising regions
after a small warm-up phase.

**Why this is the right tool for my project.** `highway-v0` has a single
learning agent. There is no team to coordinate, no swarm to align, no
opponent to model. The thing that was clearly underdeveloped in my V2 work
was that every DQN hyperparameter — learning rate, discount factor,
target-network update frequency, ε-decay schedule — was either a textbook
default or a guess. Bayesian HPO is the directly applicable response to
that gap. It is also a method I will use in every future machine-learning
project regardless of domain, so the effort transfers beyond this class.

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

Each trial trained the agent for 20,000 environment steps. Total CPU wall
time: 8 hours 18 minutes. The study is SQLite-backed so it can be resumed,
queried, and re-analyzed without re-training.

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
+27.60 was a low draw from its own quieter distribution. Once both configs
are evaluated on a common five-seed footing, the V2 default actually has a
higher mean and one third the variance. The reason this happened is that
the Optuna objective was a single-seed final-avg-50 — exactly the
quantity that suffers from seed luck. A more honest objective would have
been median over k ≥ 3 seeds per trial, which would have tripled the study
wall time from 8 to 24 hours.

I am reporting this finding rather than the inflated single-seed number
because it is the truthful one. The implication is not that Bayesian HPO
failed; it is that the V2 default hyperparameters were already well chosen
for this task, and HPO confirmed that the remaining headroom is small
enough to be drowned out by seed noise. That is a legitimate result in
its own right, and it is the kind of result a more polished publication
would also have to report.

Plots: `V3/outputs/optuna/optimization_history.png`,
`param_importances.png`, `parallel_coordinate.png`,
`V3/outputs/robustness/learning_curves.png`, `final_avg50_box.png`.

---

## 3. What I considered and chose not to implement

For each of the following, I describe the algorithm in one line, sketch the
strongest case for using it on `highway-v0`, and explain why I judged
Bayesian HPO a better V3 investment for this project.

### 3.1 Multi-agent / coordination methods (MAPPO, QMIX, IPPO, MADDPG)

**One-line description.** Algorithms that train multiple agents jointly,
either via a centralized critic with decentralized actors (MAPPO, MADDPG)
or via a mixing network that decomposes a joint value function (QMIX).

**Best case for using it here.** If `highway-v0` were configured so that
several of the vehicles on the road were learning agents — for example a
small fleet of self-driving cars trying to merge cooperatively — these
algorithms would let them learn coordinated lane-change protocols rather
than each agent treating the others as scripted obstacles.

**Why HPO was more appropriate.** `highway-v0` has exactly one learning
agent. Surrounding traffic is scripted IDM/MOBIL drivers; they do not have
policies in the RL sense. To meaningfully use a multi-agent method I would
have had to either replace the scripted traffic with several learning DQN
agents — which would change the environment my V1 and V2 work no longer
addresses — or wrap the single agent in a multi-agent shim that adds
nothing. The instructor's V3 description specifically points out that a
main-protagonist environment does not benefit from swarm/coordination
methods, and I agree. HPO targets the real source of headroom in my
existing single-agent setup; multi-agent methods would target a problem
that does not exist in my environment.

### 3.2 Distributional RL (C51, QR-DQN, IQN)

**One-line description.** Instead of estimating the expected return Q(s,a)
as a scalar, learn the full distribution of returns, either as a
categorical distribution over fixed atoms (C51) or as a set of quantiles
(QR-DQN). Useful when downstream decisions depend on risk, not just mean.

**Best case for using it here.** Driving is a risk-sensitive task. Two
policies with the same mean return can have very different collision
probabilities. A distributional agent could optimize for, e.g., a lower
tail of the return distribution rather than the mean, which corresponds
directly to "be more cautious in dangerous situations."

**Why HPO was more appropriate.** Distributional RL would require
replacing the Q-head in my V2 DQN with a 51-atom categorical head (C51) or
a quantile head (QR-DQN), then rewriting the Bellman target to use the
distributional projection or quantile-Huber loss. Once done, my V2
ablation results (which compared scalar-Q variants) would no longer be
directly comparable to V3. The work would also have eaten most of the V3
time budget on its own, leaving no room to tune anything. I chose to
deepen the work already on the table rather than start a parallel branch
that would have invalidated parts of it. Distributional RL is my first
candidate for a hypothetical V4.

### 3.3 Constrained / safety RL (Lagrangian PPO, CPO, RCPO)

**One-line description.** Formulate the problem as a constrained MDP —
maximize expected return subject to a hard cap on expected cost (e.g.,
crash probability). Solved via Lagrangian relaxation (dual variable on the
constraint) or trust-region methods (CPO).

**Best case for using it here.** This is the most "real-world driving"
candidate on the list. A safety constraint on collision probability is
exactly what a deployed driving policy needs, and the academic literature
on safe RL is built around problems that look very similar to
`highway-v0`.

**Why HPO was more appropriate.** Constrained RL methods are most mature
when built on top of a working PPO baseline, and my V2 results show that
PPO is one of the less stable agents on `highway-v0`. To do safety RL
justice I would have needed first to stabilize PPO, then add the
constraint and tune its Lagrange multiplier separately. The Lagrangian
itself is an additional hyperparameter that controls a delicate trade-off
between reward and constraint satisfaction, so I would also have ended up
back at hyperparameter tuning. The total effort is on the order of weeks
of work for results I could trust. With two days and a CPU-only laptop,
producing a clean, defensible safety-RL result was not realistic. HPO
gave me a complete, honestly reported result in the time available.

### 3.4 Model-based RL (Dreamer, MuZero, World Models)

**One-line description.** Learn a model of the environment dynamics
(commonly a recurrent state-space model trained on observation
reconstruction) and either plan inside it or train a policy on imagined
trajectories. Drastically more sample efficient than model-free methods
on pixel observations.

**Best case for using it here.** If `highway-v0` were rendered as raw
pixels — a 64×64 grayscale top-down view, for instance — then learning a
compact latent dynamics model and rolling out imagined trajectories would
be the canonical way to make the problem tractable.

**Why HPO was more appropriate.** `highway-v0` exposes a 25-dimensional
hand-engineered observation. The hard part that model-based methods solve
— compressing high-dimensional pixel observations into a useful latent
state — is already done for me by the environment's observation function.
On vector states the data-efficiency gains shrink dramatically while the
implementation lift (encoder, decoder, dynamics model, reward model,
imagination policy) stays large. The mismatch between method and
observation geometry made this the wrong tool. HPO operates directly on
the existing observation and produces a concrete result; model-based RL
would have spent most of its complexity on a problem I do not have.

### 3.5 Behavioral cloning / imitation learning / DAgger

**One-line description.** Train the policy to mimic expert demonstrations
(BC) or iteratively query an expert for corrections on states the learned
policy visits (DAgger).

**Best case for using it here.** If I had a dataset of human dashcam
driving, BC + DAgger would be the most direct way to transfer the prior
of "what humans actually do in this situation" into the policy, and would
likely produce more naturalistic lane-changing behavior than reward-driven
RL alone.

**Why HPO was more appropriate.** No dataset. `highway-v0` does not ship
with human demonstrations, and collecting them via the keyboard renderer
would have produced a tiny, noisy corpus that is closer to "me poking at
keys" than to expert driving. Real dashcam data — the input that would
actually justify imitation learning here — is its own collection,
labelling, and alignment project. Without the input data, the method
cannot produce a meaningful result. HPO uses only data the simulator
already produces.

### 3.6 Hierarchical RL / options framework

**One-line description.** Decompose the policy into a high-level policy
that selects options (temporally extended actions, e.g., "overtake the
vehicle ahead") and low-level policies that execute them.

**Best case for using it here.** If the action space were low-level
(steering angle, throttle, brake) and the agent had to learn for itself
that "overtake" is a useful temporal abstraction, hierarchical RL would
be the way to bake that structure into the policy.

**Why HPO was more appropriate.** `highway-v0`'s discrete action space is
*already* the high-level option space — "faster", "slower", "lane left",
"lane right", "idle" each spans many underlying simulator steps. Adding a
second hierarchy above options that already exist is solving a problem
the environment has already solved for me. HPO works directly on the
problem the environment does pose.

### 3.7 Offline RL (CQL, BCQ, IQL)

**One-line description.** Learn a policy from a fixed dataset of logged
transitions without further interaction with the environment. Necessary
when online exploration is unsafe or expensive (real autonomous driving,
healthcare, robotics).

**Best case for using it here.** If I were deploying onto a real vehicle
and could not afford the cost of exploration, offline RL would let me
learn from logged driving data rather than from live experience.

**Why HPO was more appropriate.** Same blocker as imitation learning — no
dataset. I would have had to generate one from my V2 agents and then run
offline RL on it, which is a circular exercise that demonstrates the
mechanics of CQL but not its actual value (avoiding live exploration).
HPO does not require any input beyond what the simulator provides.

---

## 4. What I would do with more time

These are the items I would tackle in a V4 if the project continued. They
are listed in the order I would attempt them.

1. **Re-run Optuna with a multi-seed objective** (median over k=3 seeds
   per trial). This is the obvious correction to the seed-luck issue
   identified by the robustness study. The cost is a 3× wall-time
   increase, but the result is a winner I could trust without a separate
   robustness re-run.
2. **A second, narrower Optuna pass** anchored at trial 9's neighborhood,
   sampling small ranges around `lr=5.65e-4` and `gamma=0.998` rather than
   the broad ranges of the first pass. This would test whether the
   tuned-region of the search space genuinely contains better
   configurations or whether all of it is within V2-default noise.
3. **Re-introduce PER hyperparameters** (`alpha`, `beta` annealing
   schedule) into the search space. This would require a small change to
   `V2/src/agents/deep/dqn.py` so the agent can pass these through to the
   replay buffer, which is currently hardcoded to defaults α=0.6, β=0.4.
4. **Distributional DQN (QR-DQN)** on top of the best HPO config, to
   measure whether modeling return *variance* unlocks further gains on
   collision-heavy traffic densities. This is the natural next step beyond
   HPO because it changes what the agent learns, not just how it is tuned.

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

For an environment like `highway-v0` — single agent, vector observations,
no demonstrations, no safety constraints in the original reward — Bayesian
HPO is the right method on the menu of post-V2 topics. The other
candidates either solve problems my environment does not have
(multi-agent, hierarchical) or require infrastructure my project does not
include (datasets, pixel observations, formal safety specifications). The
"right method for this environment" was the choice the V3 spec was asking
me to identify, and that was the choice I made.

---

## 6. Pointers into the repository

- [`V3/decisions.md`](decisions.md) — same content as this report, kept as
  a working document alongside the code.
- [`V3/technical-challenges.md`](technical-challenges.md) — friction
  encountered during V3 (broken venv, PowerShell stdout buffering, the
  concurrent-runs incident).
- [`V3/citations.md`](citations.md) — collaborators, AI assistants, and
  references.
- [`V3/run_optuna.py`](run_optuna.py) — Bayesian HPO study.
- [`V3/run_robustness.py`](run_robustness.py) — multi-seed comparison.
- [`V3/outputs/optuna/`](outputs/optuna/) — Optuna artifacts.
- [`V3/outputs/robustness/`](outputs/robustness/) — robustness artifacts.
- [`V2/README.md`](../V2/README.md) — the V2 benchmark this report builds
  on.
- [`README.md`](../README.md) — overall project overview.
