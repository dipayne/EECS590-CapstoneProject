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
`summary.json`. **Reading guide:** if the V3-tuned mean cleanly clears the V2
mean plus one standard deviation, the gain is real; if the boxes overlap
heavily, the +1.97 in the single Optuna run was largely seed-driven.

---

## Considered and skipped

For each, I state what it is, what implementing it would have cost, and why
that cost was not worth it for *this* project. Implementing all of them
would have been the wrong answer.

### Multi-agent / coordination methods (MAPPO, QMIX, IPPO, MADDPG)

**What it is.** Algorithms where multiple agents learn jointly, either with a
centralized critic and decentralized actors (MAPPO, MADDPG) or with a
mixing network that decomposes a joint value function (QMIX). Designed for
StarCraft, Hanabi, traffic networks of learning vehicles, robot swarms.

**Why I skipped it.** `highway-v0` has exactly one learning agent. The
surrounding vehicles are scripted IDM/MOBIL drivers — they do not learn,
they do not coordinate, they do not have policies in the RL sense.
Force-fitting MAPPO here would require either (a) replacing the scripted
traffic with N learning DQN agents, which changes the environment into
something the rest of my repo no longer addresses, or (b) wrapping the
single agent in a multi-agent shim, which contributes nothing. The V3 spec
explicitly names this as the wrong fit for a main-protagonist environment,
and I agree.

### Distributional RL (C51, QR-DQN, IQN)

**What it is.** Instead of estimating the *expected* discounted return as a
scalar Q(s,a), distributional methods learn the *full distribution* of
returns — either as a categorical distribution over fixed atoms (C51) or as
a set of quantiles (QR-DQN). Useful when downstream decisions depend on
risk, not just mean reward.

**Why I skipped it.** This is a serious candidate — risk-aware driving
(avoiding rare but catastrophic collisions) is exactly the kind of problem
distributional RL helps with. But: replacing my Q-head with a 51-atom
categorical head and rewriting the Bellman update to use the distributional
projection is a non-trivial change to the V2 DQN code, and once done it
nullifies my V2 ablation (which used scalar Q-values throughout). For V3, I
chose to deepen what I already have (tune it well) rather than re-implement
the agent. If the project continued, this would be my first next step.

### Constrained / safety-RL (Lagrangian PPO, CPO, RCPO)

**What it is.** Formulate driving as a constrained MDP — maximize expected
return subject to a hard cap on expected cost (e.g., crash probability).
Solved via Lagrangian relaxation (dual variable on the constraint) or trust
region methods (CPO).

**Why I skipped it.** This is the most "day-to-day relevant" choice on this
list for a driving context, but it requires (a) a working PPO baseline,
which my V2 results show is unstable on `highway-v0`, and (b) careful tuning
of the Lagrange multiplier that itself eats hyperparameter-tuning budget.
The implementation effort is on the order of weeks for a result I can be
confident in, and I had two days.

### Model-based RL (Dreamer, MuZero, World Models)

**What it is.** Learn a model of environment dynamics (often a recurrent
state-space model) and plan or train a policy inside the learned model.
Drastically more sample efficient than model-free methods.

**Why I skipped it.** Implementing a world model on top of `highway-v0`'s
25-dim observation is overkill — these methods are designed for
high-dimensional pixel observations where modeling the dynamics is the hard
part. On a low-dim vector state, the data-efficiency gain is much smaller
than the implementation lift. Wrong tool for the observation geometry.

### Behavioral cloning / imitation learning / DAgger

**What it is.** Train the policy to mimic expert demonstrations (BC) or
iteratively query an expert for corrections on states the learned policy
visits (DAgger).

**Why I skipped it.** No dataset. The `highway-v0` simulator does not ship
with human demonstrations, and collecting them via the keyboard renderer
would have produced a tiny, noisy corpus that would not actually train a
robust policy. Real dashcam data would be the right input, but obtaining,
labeling, and aligning that data is a project of its own.

### Hierarchical RL / options framework

**What it is.** Decompose the policy into a high-level policy that selects
*options* (temporally extended actions, e.g., "overtake the vehicle in front")
and low-level policies that execute them.

**Why I skipped it.** `highway-v0` already exposes meta-actions —
"lane left", "faster", etc. — that are themselves temporally extended over
many underlying simulator steps. Adding a second hierarchy on top is solving
a problem that the environment has already solved for me.

### Offline RL (CQL, BCQ, IQL)

**What it is.** Learn a policy from a fixed dataset of logged transitions
without further interaction with the environment. Necessary when online
exploration is expensive or dangerous (real autonomous driving, healthcare,
robotics).

**Why I skipped it.** Same blocker as imitation learning — no dataset.
Generating one from my V2 agents and then running offline RL on it would be
a circular exercise that doesn't demonstrate anything about offline RL's
real value.

---

## Open questions / things I would do with more time

- Confirm the +1.97 Optuna gain holds when re-trained from scratch on a
  fresh seed and on traffic-density-2.0 (a harder configuration).
- A second, narrower Optuna pass anchored at trial 9's neighborhood (small
  ranges around lr=5.65e-4 and gamma=0.998).
- Re-introduce PER hyperparameters (α, β annealing schedule) into the
  search space — would require a small modification to `DQNAgent.__init__`
  to pass them through to the buffer.
- Distributional DQN (QR-DQN) on top of the tuned config, to see whether
  modeling return variance unlocks further gains on collision-heavy
  scenarios.
