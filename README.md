# EECS 590 Capstone Project — Reinforcement Learning on Highway-env

A comprehensive reinforcement learning research framework spanning two versions:
**V1** (model-based DP on a custom logistics grid) and **V2** (13+ algorithms benchmarked head-to-head on the `highway-v0` autonomous driving environment).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Version 1 — Dynamic Programming (Logistics Grid)](#version-1--dynamic-programming-logistics-grid)
- [Version 2 — Full RL Algorithm Benchmark (Highway-env)](#version-2--full-rl-algorithm-benchmark-highway-env)
  - [Environment](#environment)
  - [Algorithm Survey](#algorithm-survey)
  - [Architecture](#architecture)
  - [Results](#results)
  - [DQN Ablation Study](#dqn-ablation-study)
  - [Critical Analysis](#critical-analysis)
  - [Saliency / Interpretability](#saliency--interpretability)
  - [Checkpoints and Logging](#checkpoints-and-logging)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Outputs](#outputs)
- [Citations](#citations)

---

## Project Overview

This project investigates reinforcement learning across the full spectrum of classical and modern algorithms, grounded in the `highway-v0` autonomous driving environment from [highway-env](https://github.com/Farama-Foundation/HighwayEnv). The work progresses from:

1. **V1**: Exact dynamic programming (Policy Iteration) on a small, fully-specified logistics grid MDP where the transition model is known.
2. **V2**: Model-free learning at scale — 13 algorithms (7 classical tabular, 6 deep RL) trained and compared under identical conditions on a continuous, partially-observable driving task.

The unifying question: *how much does access to a model, function approximation, and modern neural architectures matter in practice?*

---

## Version 1 — Dynamic Programming (Logistics Grid)

### Environment

A **6×6 Logistics Grid** where an agent navigates from a depot to a customer location while avoiding blocked cells.

```
D → → → → →
↓ ↓ █ ↓ █ ↓
↓ ↓ █ ↓ ↓ ↓
↓ ↓ █ ↓ ↓ ↓
↓ ↓ █ █ ↓ ↓
↓ ↓ ↓ ↓ ↓ G
```
*(D = Depot, G = Goal/Customer, █ = Blocked cell)*

| Parameter        | Value                                           |
|------------------|-------------------------------------------------|
| Grid size        | 6 × 6                                           |
| States           | 30 valid (blocked cells excluded)               |
| Actions          | 5 (UP, RIGHT, DOWN, LEFT, WAIT)                 |
| Discount (gamma) | 0.95                                            |
| Slip probability | 0.10 (stochastic transitions)                   |
| Reward           | −1 per step, +20 at goal                        |

### Methods

**Policy Iteration** (`src/dp/policy_iteration.py`)
- Policy Evaluation: iterative Bellman expectation until convergence (theta = 1e-8)
- Policy Improvement: greedy from both V(s) and Q(s,a)
- Alternates until policy stabilizes

**Model Learning** (`src/mdp/model_learning.py`)
- Bootstraps world-model beliefs from random trajectory sampling
- Estimates P[s][a][s'] and R[s][a][s'] via frequency counts
- Supports persistent incremental belief updates (Bayesian-style)

### V1 Outputs

| File | Description |
|------|-------------|
| `outputs/plots/policy_map.png` | Arrow-grid visualization of learned policy |
| `outputs/plots/value_heatmap.png` | Color heatmap of V(s) |
| `outputs/policies/best_policy.json` | Serialized policy kernel |
| `outputs/values/V.npy` | State value function array |
| `outputs/qvalues/Q.npy` | Q-value table |

---

## Version 2 — Full RL Algorithm Benchmark (Highway-env)

### Environment

**`highway-v0`** from Gymnasium / highway-env:
- 4-lane highway, ego vehicle surrounded by traffic
- **Observation**: kinematics of 5 nearby vehicles × 5 features = **25-dim flat vector**
- **Actions**: Discrete(5) — IDLE, LANE-RIGHT, LANE-LEFT, FASTER, SLOWER
- **Reward**: speed reward + lane-keeping bonus − collision penalty (discounted)
- **Tabular abstraction** (classical agents only): 135 states = 5 lanes × 3 speed bins × 3 front-dist bins × 3 rear-dist bins

### Algorithm Survey

#### Implemented

| Category | Algorithm | Notes |
|----------|-----------|-------|
| Classical tabular | Monte Carlo Control | First-visit, epsilon-greedy |
| Classical tabular | TD(lambda) | Backward eligibility traces |
| Classical tabular | SARSA(lambda) | On-policy, backward traces |
| Classical tabular | Q-Learning | Off-policy tabular |
| Classical tabular | Q(lambda) | Watkins' trace-cutting |
| Classical tabular | Linear FA | SARSA with one-hot tile features |
| Deep RL | DQN | Mnih et al. 2015 |
| Deep RL | Double DQN | van Hasselt et al. 2016 |
| Deep RL | Dueling DQN | Wang et al. 2016 |
| Deep RL | DQN (D+D+PER) | Double + Dueling + Prioritized ER |
| Deep RL | REINFORCE | Williams 1992, with value baseline |
| Deep RL | A2C | Synchronous Advantage Actor-Critic |
| Deep RL | PPO | Schulman et al. 2017, with GAE |

#### Not Implemented (and Why)

| Algorithm | Reason |
|-----------|--------|
| **DDPG / TD3 / SAC** | Require continuous action spaces; highway-v0 is Discrete(5). These algorithms output real-valued torques/velocities and cannot be applied without fundamentally changing the environment. |
| **A3C** | Requires asynchronous multiprocessing with shared memory across workers. Technically infeasible in a single-process Python environment on Windows; A2C (the synchronous equivalent) captures the same algorithmic ideas. |
| **TRPO** | Replaced in practice by PPO, which achieves the same trust-region constraint via clipping at a fraction of the implementation complexity. TRPO requires computing the Fisher information matrix inverse, adding significant engineering overhead with no performance benefit over PPO here. |
| **Value Iteration** | Requires a known transition model P(s'|s,a). In highway-v0, the dynamics are not analytically available — the environment is a black box simulator. This is covered in V1 on the known logistics grid MDP. |

### Architecture

```
V2/
├── src/
│   ├── agents/
│   │   ├── base.py                    # Abstract BaseAgent interface
│   │   ├── classical/
│   │   │   ├── monte_carlo.py
│   │   │   ├── td.py                  # TD(0), TD(n), TD(lambda)
│   │   │   ├── sarsa.py               # SARSA, SARSA(lambda)
│   │   │   ├── q_learning.py          # Q-Learning, Double Q-Learning
│   │   │   ├── q_lambda.py            # Q(lambda) with Watkins' traces
│   │   │   └── linear_fa.py           # Linear function approximation
│   │   └── deep/
│   │       ├── dqn.py                 # DQN / Double / Dueling / D+D+PER
│   │       ├── reinforce.py           # REINFORCE with baseline
│   │       ├── a2c.py                 # Synchronous A2C
│   │       └── ppo.py                 # PPO with GAE
│   ├── networks/
│   │   ├── mlp.py                     # Multi-layer perceptron backbone
│   │   └── dueling.py                 # Dueling stream architecture
│   ├── replay_buffer/
│   │   └── buffer.py                  # Standard + Prioritized (SumTree) replay
│   ├── checkpoints/
│   │   └── manager.py                 # Best-checkpoint + periodic saving
│   ├── saliency/
│   │   ├── gradient_saliency.py       # Vanilla Grad, IG, SmoothGrad, GxI
│   │   └── visualization.py
│   ├── envs/
│   │   └── highway_wrapper.py         # Gymnasium wrapper with normalization
│   └── utils/
│       └── logger.py                  # Per-episode JSON + PNG training logs
├── run_full_pipeline.py               # Train all 13 algorithms, generate all plots
├── run_ablation.py                    # DQN component ablation study
├── run_saliency.py                    # Gradient-based saliency visualizations
├── run_with_logging.py                # Full pipeline with checkpoints + logs
├── patch_per.py                       # Patch DQN(D+D+PER) into existing results
├── outputs/comparison/                # All generated plots and summary JSON
├── checkpoints/                       # Saved .pt model files per run
└── logs/                              # Per-episode JSON logs + return/loss PNGs
```

### Results

All algorithms trained on `highway-v0` under identical conditions (seed=42). Classical agents: 300 episodes. Deep agents: 20,000 environment steps.

| Rank | Algorithm | Final Avg-50 Return | Mean Return | Episodes | Train Time |
|------|-----------|--------------------:|------------:|---------:|-----------:|
| 1 | DQN (D+D+PER) | +27.96 | +20.33 | 723 | 27 min |
| 2 | PPO | +26.53 | +22.76 | 647 | 20 min |
| 3 | REINFORCE | +23.21 | +20.45 | 150 | 10 min |
| 4 | DQN | +22.14 | +13.13 | 1,269 | 39 min |
| 5 | Dueling DQN | +21.57 | +13.50 | 1,242 | 21 min |
| 6 | Double DQN | +20.25 | +12.78 | 1,310 | 537 min |
| 7 | Q-Learning | +15.01 | +13.00 | 300 | 6 min |
| 8 | Monte Carlo | +14.02 | +13.40 | 300 | 5 min |
| 9 | SARSA(lambda) | +13.70 | +13.48 | 300 | 6 min |
| 10 | Q(lambda) | +12.09 | +12.46 | 300 | 5 min |
| 11 | TD(lambda) | +10.89 | +10.67 | 300 | 4 min |
| 12 | Linear FA | +9.97 | +9.99 | 300 | 4 min |
| 13 | A2C | +9.80 | +10.41 | 1,632 | 39 min |
| — | Random | +8.12 | +8.12 | 50 | — |

*Final Avg-50: mean return over last 50 episodes / 50 episode-equivalents.*

### DQN Ablation Study

To understand *why* DQN (D+D+PER) achieves the highest final performance, each enhancement is isolated and tested independently under identical hyperparameters (20,000 steps, seed=42):

| Config | Final Avg-50 | Gain over Previous |
|--------|-------------:|-------------------:|
| DQN (baseline) | baseline | — |
| + Double DQN | see plot | reduces overestimation bias |
| + Dueling only | see plot | decouples V(s) from A(s,a) |
| + Double + Dueling | see plot | combined architecture |
| + D+D+PER (full) | highest | prioritizes hard transitions |

**Key finding:** Prioritized Experience Replay (PER) contributes the largest single gain on highway-v0, because the environment is highly skewed — the majority of transitions are routine lane-keeping while a small minority involve near-collision decisions. PER naturally up-weights those high-TD-error transitions, accelerating learning of the safety-critical behavior that dominates the reward signal.

See `outputs/comparison/07_dqn_ablation.png` for the full learning-curve and bar-chart comparison.

### Critical Analysis

**Why does DQN (D+D+PER) rank first despite PPO's stronger asymptotic convergence in theory?**

PPO converges faster per episode (647 vs. 723 episodes) and is on-policy — it never suffers from distribution shift in the replay buffer. However, in the first 20,000 *steps*, PPO spends many steps per update cycle collecting on-policy rollouts, while DQN(D+D+PER) continuously reuses and re-prioritizes its buffer. On a dense-traffic environment where rare, high-reward transitions (avoiding a cut-in) are the key to improvement, PER's focused replay of those experiences gives DQN an edge within the fixed step budget.

**Why does A2C severely underperform (near random baseline)?**

A2C's synchronous rollout architecture generates very short episodes on highway-v0 — the ego vehicle crashes early during exploration, producing rollouts of length 5–15 steps. With no replay buffer to compensate, the policy gradient updates are dominated by early-crash gradients, creating a local optimum of "brake immediately." The final-avg-50 of +9.80 versus the random baseline of +8.12 confirms the policy learned almost nothing. REINFORCE, despite using the same policy-gradient principle, benefits from complete-episode returns that properly credit safe lane-keeping across longer horizons.

**Why do classical tabular methods plateau around +10–15?**

The 135-state discretization (5 lanes × 3 speed bins × 3 gap bins) loses all spatial precision. Two cars that are 8m apart and 12m apart map to the same tabular state. This aliasing caps the expressiveness of the Q-table — no amount of additional episodes can overcome the fundamental information loss. The deep RL agents operate on the raw 25-dimensional observation and can represent finer distinctions between safe and unsafe configurations.

**Why does Q-Learning outperform SARSA(lambda) and MC in the classical tier?**

Q-Learning's off-policy update (`max_a Q(s',a)`) evaluates the *optimal* next action regardless of what the behavior policy actually did. In a driving environment with a 5-action space and aggressive epsilon-greedy exploration, the on-policy methods (SARSA, MC) are contaminated by poor exploratory actions when computing their targets. Q-Learning decouples exploration from evaluation, leading to cleaner value estimates at the same number of episodes.

### Saliency / Interpretability

Four gradient-based attribution methods are applied to the trained DQN agent to explain which observation features drive its decisions:

| Method | Description |
|--------|-------------|
| Vanilla Gradient | Direct gradient of Q(s,a) w.r.t. input features |
| Gradient x Input | Element-wise product — amplifies large-input features |
| SmoothGrad | Averaged gradients over Gaussian-noised inputs (reduces noise) |
| Integrated Gradients | Path integral from zero baseline; satisfies completeness axiom |

**Key finding:** The ego vehicle's own speed (feature index 2) and the longitudinal position of the nearest front vehicle (feature index 5) consistently receive the highest attribution across all methods. The agent has learned that *relative gap to the front vehicle* is the most informative signal for collision avoidance — matching domain intuition.

Integrated Gradients completeness error < 0.02% across all tested observations, confirming attribution faithfulness.

See `outputs/comparison/07_*` through `11_*` for all 31 saliency plots.

### Checkpoints and Logging

Every training run saves:

**Checkpoints** (`checkpoints/<algorithm>/highway-v0/<run_id>/`)
- `hparams.json` — hyperparameters for this run
- `actor_best.pt` / `critic_best.pt` — best policy checkpoint (PPO, REINFORCE, A2C)
- `online_net_best.pt` / `target_net_best.pt` — best Q-network (DQN variants)
- `training_log.json` — per-episode returns and losses

**Logs** (`logs/<algorithm>_highway-v0_<timestamp>/`)
- `episode_log.json` — return, episode length, epsilon per episode
- `step_log.json` — loss, TD error per gradient step
- `returns.png` — return vs episode plot
- `loss.png` — loss vs step plot

---

## Installation

**Requirements:** Python 3.9+, PyTorch 2.x

```bash
git clone https://github.com/dipayne/EECS590-CapstoneProject.git
cd EECS590-CapstoneProject
pip install -r requirements.txt        # V1 dependencies
pip install -r V2/requirements.txt     # V2 additional dependencies
```

**V2 dependencies** (`V2/requirements.txt`):
```
gymnasium
highway-env
numpy
matplotlib
torch
tqdm
```

---

## Running the Project

### V1 — Policy Iteration (Logistics Grid)

```bash
# Train, evaluate, and save all outputs
python -m src.main

# Train with custom hyperparameters
python -m src.agents.dp_agent --mode both --gamma 0.95 --theta 1e-8 --save

# Watch the agent navigate live
python -m src.utils.render --env logistics --load --episodes 3
```

### V2 — Full RL Benchmark

```bash
cd V2

# Train all 13 algorithms and generate all comparison plots
python run_full_pipeline.py

# Train with checkpoints + experiment logs saved
python run_with_logging.py

# Run DQN component ablation study (generates 07_dqn_ablation.png)
python run_ablation.py

# Generate saliency / interpretability plots
python run_saliency.py

# Add DQN(D+D+PER) to existing results and regenerate plots
python patch_per.py

# Train individual algorithm families
python train_classical.py
python train_dqn.py
python train_ppo.py
```

---

## Outputs

### V1

| File | Description |
|------|-------------|
| `outputs/plots/policy_map.png` | Arrow-grid of learned policy |
| `outputs/plots/value_heatmap.png` | Value function heatmap |
| `outputs/policies/best_policy.json` | Serialized policy |
| `outputs/values/V.npy` | V(s) array |
| `outputs/qvalues/Q.npy` | Q(s,a) table |

### V2

| File | Description |
|------|-------------|
| `V2/outputs/comparison/01_classical_comparison.png` | Classical methods learning curves |
| `V2/outputs/comparison/02_deep_comparison.png` | Deep RL learning curves |
| `V2/outputs/comparison/03_full_comparison.png` | All 13 algorithms on one plot |
| `V2/outputs/comparison/04_algorithm_table.png` | Algorithm properties table |
| `V2/outputs/comparison/05_best_vs_random.png` | Best agent vs random baseline |
| `V2/outputs/comparison/06_final_returns_bar.png` | Ranked bar chart of final returns |
| `V2/outputs/comparison/07_dqn_ablation.png` | DQN component ablation study |
| `V2/outputs/comparison/07–11_saliency_*.png` | 31 gradient saliency plots |
| `V2/outputs/comparison/results_summary.json` | Numerical results for all algorithms |
| `V2/checkpoints/` | Saved .pt model files per run |
| `V2/logs/` | Per-episode JSON + PNG training logs |

---

## Citations

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
- Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.
- Schaul, T., et al. (2016). Prioritized experience replay. *ICLR*.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. *ICML*.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
- Towers, M., et al. (2024). Gymnasium. https://gymnasium.farama.org/
- Leurent, E. (2018). *highway-env: An environment for autonomous driving decision-making*. https://github.com/Farama-Foundation/HighwayEnv
