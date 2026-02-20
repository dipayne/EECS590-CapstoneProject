# EECS590 Capstone Project — Logistics Grid RL Framework

A research-oriented reinforcement learning framework built on an explicit Markov Decision Process (MDP). This project implements a stochastic logistics delivery environment, dynamic programming methods for policy evaluation and improvement, a model-learning pipeline for bootstrapping world-model beliefs, and visualization utilities for inspecting learned policies and value functions.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Environment / MDP](#environment--mdp)
- [MDP Components](#mdp-components)
- [Dynamic Programming](#dynamic-programming)
- [Model Learning](#model-learning)
- [Visualization](#visualization)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Outputs](#outputs)
- [Collaborations](#collaborations)
- [Citations](#citations)

---

## Project Structure

```
EECS590-CapstoneProject/
├── notebooks/
│   └── 01_policy_iteration.ipynb # Interactive walkthrough: MDP setup, training, visualization
├── outputs/
│   ├── plots/
│   │   ├── policy_map.png        # Visual grid of the learned policy
│   │   └── value_heatmap.png     # Heatmap of the learned value function
│   ├── policies/
│   │   └── best_policy.json      # Best policy kernel (action per state)
│   ├── values/
│   │   └── V.npy                 # State value function array
│   └── qvalues/
│       └── Q.npy                 # Q-value table (state x action)
├── src/
│   ├── agents/
│   │   ├── base_agent.py         # Abstract base class for all agents
│   │   └── dp_agent.py           # Policy Iteration agent with CLI interface
│   ├── dp/
│   │   └── policy_iteration.py   # Policy evaluation, improvement, and iteration
│   ├── envs/
│   │   ├── logistics_grid_mdp.py # Custom MDP environment definition
│   │   ├── make_env.py           # Environment factory
│   │   └── test_foundation_env.py
│   ├── mdp/
│   │   └── model_learning.py     # MDP estimation, belief save/load/update
│   ├── utils/
│   │   └── viz.py                # Policy map and value heatmap utilities
│   ├── main.py                   # Entry point: runs policy iteration and saves outputs
│   └── train_eval.py             # Random policy baseline / environment smoke test
├── requirements.txt
└── README.md
```

---

## Environment / MDP

The foundation environment is a **6×6 Logistics Grid** in which an agent must navigate from a depot to a customer location while avoiding blocked cells. The environment is defined in `src/envs/logistics_grid_mdp.py`.

```
D → → → → →
↓ ↓ █ ↓ █ ↓
↓ ↓ █ ↓ ↓ ↓
↓ ↓ █ ↓ ↓ ↓
↓ ↓ █ █ ↓ ↓
↓ ↓ ↓ ↓ ↓ G
```
*(D = Depot, G = Goal/Customer, █ = Blocked cell)*

| Parameter       | Value         |
|----------------|---------------|
| Grid size       | 6 × 6         |
| Depot           | (0, 0)        |
| Customer/Goal   | (5, 5)        |
| Blocked cells   | (1,2),(2,2),(3,2),(4,2),(4,3),(1,4) |
| Discount (γ)    | 0.95          |
| Slip probability| 0.10          |

---

## MDP Components

### States
Integer-encoded grid positions `s = row * cols + col`. Blocked cells are excluded from the valid state set. The terminal (absorbing) state is the customer location.

### Actions
| ID | Action |
|----|--------|
| 0  | UP     |
| 1  | RIGHT  |
| 2  | DOWN   |
| 3  | LEFT   |
| 4  | WAIT   |

### Stochastic Transitions
Movement is stochastic. When the agent selects a direction, there is:
- **80%** probability of moving in the intended direction
- **10%** probability of slipping left of the intended direction
- **10%** probability of slipping right of the intended direction

The WAIT action is deterministic (agent stays in place). Moves into walls or blocked cells result in the agent staying in its current state.

### Rewards
| Event               | Reward |
|--------------------|--------|
| Each step taken    | −1.0   |
| Reaching customer  | +20.0  |
| Terminal state     | 0.0    |

---

## Dynamic Programming

Implemented in `src/dp/policy_iteration.py`.

### Policy Evaluation
Iteratively computes the state value function **V(s)** under a fixed policy using the Bellman expectation equation until convergence (threshold θ = 1e-8).

### Policy Improvement from V
Greedily improves the policy by selecting the action that maximizes the expected value over successor states.

### Q-Value Computation
Derives the full **Q(s, a)** table from the converged value function **V** using the Bellman equation across all state-action pairs.

### Policy Improvement from Q
Greedily improves the policy by selecting `argmax_a Q(s, a)` for each state.

### Policy Iteration
Alternates between policy evaluation and Q-based policy improvement until the policy stabilizes. Both V-based and Q-based improvement are computed at each iteration (as required), with Q-greedy used as the update rule.

---

## Model Learning

Implemented in `src/mdp/model_learning.py`.

The `estimate_mdp()` function bootstraps the agent's world model by sampling random trajectories from the foundation environment. It estimates:
- **Transition probabilities** `P[s][a][s']` via frequency counts
- **Expected rewards** `R[s][a][s']` via sample averaging

### Persistent Belief Updates
Raw transition counts and reward sums are serialized to disk via `counts_path`. On subsequent runs the prior counts are loaded and new samples are merged in — implementing incremental Bayesian-style belief updates over the MDP dynamics without discarding past experience.

| Function | Description |
|----------|-------------|
| `estimate_mdp(..., counts_path=...)` | Sample environment and update stored beliefs |
| `save_model(P, R, path)` | Save a finalized (P, R) model to disk |
| `load_model(path)` | Reload a saved (P, R) model |

---

## Visualization

Implemented in `src/utils/viz.py`.

| Utility | Description |
|--------|-------------|
| `print_policy_grid()` | Prints an ASCII arrow map of the learned policy to the terminal |
| `save_policy_map_png()` | Saves a PNG grid visualization of the policy (arrows, depot, goal, blocked cells) |
| `save_value_heatmap_png()` | Saves a color heatmap of the state value function V |

---

## Installation

**Requirements:** Python 3.9+

```bash
git clone https://github.com/dipayne/EECS590-CapstoneProject.git
cd EECS590-CapstoneProject
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):
```
gymnasium
numpy
matplotlib
pygame
tqdm
```

---

## Running the Project

### Run Policy Iteration (main entry point)
Runs policy iteration on the Logistics Grid MDP, prints the policy map, and saves all outputs.

```bash
python -m src.main
```

### Train and Evaluate the DPAgent (with hyperparameters)
```bash
# Train, evaluate, and save outputs
python -m src.agents.dp_agent --mode both --gamma 0.95 --theta 1e-8 --max_iter 1000 --episodes 10 --save

# Evaluate only using a previously saved policy
python -m src.agents.dp_agent --mode eval --load --episodes 20

# Train only
python -m src.agents.dp_agent --mode train --gamma 0.99 --save
```

### Run Random Policy Baseline
Evaluates a random agent on a Gymnasium environment for a specified number of episodes.

```bash
python -m src.train_eval --env highway-v0 --episodes 5 --max_steps 500
```

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

---

## Outputs

After running `src/main.py`, the following files are saved to `outputs/`:

| File | Description |
|------|-------------|
| `outputs/policies/best_policy.json` | Best policy kernel — action index per state |
| `outputs/values/V.npy` | State value function array (shape: `[nS]`) |
| `outputs/qvalues/Q.npy` | Q-value table (shape: `[nS, nA]`) |
| `outputs/plots/policy_map.png` | PNG visualization of the learned policy |
| `outputs/plots/value_heatmap.png` | PNG heatmap of the value function |

---

## Collaborations

<!-- List any collaborators or study partners here. Example: -->
- Solo project — no collaborators for Version 1.

---

## Citations

<!-- Add any papers, textbooks, or resources you referenced. Examples below: -->
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Gymnasium documentation: https://gymnasium.farama.org/
