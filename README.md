# EECS 590 Capstone Project — Version 1: Dynamic Programming on a Logistics Grid

This is the first version of my EECS 590 capstone project. The goal here was to build a small but fully specified Markov Decision Process from scratch, then solve it exactly using Dynamic Programming. Before jumping into model-free reinforcement learning (which is what Version 2 covers), I wanted to make sure I understood what an optimal policy actually looks like when you have perfect knowledge of the environment.

Version 2, which covers 13 reinforcement learning algorithms benchmarked on an autonomous driving task, lives in the `V2/` folder and has its own README there.

---

## What This Version Does

I designed a 6x6 logistics grid where an agent has to navigate from a depot to a customer location while avoiding blocked cells. The environment is stochastic — the agent slips sideways 10% of the time — so the optimal policy has to account for uncertainty. I then applied Policy Iteration to find the exact optimal policy, and built a model learning pipeline that estimates the transition dynamics from sampled trajectories.

```
D  right  right  right  right  right
down down  BLOCK  down  BLOCK  down
down down  BLOCK  down  down   down
down down  BLOCK  down  down   down
down down  BLOCK  BLOCK down   down
down down  down   down  down   GOAL
```

| Setting          | Value                                  |
|------------------|----------------------------------------|
| Grid size        | 6 x 6                                  |
| Valid states     | 30 (blocked cells excluded)            |
| Actions          | UP, RIGHT, DOWN, LEFT, WAIT            |
| Discount factor  | 0.95                                   |
| Slip probability | 0.10                                   |
| Step reward      | -1.0                                   |
| Goal reward      | +20.0                                  |

---

## How It Works

### Policy Iteration

Policy Iteration alternates between two steps until the policy stops changing. First, Policy Evaluation computes V(s) — the expected discounted return from every state under the current policy — using the Bellman expectation equation, iterated until convergence (threshold 1e-8). Second, Policy Improvement greedily updates the policy by picking whichever action maximizes the expected Q(s, a) at each state. I compute both a V-based and a Q-based improvement at every iteration, as required by the assignment, though Q-greedy is the one that drives updates.

The algorithm converges in a small number of iterations because the state space is tiny (30 states, 5 actions).

### Model Learning

In `src/mdp/model_learning.py`, the agent bootstraps its world model by rolling out random trajectories and counting transitions. This gives frequency-based estimates of P[s][a][s'] and R[s][a][s']. The counts are saved to disk so that subsequent runs can load and incrementally update the belief — a simple form of Bayesian belief updating without discarding past experience.

---

## Project Structure

```
EECS590-CapstoneProject/
├── notebooks/
│   └── 01_policy_iteration.ipynb     Interactive walkthrough of the full pipeline
├── outputs/
│   ├── plots/
│   │   ├── policy_map.png            Arrow grid of the learned policy
│   │   └── value_heatmap.png         Heatmap of V(s)
│   ├── policies/
│   │   └── best_policy.json          Action per state (serialized policy)
│   ├── values/
│   │   └── V.npy                     State value function array
│   └── qvalues/
│       └── Q.npy                     Q-value table (states x actions)
├── src/
│   ├── agents/
│   │   ├── base_agent.py             Abstract base class for all agents
│   │   └── dp_agent.py               Policy Iteration agent with CLI
│   ├── dp/
│   │   └── policy_iteration.py       Policy evaluation, improvement, and iteration
│   ├── envs/
│   │   ├── logistics_grid_mdp.py     The MDP environment definition
│   │   ├── make_env.py               Environment factory
│   │   └── test_foundation_env.py    Smoke test
│   ├── mdp/
│   │   └── model_learning.py         MDP estimation from sampled trajectories
│   ├── utils/
│   │   ├── viz.py                    Policy map and value heatmap utilities
│   │   └── render.py                 Live rendering with matplotlib and pygame
│   ├── main.py                       Entry point: runs policy iteration and saves outputs
│   └── train_eval.py                 Random policy baseline
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/dipayne/EECS590-CapstoneProject.git
cd EECS590-CapstoneProject
pip install -r requirements.txt
```

Dependencies: `gymnasium`, `highway-env`, `numpy`, `matplotlib`, `pygame`, `tqdm`

---

## Running It

```bash
# Run Policy Iteration and save all outputs
python -m src.main

# Train and evaluate the agent with custom settings
python -m src.agents.dp_agent --mode both --gamma 0.95 --theta 1e-8 --max_iter 1000 --episodes 10 --save

# Evaluate using a previously saved policy
python -m src.agents.dp_agent --mode eval --load --episodes 20

# Watch the agent navigate the grid live
python -m src.utils.render --env logistics --load --episodes 3

# Run a random policy baseline on highway-v0
python -m src.train_eval --env highway-v0 --episodes 5 --max_steps 500

# Open the interactive notebook
jupyter notebook notebooks/
```

---

## Outputs

After running `src/main.py`:

| File | What it shows |
|------|---------------|
| `outputs/plots/policy_map.png` | The optimal policy visualized as directional arrows on the grid |
| `outputs/plots/value_heatmap.png` | How valuable each state is under the optimal policy |
| `outputs/policies/best_policy.json` | The policy itself (action index per state) |
| `outputs/values/V.npy` | V(s) array |
| `outputs/qvalues/Q.npy` | Q(s, a) table |

---

## References

Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.

Towers, M., et al. (2024). Gymnasium. https://gymnasium.farama.org/

Leurent, E. (2018). highway-env: An environment for autonomous driving decision-making. https://github.com/Farama-Foundation/HighwayEnv
