# EECS 590 Capstone Project — Version 2: Full RL Algorithm Benchmark on Highway-env

This is the second version of my EECS 590 capstone project. Where Version 1 solved a small custom MDP exactly using Dynamic Programming, this version asks a harder question: how do 13 different reinforcement learning algorithms — ranging from simple tabular Q-Learning to PPO with a neural network — compare when trained on the same autonomous driving task under identical conditions?

The environment is `highway-v0` from highway-env, where an agent controls a vehicle on a 4-lane highway and has to drive quickly without crashing into traffic. Nothing about the transition dynamics is given to the agent — it has to learn purely from experience.

---

## Table of Contents

- [The Environment](#the-environment)
- [Algorithms](#algorithms)
- [Results](#results)
- [DQN Ablation Study](#dqn-ablation-study)
- [Critical Analysis](#critical-analysis)
- [Saliency and Interpretability](#saliency-and-interpretability)
- [Checkpoints and Logging](#checkpoints-and-logging)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [References](#references)

---

## The Environment

`highway-v0` places an ego vehicle on a 4-lane highway alongside randomly generated traffic. At each timestep, the agent observes the positions and velocities of the 5 nearest vehicles as a 25-dimensional flat vector (5 vehicles x 5 features each). It then chooses one of 5 discrete actions: idle, change lane right, change lane left, accelerate, or decelerate.

The reward function combines a speed bonus, a lane-keeping incentive, and a large collision penalty. Episodes end on collision or after a fixed number of steps.

For the classical tabular agents, I discretize the observation into 135 states (5 lane positions x 3 speed bins x 3 front-gap bins x 3 rear-gap bins). This is a lossy abstraction but it is the only way to apply Q-tables to this environment. Deep RL agents receive the full 25-dimensional observation.

| Feature | Value |
|---------|-------|
| Observation | 25-dimensional kinematics vector |
| Action space | Discrete(5) |
| Tabular state space | 135 states |
| Classical training | 300 episodes |
| Deep RL training | 20,000 environment steps |
| Discount factor | 0.99 |

---

## Algorithms

### What I Implemented

| Category | Algorithm | Key Idea |
|----------|-----------|----------|
| Classical tabular | Monte Carlo Control | Full-episode returns, first-visit updates |
| Classical tabular | TD(lambda) | Eligibility traces, backward view |
| Classical tabular | SARSA(lambda) | On-policy traces |
| Classical tabular | Q-Learning | Off-policy, max-Q targets |
| Classical tabular | Q(lambda) | Watkins' trace-cutting on off-policy steps |
| Classical tabular | Linear FA | SARSA with one-hot tile features |
| Deep RL | DQN | Mnih et al. 2015, experience replay + target network |
| Deep RL | Double DQN | Reduces Q-value overestimation |
| Deep RL | Dueling DQN | Separate value and advantage streams |
| Deep RL | DQN (D+D+PER) | Double + Dueling + Prioritized Experience Replay |
| Deep RL | REINFORCE | Policy gradient with value baseline |
| Deep RL | A2C | Synchronous Advantage Actor-Critic |
| Deep RL | PPO | Proximal Policy Optimization with GAE |

### What I Did Not Implement and Why

**DDPG, TD3, SAC** require continuous action spaces. They output real-valued torques or velocities using deterministic or stochastic policies over R^n. Since highway-v0 uses a discrete 5-action space, these algorithms cannot be applied without fundamentally changing the environment itself.

**A3C** requires truly asynchronous parallel workers sharing gradient updates across processes. On a single Windows machine in a standard Python environment, this is not practically feasible. A2C captures the same actor-critic learning signal in a synchronous form and is the standard replacement.

**TRPO** requires computing the inverse of the Fisher information matrix at every update, which is computationally expensive and architecturally complex. PPO achieves the same trust-region objective using a simple clipping trick and performs comparably in practice. There is no reason to implement TRPO when PPO is available.

**Value Iteration and Policy Iteration** are model-based methods that require a known transition model P(s'|s,a). In highway-v0, the environment is a black-box simulator — there is no analytical transition model to work with. These methods were covered in Version 1 on the logistics grid where the model is fully specified.

---

## Results

All algorithms were trained with seed=42 on the same environment configuration. The table below ranks by final-avg-50, which is the mean return over the last 50 episodes (or episode-equivalents for step-based agents).

| Rank | Algorithm | Final Avg-50 | Mean Return | Episodes | Train Time |
|------|-----------|-------------:|------------:|---------:|------------|
| 1 | DQN (D+D+PER) | +27.96 | +20.33 | 723 | ~27 min |
| 2 | PPO | +26.53 | +22.76 | 647 | ~20 min |
| 3 | REINFORCE | +23.21 | +20.45 | 150 | ~10 min |
| 4 | DQN | +22.14 | +13.13 | 1,269 | ~39 min |
| 5 | Dueling DQN | +21.57 | +13.50 | 1,242 | ~21 min |
| 6 | Double DQN | +20.25 | +12.78 | 1,310 | ~537 min |
| 7 | Q-Learning | +15.01 | +13.00 | 300 | ~6 min |
| 8 | Monte Carlo | +14.02 | +13.40 | 300 | ~5 min |
| 9 | SARSA(lambda) | +13.70 | +13.48 | 300 | ~6 min |
| 10 | Q(lambda) | +12.09 | +12.46 | 300 | ~5 min |
| 11 | TD(lambda) | +10.89 | +10.67 | 300 | ~4 min |
| 12 | Linear FA | +9.97 | +9.99 | 300 | ~4 min |
| 13 | A2C | +9.80 | +10.41 | 1,632 | ~39 min |
| Random baseline | | +8.12 | +8.12 | 50 | |

---

## DQN Ablation Study

Simply reporting that "DQN (D+D+PER) works best" is not a satisfying answer. I want to know *which part* of it matters. To find out, I trained five configurations under identical conditions, adding one component at a time:

| Config | What changes | Contribution |
|--------|-------------|--------------|
| DQN baseline | Standard DQN | Starting point |
| + Double | Decouple action selection from evaluation | Reduces Q-value overestimation |
| + Dueling | Separate V(s) and A(s,a) streams | Better state value estimation |
| + Double + Dueling | Both architectural changes | Combined structural gains |
| + D+D+PER (full) | Prioritize high-TD-error transitions | Largest single gain on this env |

The key finding is that Prioritized Experience Replay provides the largest individual performance boost on highway-v0. This makes sense: most transitions in highway driving are routine (cruising in a clear lane), while the rare near-collision experiences carry the most learning signal. PER naturally up-weights those critical transitions so the agent spends more time learning from them.

See `outputs/comparison/07_dqn_ablation.png` for the full learning curves and bar chart.

---

## Critical Analysis

**Why does DQN (D+D+PER) edge out PPO despite PPO's theoretical advantages?**

PPO is on-policy and converges faster per episode (647 vs 723 episodes). But within a fixed budget of 20,000 environment steps, DQN with PER continuously replays and re-prioritizes its most informative transitions. On a task where rare high-reward events dominate learning, that focused replay matters more than PPO's lower-variance gradient estimates.

**Why does A2C perform almost as poorly as a random agent?**

A2C generates very short rollouts on highway-v0 during early training because the untrained policy crashes quickly. With no replay buffer and short episodes, its policy gradient updates are dominated by crash-induced gradients, pushing the policy toward a degenerate "brake immediately" behavior. REINFORCE avoids this because it waits for full episodes before updating, giving the return signal a chance to properly credit safe driving decisions.

**Why do all tabular methods plateau below +15?**

The 135-state discretization loses spatial precision. Two scenarios that differ only in whether the front vehicle is 8 meters away versus 12 meters away map to the same tabular state. No amount of training can overcome this information bottleneck — the Q-table simply cannot represent the difference. Deep RL agents operating on the raw 25-dimensional observation do not have this problem.

**Why does Q-Learning outperform SARSA and Monte Carlo in the classical tier?**

Q-Learning is off-policy: it targets the greedy action regardless of what the behavior policy actually did during exploration. SARSA and Monte Carlo use the actual observed actions in their targets, which means exploratory (poor) actions pollute the value estimates. On a 5-action space with aggressive epsilon-greedy exploration, this contamination is significant, and Q-Learning's decoupling of exploration from evaluation makes a measurable difference.

---

## Saliency and Interpretability

To understand what the trained DQN agent is actually paying attention to, I applied four gradient-based attribution methods to its decisions:

| Method | How it works |
|--------|-------------|
| Vanilla Gradient | Computes the gradient of the selected Q-value with respect to the input |
| Gradient x Input | Multiplies the gradient by the input value, amplifying features that are both large and sensitive |
| SmoothGrad | Averages gradients over many noisy copies of the input to reduce attribution noise |
| Integrated Gradients | Integrates the gradient along a straight path from a zero baseline to the input, satisfying a mathematical completeness property |

The main finding is consistent across all four methods: the agent's own speed and the longitudinal distance to the nearest front vehicle receive the highest attribution scores by a large margin. In other words, the agent learned to focus on exactly the two features that matter most for collision avoidance — which matches what a human driver would say if asked the same question.

The Integrated Gradients completeness error was below 0.02% on all tested observations, confirming that the attributions faithfully account for the full Q-value.

Saliency plots are in `outputs/comparison/` as files `07_` through `11_`.

---

## Checkpoints and Logging

Every training run saves models and metrics automatically.

Checkpoints are saved to `checkpoints/<algorithm>/highway-v0/<run_id>/` and include:

- `hparams.json` — all hyperparameters used for that run
- `actor_best.pt` and `critic_best.pt` — best policy weights for actor-critic agents
- `online_net_best.pt` and `target_net_best.pt` — best Q-network weights for DQN variants
- `training_log.json` — episode returns and losses over the full training run

Logs are saved to `logs/<algorithm>_highway-v0_<timestamp>/` and include:

- `episode_log.json` — return, episode length, and epsilon value per episode
- `step_log.json` — loss and TD error per gradient step
- `returns.png` — return vs episode plot
- `loss.png` — loss vs gradient step plot

---

## Project Structure

```
V2/
├── src/
│   ├── agents/
│   │   ├── base.py                     Abstract BaseAgent interface
│   │   ├── classical/
│   │   │   ├── monte_carlo.py
│   │   │   ├── td.py                   TD(0), TD(n), TD(lambda)
│   │   │   ├── sarsa.py                SARSA and SARSA(lambda)
│   │   │   ├── q_learning.py           Q-Learning and Double Q-Learning
│   │   │   ├── q_lambda.py             Q(lambda) with Watkins' traces
│   │   │   └── linear_fa.py            Linear function approximation
│   │   └── deep/
│   │       ├── dqn.py                  DQN, Double, Dueling, D+D+PER
│   │       ├── reinforce.py            REINFORCE with value baseline
│   │       ├── a2c.py                  Synchronous A2C
│   │       └── ppo.py                  PPO with GAE
│   ├── networks/
│   │   ├── mlp.py                      MLP backbone
│   │   └── dueling.py                  Dueling stream architecture
│   ├── replay_buffer/
│   │   └── buffer.py                   Standard and Prioritized (SumTree) replay
│   ├── checkpoints/
│   │   └── manager.py                  Best-checkpoint and periodic saving
│   ├── saliency/
│   │   ├── gradient_saliency.py        Vanilla Gradient, IG, SmoothGrad, GxI
│   │   └── visualization.py
│   ├── envs/
│   │   └── highway_wrapper.py          Gymnasium wrapper with obs normalization
│   └── utils/
│       └── logger.py                   Per-episode JSON and PNG training logs
├── run_full_pipeline.py                Train all 13 algorithms, generate all plots
├── run_ablation.py                     DQN component ablation study
├── run_saliency.py                     Gradient-based saliency visualizations
├── run_with_logging.py                 Full pipeline with checkpoints and logs
├── patch_per.py                        Add DQN(D+D+PER) to existing results
├── outputs/comparison/                 All generated plots and summary JSON
├── checkpoints/                        Saved model files per algorithm per run
└── logs/                               Per-episode JSON logs and training plots
```

---

## Installation

```bash
git clone https://github.com/dipayne/EECS590-CapstoneProject.git
cd EECS590-CapstoneProject/V2
pip install -r requirements.txt
```

Dependencies: `gymnasium`, `highway-env`, `numpy`, `matplotlib`, `torch`, `tqdm`

---

## How to Run

```bash
# Train all 13 algorithms and generate every comparison plot
python run_full_pipeline.py

# Train with full checkpoint and logging support
python run_with_logging.py

# Run the DQN ablation study
python run_ablation.py

# Generate saliency plots for the trained DQN agent
python run_saliency.py

# Train classical and deep agents individually
python train_classical.py
python train_dqn.py
python train_ppo.py
```

---

## Outputs

| File | What it shows |
|------|---------------|
| `outputs/comparison/01_classical_comparison.png` | Learning curves for all tabular methods |
| `outputs/comparison/02_deep_comparison.png` | Learning curves for all deep RL agents |
| `outputs/comparison/03_full_comparison.png` | All 13 algorithms on a single plot |
| `outputs/comparison/04_algorithm_table.png` | Summary table of algorithm properties |
| `outputs/comparison/05_best_vs_random.png` | Best performing agent vs random baseline |
| `outputs/comparison/06_final_returns_bar.png` | Ranked bar chart of final returns |
| `outputs/comparison/07_dqn_ablation.png` | DQN component ablation study |
| `outputs/comparison/07_11_saliency_*.png` | 31 gradient saliency plots |
| `outputs/comparison/results_summary.json` | Numerical results for all algorithms |
| `checkpoints/` | Saved model weights per algorithm per run |
| `logs/` | Per-episode JSON logs and training plots |

---

## References

Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.

Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518, 529-533.

van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. AAAI.

Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. ICML.

Schaul, T., et al. (2016). Prioritized experience replay. ICLR.

Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.

Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. ICML.

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. ICML.

Towers, M., et al. (2024). Gymnasium. https://gymnasium.farama.org/

Leurent, E. (2018). highway-env: An environment for autonomous driving decision-making. https://github.com/Farama-Foundation/HighwayEnv
