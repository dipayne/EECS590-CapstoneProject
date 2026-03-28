# Comparison Plots

| File | Description |
|------|-------------|
| `01_classical_comparison.png` | Learning curves for all 6 classical tabular methods (Monte Carlo, TD(lambda), SARSA(lambda), Q-Learning, Q(lambda), Linear FA) over 300 episodes. |
| `02_deep_comparison.png` | Learning curves for all 7 deep RL agents (REINFORCE, A2C, DQN, Double DQN, Dueling DQN, DQN D+D+PER, PPO) over 20,000 environment steps. |
| `03_full_comparison.png` | All 13 algorithms on a single plot, showing the performance gap between tabular and deep RL methods on the same scale. |
| `04_algorithm_table.png` | Summary table listing each algorithm's final average return, standard deviation, number of episodes, and training time. |
| `05_best_vs_random.png` | Head-to-head comparison of the best performing agent (DQN D+D+PER) against the random policy baseline. |
| `06_final_returns_bar.png` | Horizontal bar chart ranking all 13 algorithms by final average return over the last 50 episodes, with the random baseline marked as a reference line. |
| `07_dqn_ablation.png` | Ablation study isolating the contribution of each DQN enhancement (Double, Dueling, PER), showing that PER provides the largest single gain and stabilizes the combined Double+Dueling architecture. |
| `results_summary.json` | Numerical results for all algorithms including mean return, standard deviation, episode count, final average return, and training time. |
