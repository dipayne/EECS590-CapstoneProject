"""Quick demo: train all algorithms for a short time and generate all plots.

This is a fast version of run_full_pipeline.py for demo/submission purposes.
Runs 30 episodes per classical agent, 3000 steps per deep RL agent.
Expected runtime: ~8-12 minutes on CPU.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
V1_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, V1_SRC)

from src.envs.highway_wrapper import make_highway_env

OUTPUT_DIR = Path("outputs/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Monte Carlo":    "#e41a1c",
    "TD(lam)":        "#ff7f00",
    "SARSA(lam)":     "#4daf4a",
    "Q-Learning":     "#377eb8",
    "Q(lam)":         "#984ea3",
    "Linear FA":      "#a65628",
    "REINFORCE":      "#f781bf",
    "A2C":            "#888888",
    "DQN":            "#1b7837",
    "Double DQN":     "#762a83",
    "Dueling DQN":    "#e08214",
    "DQN (D+D+PER)":  "#d6604d",
    "PPO":            "#4393c3",
    "Random":         "#cccccc",
}

N_CLASSICAL = 25     # episodes per classical agent
N_DEEP      = 1_200  # env steps per deep agent
MAX_STEPS   = 60     # max steps per episode

def moving_average(x, w):
    if len(x) < w: return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

def train_classical(env, AgentClass, kwargs, label):
    agent = AgentClass(**kwargs)
    t0 = time.time()
    rets = agent.train(env, N_CLASSICAL, MAX_STEPS, verbose_every=0)
    print(f"  {label:20s}: {len(rets):3d} ep  avg={np.mean(rets[-10:]):+.2f}  ({time.time()-t0:.0f}s)")
    return rets

def train_deep(AgentClass, kwargs, train_kwargs, label):
    env = make_highway_env(normalize_obs=True)
    agent = AgentClass(**kwargs)
    t0 = time.time()
    result = agent.train(env, **train_kwargs)
    rets = result["returns"] if isinstance(result, dict) else result
    env.close()
    print(f"  {label:20s}: {len(rets):3d} ep  avg={np.mean(rets[-5:]) if rets else 0:+.2f}  ({time.time()-t0:.0f}s)")
    return rets

def main():
    np.random.seed(42)
    print("\n=== Demo Pipeline (fast mode) ===\n")
    results = {}

    # ---- classical ----
    tab_env = make_highway_env()
    nS, nA = tab_env.n_tabular_states, tab_env.n_actions
    obs_fn  = tab_env.tabular_state
    common  = dict(nS=nS, nA=nA, obs_to_state=obs_fn, gamma=0.99,
                   epsilon=1.0, epsilon_decay=0.97, epsilon_min=0.1)

    from src.agents.classical.monte_carlo import MonteCarloAgent
    from src.agents.classical.td import TDAgent
    from src.agents.classical.sarsa import SARSAAgent
    from src.agents.classical.q_learning import QLearningAgent
    from src.agents.classical.q_lambda import QLambdaAgent
    from src.agents.classical.linear_fa import LinearFAAgent

    results["Monte Carlo"] = train_classical(tab_env, MonteCarloAgent, {**common}, "Monte Carlo")
    results["TD(lam)"]     = train_classical(tab_env, TDAgent, {**common, "alpha": 0.1, "lam": 0.9, "mode": "tdlam_backward"}, "TD(lam)")
    results["SARSA(lam)"]  = train_classical(tab_env, SARSAAgent, {**common, "alpha": 0.1, "lam": 0.9, "mode": "sarsa_lam_backward"}, "SARSA(lam)")
    results["Q-Learning"]  = train_classical(tab_env, QLearningAgent, {**common, "alpha": 0.1}, "Q-Learning")
    results["Q(lam)"]      = train_classical(tab_env, QLambdaAgent, {**common, "alpha": 0.1, "lam": 0.8}, "Q(lam)")
    results["Linear FA"]   = train_classical(tab_env, LinearFAAgent, {**common, "alpha": 0.02, "method": "sarsa"}, "Linear FA")
    tab_env.close()

    # ---- deep ----
    from src.agents.deep.reinforce import REINFORCEAgent
    from src.agents.deep.a2c import A2CAgent
    from src.agents.deep.dqn import DQNAgent
    from src.agents.deep.ppo import PPOAgent

    env_tmp = make_highway_env(normalize_obs=True)
    obs_dim, n_actions = env_tmp.obs_dim, env_tmp.n_actions
    env_tmp.close()

    results["REINFORCE"]    = train_deep(REINFORCEAgent,
        {"obs_dim": obs_dim, "n_actions": n_actions, "gamma": 0.99, "lr_actor": 5e-4, "use_baseline": True},
        {"n_episodes": N_CLASSICAL, "max_steps": MAX_STEPS, "verbose_every": 0}, "REINFORCE")

    results["A2C"]          = train_deep(A2CAgent,
        {"obs_dim": obs_dim, "n_actions": n_actions, "n_steps": 5, "gamma": 0.99, "lr": 7e-4},
        {"total_steps": N_DEEP, "max_ep_steps": MAX_STEPS, "verbose_every": 0}, "A2C")

    results["DQN"]          = train_deep(DQNAgent,
        {"obs_dim": obs_dim, "n_actions": n_actions, "variant": "dqn", "gamma": 0.99, "lr": 1e-4,
         "batch_size": 32, "buffer_capacity": 800, "target_update_freq": 80,
         "epsilon_start": 1.0, "epsilon_end": 0.1, "epsilon_decay_steps": 1_000,
         "use_per": False, "train_start": 80},
        {"n_steps": N_DEEP, "max_ep_steps": MAX_STEPS, "verbose_every": 0}, "DQN")

    results["Double DQN"]   = train_deep(DQNAgent,
        {"obs_dim": obs_dim, "n_actions": n_actions, "variant": "double", "gamma": 0.99, "lr": 1e-4,
         "batch_size": 32, "buffer_capacity": 800, "target_update_freq": 80,
         "epsilon_start": 1.0, "epsilon_end": 0.1, "epsilon_decay_steps": 1_000,
         "use_per": False, "train_start": 80},
        {"n_steps": N_DEEP, "max_ep_steps": MAX_STEPS, "verbose_every": 0}, "Double DQN")

    results["Dueling DQN"]  = train_deep(DQNAgent,
        {"obs_dim": obs_dim, "n_actions": n_actions, "variant": "dueling", "gamma": 0.99, "lr": 1e-4,
         "batch_size": 32, "buffer_capacity": 800, "target_update_freq": 80,
         "epsilon_start": 1.0, "epsilon_end": 0.1, "epsilon_decay_steps": 1_000,
         "use_per": False, "train_start": 80},
        {"n_steps": N_DEEP, "max_ep_steps": MAX_STEPS, "verbose_every": 0}, "Dueling DQN")

    results["DQN (D+D+PER)"] = train_deep(DQNAgent,
        {"obs_dim": obs_dim, "n_actions": n_actions, "variant": "double_dueling", "gamma": 0.99,
         "lr": 1e-4, "batch_size": 32, "buffer_capacity": 800, "target_update_freq": 80,
         "epsilon_start": 1.0, "epsilon_end": 0.1, "epsilon_decay_steps": 1_000,
         "use_per": True, "train_start": 80},
        {"n_steps": N_DEEP, "max_ep_steps": MAX_STEPS, "verbose_every": 0}, "DQN (D+D+PER)")

    results["PPO"]          = train_deep(PPOAgent,
        {"obs_dim": obs_dim, "n_actions": n_actions, "n_steps": 128, "n_epochs": 3,
         "batch_size": 32, "gamma": 0.99, "lam": 0.95, "clip_eps": 0.2, "lr": 3e-4,
         "ent_coef": 0.01, "normalize_obs": True},
        {"total_steps": N_DEEP, "max_ep_steps": MAX_STEPS, "verbose_every": 0}, "PPO")

    # random baseline
    rand_env = make_highway_env()
    rand_rets = []
    for _ in range(30):
        obs, _ = rand_env.reset()
        ep_r, g = 0.0, 1.0
        for _ in range(MAX_STEPS):
            obs, r, te, tr, _ = rand_env.step(rand_env.action_space.sample())
            ep_r += g*r; g *= 0.99
            if te or tr: break
        rand_rets.append(ep_r)
    results["Random"] = rand_rets
    rand_env.close()

    # ---- save results ----
    summary = {}
    for name, rets in results.items():
        if rets:
            summary[name] = {"mean": float(np.mean(rets)), "std": float(np.std(rets)),
                             "final_avg": float(np.mean(rets[-10:])), "n_ep": len(rets)}
    with open(OUTPUT_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plots ----
    w = 8

    def plot_group(data, title, fname, window=8):
        fig, ax = plt.subplots(figsize=(12, 5))
        colors_local = list(plt.cm.tab10.colors) + list(plt.cm.tab20b.colors)
        for i, (name, rets) in enumerate(data.items()):
            if not rets: continue
            color = COLORS.get(name, colors_local[i % len(colors_local)])
            eps = list(range(len(rets)))
            ax.plot(eps, rets, alpha=0.15, color=color)
            if len(rets) >= window:
                sm = moving_average(rets, window)
                ax.plot(eps[window-1:], sm, color=color, linewidth=2, label=name)
            else:
                ax.plot(eps, rets, color=color, linewidth=2, label=name)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Episode"); ax.set_ylabel("Discounted Return")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, ncol=2, loc="lower right"); ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(OUTPUT_DIR / fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved:", fname)

    classical_keys = ["Monte Carlo", "TD(lam)", "SARSA(lam)", "Q-Learning", "Q(lam)", "Linear FA"]
    deep_keys = ["REINFORCE", "A2C", "DQN", "Double DQN", "Dueling DQN", "DQN (D+D+PER)", "PPO"]

    plot_group({k: results[k] for k in classical_keys if k in results},
               "Classical RL Methods - highway-v0", "01_classical_comparison.png")
    plot_group({k: results[k] for k in deep_keys if k in results},
               "Deep RL Methods - highway-v0", "02_deep_comparison.png", window=5)
    plot_group({k: results[k] for k in list(results.keys()) if results[k]},
               "All Algorithms - highway-v0", "03_full_comparison.png")

    # Bar chart
    names  = [k for k in summary if k != "Random"]
    vals   = [summary[k]["final_avg"] for k in names]
    stds   = [summary[k]["std"] for k in names]
    order  = np.argsort(vals)
    names  = [names[i] for i in order]; vals = [vals[i] for i in order]; stds = [stds[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, max(4, len(names)*0.45)))
    colors_bar = [COLORS.get(n, "#aaaaaa") for n in names]
    ax.barh(range(len(names)), vals, color=colors_bar, alpha=0.85)
    ax.errorbar(vals, range(len(names)), xerr=stds, fmt="none", color="black", capsize=3)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(summary.get("Random", {}).get("final_avg", 0), color="gray",
               linewidth=1.2, linestyle="--", label="Random baseline")
    ax.set_xlabel("Final Avg-10 Discounted Return", fontsize=11)
    ax.set_title("Algorithm Ranking - highway-v0", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "06_final_returns_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 06_final_returns_bar.png")

    # Best vs Random
    best_k = max([k for k in summary if k != "Random"], key=lambda k: summary[k]["final_avg"])
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, color, lw in [(best_k, "#d6604d", 2.5), ("Random", "#888888", 1.5)]:
        if k not in results or not results[k]: continue
        rets = results[k]; eps = list(range(len(rets)))
        ax.plot(eps, rets, alpha=0.15, color=color)
        sm = moving_average(rets, min(8, len(rets)))
        ax.plot(eps[min(8,len(rets))-1:], sm, color=color, linewidth=lw, label=k)
    ax.set_xlabel("Episode"); ax.set_ylabel("Discounted Return")
    ax.set_title(f"Best Agent ({best_k}) vs Random Baseline")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "05_best_vs_random.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 05_best_vs_random.png")

    print(f"\n=== Demo complete.  Plots in {OUTPUT_DIR} ===")
    print("\nAlgorithm ranking (final avg-10 return):")
    for k, v in sorted(summary.items(), key=lambda x: x[1]["final_avg"], reverse=True):
        print(f"  {k:<22} {v['final_avg']:+.3f}")

if __name__ == "__main__":
    main()
