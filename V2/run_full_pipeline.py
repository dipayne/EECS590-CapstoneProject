"""Full V2 pipeline: train ALL applicable algorithms, generate comparison plots.

This script runs every algorithm studied in class (that applies to the
discrete-action highway-v0 environment), saves results, and produces:

  outputs/comparison/
    01_classical_comparison.png   - tabular methods side-by-side
    02_deep_comparison.png        - deep RL methods side-by-side
    03_full_comparison.png        - all methods on one canvas
    04_algorithm_table.png        - printable applicability/property table
    05_best_vs_random.png         - best agent vs random policy baseline
    results_summary.json          - mean/std returns per algorithm

Usage
-----
    python run_full_pipeline.py                    # default quick run
    python run_full_pipeline.py --mode full        # longer training
    python run_full_pipeline.py --mode quick       # fast demo (~5 min)

Note: DDPG, TD3, SAC require continuous action spaces and are NOT applicable
to highway-v0 (Discrete(5)).  TRPO and A3C are noted but not run due to
implementation complexity (TRPO requires Fisher-vector products; A3C
requires multiprocessing workers).  Both are discussed in the applicability
table below.
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
V1_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, V1_SRC)

from src.envs.highway_wrapper import make_highway_env

OUTPUT_DIR = Path("outputs/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Training config per mode
# ---------------------------------------------------------------------------

CONFIGS = {
    "quick": {
        "classical_episodes": 100,
        "max_steps":          100,
        "deep_steps":         10_000,
        "reinforce_episodes": 80,
        "a2c_steps":          10_000,
        "verbose_every":      50,
        "smooth_window":      10,
    },
    "full": {
        "classical_episodes": 1500,
        "max_steps":          200,
        "deep_steps":         150_000,
        "reinforce_episodes": 500,
        "a2c_steps":          100_000,
        "verbose_every":      200,
        "smooth_window":      30,
    },
}


# ---------------------------------------------------------------------------
# Algorithm registry (name → (category, color, run_fn))
# ---------------------------------------------------------------------------

COLORS = {
    # Classical tabular
    "Monte Carlo":    "#e41a1c",
    "TD(lam)":        "#ff7f00",
    "SARSA(lam)":     "#4daf4a",
    "Q-Learning":     "#377eb8",
    "Q(lam)":         "#984ea3",
    "Linear FA":      "#a65628",
    # Deep RL
    "REINFORCE":      "#f781bf",
    "A2C":            "#999999",
    "DQN":            "#1b7837",
    "Double DQN":     "#762a83",
    "Dueling DQN":    "#e08214",
    "DQN (D+D+PER)":  "#d6604d",
    "PPO":            "#4393c3",
}


def moving_average(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ---------------------------------------------------------------------------
# Individual agent trainers
# ---------------------------------------------------------------------------

def run_monte_carlo(env, cfg) -> list:
    from src.agents.classical.monte_carlo import MonteCarloAgent
    nS = env.n_tabular_states
    nA = env.n_actions
    agent = MonteCarloAgent(nS, nA, env.tabular_state,
                             gamma=0.99, epsilon=1.0,
                             epsilon_decay=0.997, epsilon_min=0.05)
    return agent.train(env, cfg["classical_episodes"], cfg["max_steps"],
                       verbose_every=cfg["verbose_every"])


def run_td_lambda(env, cfg) -> list:
    from src.agents.classical.td import TDAgent
    nS = env.n_tabular_states
    nA = env.n_actions
    agent = TDAgent(nS, nA, env.tabular_state, gamma=0.99, alpha=0.1,
                    lam=0.9, mode="tdlam_backward", epsilon=1.0,
                    epsilon_decay=0.997, epsilon_min=0.05)
    return agent.train(env, cfg["classical_episodes"], cfg["max_steps"],
                       verbose_every=cfg["verbose_every"])


def run_sarsa_lambda(env, cfg) -> list:
    from src.agents.classical.sarsa import SARSAAgent
    nS = env.n_tabular_states
    nA = env.n_actions
    agent = SARSAAgent(nS, nA, env.tabular_state, gamma=0.99, alpha=0.1,
                       lam=0.9, mode="sarsa_lam_backward", epsilon=1.0,
                       epsilon_decay=0.997, epsilon_min=0.05)
    return agent.train(env, cfg["classical_episodes"], cfg["max_steps"],
                       verbose_every=cfg["verbose_every"])


def run_q_learning(env, cfg) -> list:
    from src.agents.classical.q_learning import QLearningAgent
    nS = env.n_tabular_states
    nA = env.n_actions
    agent = QLearningAgent(nS, nA, env.tabular_state, gamma=0.99, alpha=0.1,
                           epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.05)
    return agent.train(env, cfg["classical_episodes"], cfg["max_steps"],
                       verbose_every=cfg["verbose_every"])


def run_q_lambda(env, cfg) -> list:
    from src.agents.classical.q_lambda import QLambdaAgent
    nS = env.n_tabular_states
    nA = env.n_actions
    agent = QLambdaAgent(nS, nA, env.tabular_state, gamma=0.99, alpha=0.1,
                         lam=0.8, epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.05)
    return agent.train(env, cfg["classical_episodes"], cfg["max_steps"],
                       verbose_every=cfg["verbose_every"])


def run_linear_fa(env, cfg) -> list:
    from src.agents.classical.linear_fa import LinearFAAgent
    nS = env.n_tabular_states
    nA = env.n_actions
    agent = LinearFAAgent(nS, nA, env.tabular_state, gamma=0.99, alpha=0.02,
                          method="sarsa", epsilon=1.0,
                          epsilon_decay=0.997, epsilon_min=0.05)
    return agent.train(env, cfg["classical_episodes"], cfg["max_steps"],
                       verbose_every=cfg["verbose_every"])


def run_reinforce(env_factory, cfg) -> list:
    from src.agents.deep.reinforce import REINFORCEAgent
    env = env_factory()
    agent = REINFORCEAgent(obs_dim=env.obs_dim, n_actions=env.n_actions,
                           gamma=0.99, lr_actor=5e-4, use_baseline=True)
    result = agent.train(env, cfg["reinforce_episodes"], max_steps=cfg["max_steps"],
                         verbose_every=max(1, cfg["verbose_every"] // 2))
    env.close()
    return result["returns"]


def run_a2c(env_factory, cfg) -> list:
    from src.agents.deep.a2c import A2CAgent
    env = env_factory()
    agent = A2CAgent(obs_dim=env.obs_dim, n_actions=env.n_actions,
                     n_steps=5, gamma=0.99, lr=7e-4,
                     ent_coef=0.01, shared_backbone=True)
    result = agent.train(env, cfg["a2c_steps"], max_ep_steps=cfg["max_steps"],
                         verbose_every=cfg["verbose_every"] * 10)
    env.close()
    return result["returns"]


def run_dqn_variant(env_factory, cfg, variant="dqn", use_per=False) -> list:
    from src.agents.deep.dqn import DQNAgent
    env = env_factory()
    n_steps = cfg["deep_steps"]
    agent = DQNAgent(
        obs_dim=env.obs_dim, n_actions=env.n_actions,
        variant=variant, gamma=0.99, lr=1e-4, batch_size=64,
        buffer_capacity=min(20_000, n_steps),
        target_update_freq=300,
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=int(n_steps * 0.7),
        use_per=use_per, train_start=500,
    )
    result = agent.train(env, n_steps, max_ep_steps=cfg["max_steps"],
                         verbose_every=cfg["verbose_every"] * 10)
    env.close()
    return result["returns"]


def run_ppo(env_factory, cfg) -> list:
    from src.agents.deep.ppo import PPOAgent
    env = env_factory()
    n_steps = cfg["deep_steps"]
    agent = PPOAgent(
        obs_dim=env.obs_dim, n_actions=env.n_actions,
        n_steps=512, n_epochs=4, batch_size=64,
        gamma=0.99, lam=0.95, clip_eps=0.2,
        lr=3e-4, ent_coef=0.01, normalize_obs=True,
    )
    result = agent.train(env, n_steps, max_ep_steps=cfg["max_steps"],
                         verbose_every=max(1, cfg["verbose_every"] // 5))
    env.close()
    return result["returns"]


def run_random_baseline(env, n_episodes=100) -> list:
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        gamma_t = 1.0
        for _ in range(200):
            a = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += gamma_t * r
            gamma_t *= 0.99
            if terminated or truncated:
                break
        returns.append(ep_ret)
    return returns


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    cfg = CONFIGS[args.mode]

    print(f"\n{'='*60}")
    print(f"  V2 Full Pipeline  mode={args.mode}  seed={args.seed}")
    print(f"{'='*60}\n")

    # Shared tabular env (kept open for classical methods)
    tabular_env = make_highway_env()
    # Factory for deep RL (each agent creates its own env)
    env_factory = lambda: make_highway_env(normalize_obs=True)

    results: dict = {}
    timing:  dict = {}

    # ----------------------------------------------------------------
    # Classical methods
    # ----------------------------------------------------------------
    classical_agents = [
        ("Monte Carlo",  run_monte_carlo,  tabular_env),
        ("TD(lam)",      run_td_lambda,    tabular_env),
        ("SARSA(lam)",   run_sarsa_lambda, tabular_env),
        ("Q-Learning",   run_q_learning,   tabular_env),
        ("Q(lam)",       run_q_lambda,     tabular_env),
        ("Linear FA",    run_linear_fa,    tabular_env),
    ]

    for name, runner, env in classical_agents:
        print(f"--- Training {name} ---")
        t0 = time.time()
        try:
            returns = runner(env, cfg)
            results[name] = returns
            elapsed = time.time() - t0
            timing[name] = elapsed
            print(f"    done in {elapsed:.1f}s  "
                  f"final avg: {np.mean(returns[-50:]):.3f}\n")
        except Exception as e:
            print(f"    ERROR: {e}\n")
            results[name] = []

    tabular_env.close()

    # ----------------------------------------------------------------
    # Deep RL methods
    # ----------------------------------------------------------------
    deep_agents = [
        ("REINFORCE",     lambda: run_reinforce(env_factory, cfg)),
        ("A2C",           lambda: run_a2c(env_factory, cfg)),
        ("DQN",           lambda: run_dqn_variant(env_factory, cfg, "dqn", False)),
        ("Double DQN",    lambda: run_dqn_variant(env_factory, cfg, "double", False)),
        ("Dueling DQN",   lambda: run_dqn_variant(env_factory, cfg, "dueling", False)),
        ("DQN (D+D+PER)", lambda: run_dqn_variant(env_factory, cfg, "double_dueling", True)),
        ("PPO",           lambda: run_ppo(env_factory, cfg)),
    ]

    for name, runner in deep_agents:
        print(f"--- Training {name} ---")
        t0 = time.time()
        try:
            returns = runner()
            results[name] = returns
            elapsed = time.time() - t0
            timing[name] = elapsed
            final_avg = np.mean(returns[-20:]) if returns else 0.0
            print(f"    done in {elapsed:.1f}s  "
                  f"episodes={len(returns)}  final avg: {final_avg:.3f}\n")
        except Exception as e:
            print(f"    ERROR: {e}\n")
            results[name] = []

    # Random baseline
    print("--- Random policy baseline ---")
    baseline_env = make_highway_env()
    results["Random"] = run_random_baseline(baseline_env, n_episodes=50)
    baseline_env.close()

    # ----------------------------------------------------------------
    # Save raw results
    # ----------------------------------------------------------------
    summary = {}
    for name, rets in results.items():
        if rets:
            summary[name] = {
                "mean_return": float(np.mean(rets)),
                "std_return":  float(np.std(rets)),
                "n_episodes":  len(rets),
                "final_avg50": float(np.mean(rets[-50:])) if len(rets) >= 50 else float(np.mean(rets)),
                "training_time_s": timing.get(name, 0.0),
            }
        else:
            summary[name] = {"error": "training failed"}

    with open(OUTPUT_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    w = cfg["smooth_window"]

    _plot_group(
        {k: results[k] for k in ["Monte Carlo", "TD(lam)", "SARSA(lam)",
                                   "Q-Learning", "Q(lam)", "Linear FA"]
         if k in results},
        title="Classical (Tabular + Linear FA) Methods - highway-v0",
        save_path=str(OUTPUT_DIR / "01_classical_comparison.png"),
        window=w, xlabel="Episode",
    )

    _plot_group(
        {k: results[k] for k in ["REINFORCE", "A2C", "DQN", "Double DQN",
                                   "Dueling DQN", "DQN (D+D+PER)", "PPO"]
         if k in results},
        title="Deep RL Methods - highway-v0",
        save_path=str(OUTPUT_DIR / "02_deep_comparison.png"),
        window=max(5, w // 3), xlabel="Episode",
    )

    all_keys = list(results.keys())
    _plot_group(
        {k: results[k] for k in all_keys if results[k]},
        title="All Algorithms Comparison - highway-v0",
        save_path=str(OUTPUT_DIR / "03_full_comparison.png"),
        window=w, xlabel="Episode",
    )

    _plot_best_vs_random(results, summary,
                          save_path=str(OUTPUT_DIR / "05_best_vs_random.png"))

    _plot_algorithm_table(save_path=str(OUTPUT_DIR / "04_algorithm_table.png"))

    _plot_final_returns_bar(summary,
                             save_path=str(OUTPUT_DIR / "06_final_returns_bar.png"))

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.  All plots in: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    for name, stats in sorted(summary.items(),
                               key=lambda x: x[1].get("final_avg50", -999),
                               reverse=True):
        if "error" not in stats:
            print(f"  {name:<22} final_avg50={stats['final_avg50']:+.3f}  "
                  f"n_ep={stats['n_episodes']}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_group(data: dict, title: str, save_path: str,
                window: int = 20, xlabel: str = "Episode"):
    n = len(data)
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20b.colors)

    for i, (name, returns) in enumerate(data.items()):
        if not returns:
            continue
        color = COLORS.get(name, colors[i % len(colors)])
        eps = list(range(len(returns)))
        ax.plot(eps, returns, alpha=0.12, color=color)
        if len(returns) >= window:
            smooth = moving_average(returns, window)
            ax.plot(eps[window - 1:], smooth, color=color,
                    linewidth=2, label=name)
        else:
            ax.plot(eps, returns, color=color, linewidth=2, label=name)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Discounted Return", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_best_vs_random(results, summary, save_path: str):
    """Compare best algorithm vs random policy with confidence bands."""
    # Find best algorithm by final_avg50
    best_name = max(
        [k for k in summary if "final_avg50" in summary[k] and k != "Random"],
        key=lambda k: summary[k]["final_avg50"],
        default=None,
    )
    if best_name is None or "Random" not in results:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    for name, color, lw in [(best_name, "#d6604d", 2.5), ("Random", "#999999", 1.5)]:
        rets = results[name]
        if not rets:
            continue
        eps = list(range(len(rets)))
        w = max(10, len(rets) // 20)
        ax.plot(eps, rets, alpha=0.15, color=color)
        if len(rets) >= w:
            sm = moving_average(rets, w)
            ax.plot(eps[w - 1:], sm, color=color, linewidth=lw,
                    label=name if name != "Random" else "Random policy")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Discounted Return")
    ax.set_title(f"Best Agent ({best_name}) vs. Random Policy Baseline")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_final_returns_bar(summary: dict, save_path: str):
    """Horizontal bar chart of final-avg-50 returns per algorithm."""
    names  = [k for k in summary if "final_avg50" in summary[k]]
    values = [summary[k]["final_avg50"] for k in names]
    stds   = [summary[k].get("std_return", 0) for k in names]

    # Sort by value
    order  = np.argsort(values)
    names  = [names[i] for i in order]
    values = [values[i] for i in order]
    stds   = [stds[i]   for i in order]
    colors = [COLORS.get(n, "#aaaaaa") for n in names]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.85)
    ax.errorbar(values, range(len(names)), xerr=stds, fmt="none",
                color="black", capsize=3, linewidth=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Final Avg-50 Discounted Return", fontsize=11)
    ax.set_title("Algorithm Ranking - highway-v0", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_algorithm_table(save_path: str):
    """Printable table: applicability, properties, and notes per algorithm."""
    columns = ["Algorithm", "Class", "Action\nSpace", "On/Off\nPolicy",
               "Model\nFree", "FA Type", "Applicable\nto highway-v0", "Notes"]

    rows = [
        # DP
        ["Policy Iteration",  "DP",        "Discrete", "-",         "No",  "Tabular", "Yes (V1)",    "Requires P, R"],
        ["Value Iteration",   "DP",        "Discrete", "-",         "No",  "Tabular", "Yes (V1)",    "Faster than PI"],
        # Tabular
        ["Monte Carlo",       "Classical", "Discrete", "On",        "Yes", "Tabular", "Yes",          "High variance"],
        ["TD(0)/TD(lam)",     "Classical", "Discrete", "On",        "Yes", "Tabular", "Yes",        "Online, low var"],
        ["SARSA(lam)",        "Classical", "Discrete", "On",        "Yes", "Tabular", "Yes",        "Safe exploration"],
        ["Q-Learning",        "Classical", "Discrete", "Off",       "Yes", "Tabular", "Yes",          "Off-policy TD"],
        ["Q(lam)",              "Classical", "Discrete", "Off",       "Yes", "Tabular", "Yes",          "Trace cutting"],
        ["Linear FA (GradTD)","Classical", "Discrete", "On",        "Yes", "Linear",  "Yes",          "Bridges tabular<->DRL"],
        # Deep RL - applicable
        ["REINFORCE",         "Deep PG",   "Any",      "On",        "Yes", "Neural",  "Yes",          "High variance, no replay"],
        ["A2C",               "Deep PG",   "Any",      "On",        "Yes", "Neural",  "Yes",          "Sync A3C, low overhead"],
        ["A3C",               "Deep PG",   "Any",      "On",        "Yes", "Neural",  "Yes (complex)","Requires async workers"],
        ["DQN",               "Deep RL",   "Discrete", "Off",       "Yes", "Neural",  "Yes",          "Replay + target net"],
        ["Double DQN",        "Deep RL",   "Discrete", "Off",       "Yes", "Neural",  "Yes",          "Reduces overestimation"],
        ["Dueling DQN",       "Deep RL",   "Discrete", "Off",       "Yes", "Neural",  "Yes",          "V+A decomposition"],
        ["PPO",               "Deep PG",   "Any",      "On",        "Yes", "Neural",  "Yes",          "Stable, clipped gradient"],
        ["TRPO",              "Deep PG",   "Any",      "On",        "Yes", "Neural",  "Yes (complex)","Needs natural gradient"],
        # Not applicable
        ["DDPG",              "Deep RL",   "Cont.",    "Off",       "Yes", "Neural",  "No",          "Requires continuous A"],
        ["TD3",               "Deep RL",   "Cont.",    "Off",       "Yes", "Neural",  "No",          "Requires continuous A"],
        ["SAC",               "Deep RL",   "Cont.*",   "Off",       "Yes", "Neural",  "No*",         "Continuous-action focus"],
    ]

    fig, ax = plt.subplots(figsize=(18, len(rows) * 0.55 + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    # Colour header row
    for j in range(len(columns)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colour applicability column and category rows
    cat_colors = {
        "DP":       "#d5e8d4",
        "Classical": "#dae8fc",
        "Deep PG":  "#fff2cc",
        "Deep RL":  "#ffe6cc",
    }
    na_rows = {i for i, r in enumerate(rows) if "No" in r[6]}
    for i, row in enumerate(rows):
        cat = row[1]
        bg = cat_colors.get(cat, "white")
        for j in range(len(columns)):
            cell = table[i + 1, j]
            if i in na_rows:
                cell.set_facecolor("#f4cccc")
            else:
                cell.set_facecolor(bg)
            if j == 6:  # Applicable column
                if "Yes" in row[j] and "complex" not in row[j]:
                    cell.set_text_props(color="#1a7a1a", fontweight="bold")
                elif "No" in row[j]:
                    cell.set_text_props(color="#cc0000", fontweight="bold")
                else:
                    cell.set_text_props(color="#886600")

    ax.set_title(
        "Algorithm Applicability to highway-v0 (Discrete(5) action space)",
        fontsize=12, fontweight="bold", pad=20,
    )

    legend_patches = [
        mpatches.Patch(color="#d5e8d4", label="Dynamic Programming"),
        mpatches.Patch(color="#dae8fc", label="Classical Tabular RL"),
        mpatches.Patch(color="#fff2cc", label="Deep Policy Gradient"),
        mpatches.Patch(color="#ffe6cc", label="Deep RL (value-based)"),
        mpatches.Patch(color="#f4cccc", label="Not applicable (continuous action)"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
