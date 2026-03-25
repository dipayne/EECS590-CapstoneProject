"""Train classical RL agents on highway-v0 (tabular state abstraction).

Usage
-----
    python train_classical.py --algo sarsa --mode sarsa_lam_backward --episodes 2000
    python train_classical.py --algo q_learning --double --episodes 2000
    python train_classical.py --algo monte_carlo --episodes 1000
    python train_classical.py --algo td --mode tdlam_backward --episodes 2000

All agents use the V1 tabular state abstraction (135 states) built from
the highway-env Kinematics observation.

Outputs
-------
  outputs/classical/<algo>_<mode>/
      Q.npy            — final Q-table  (nS × nA)
      returns.npy      — per-episode discounted returns
      returns.png      — training curve
      hparams.json     — hyperparameters for this run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- path setup ----
sys.path.insert(0, os.path.dirname(__file__))
V1_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, V1_SRC)

from src.envs.highway_wrapper import make_highway_env
from src.agents.classical import MonteCarloAgent, TDAgent, SARSAAgent, QLearningAgent


def parse_args():
    p = argparse.ArgumentParser(description="Train classical RL on highway-v0")
    p.add_argument("--algo", choices=["monte_carlo", "td", "sarsa", "q_learning"],
                   default="sarsa", help="Algorithm family")
    p.add_argument("--mode", default=None,
                   help="Variant within family (e.g. sarsa_lam_backward)")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-decay", type=float, default=0.999)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--n", type=int, default=5,
                   help="n-step depth for TD(n) / SARSA(n)")
    p.add_argument("--double", action="store_true",
                   help="Use Double Q-Learning (q_learning only)")
    p.add_argument("--trace", choices=["accumulating", "replacing"],
                   default="accumulating")
    p.add_argument("--verbose-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_agent(args, nS: int, nA: int, obs_to_state):
    """Factory: construct the correct classical agent from CLI args."""
    common = dict(
        nS=nS, nA=nA, obs_to_state=obs_to_state,
        gamma=args.gamma, epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min,
    )

    if args.algo == "monte_carlo":
        mode = args.mode  # None → default first_visit=True
        return MonteCarloAgent(**common,
                               first_visit=(mode != "every_visit")), "monte_carlo"

    if args.algo == "td":
        mode = args.mode or "tdlam_backward"
        return TDAgent(**common, alpha=args.alpha, n=args.n, lam=args.lam,
                       mode=mode, trace_type=args.trace), mode

    if args.algo == "sarsa":
        mode = args.mode or "sarsa_lam_backward"
        return SARSAAgent(**common, alpha=args.alpha, n=args.n, lam=args.lam,
                          mode=mode, trace_type=args.trace), mode

    if args.algo == "q_learning":
        agent = QLearningAgent(**common, alpha=args.alpha, double=args.double)
        mode = "double_q" if args.double else "q_learning"
        return agent, mode

    raise ValueError(f"Unknown algo: {args.algo}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # ---- environment ----
    env = make_highway_env()
    nS = env.n_tabular_states   # 135
    nA = env.n_actions          # 5

    obs_to_state = env.tabular_state

    # ---- agent ----
    agent, label = make_agent(args, nS, nA, obs_to_state)

    # ---- output directory ----
    tag = f"{args.algo}_{label}"
    out_dir = os.path.join("outputs", "classical", tag)
    os.makedirs(out_dir, exist_ok=True)

    hparams = vars(args)
    hparams["n_states"] = nS
    hparams["n_actions"] = nA
    with open(os.path.join(out_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    print(f"\n=== Training {tag} for {args.episodes} episodes ===")
    print(f"    nS={nS}  nA={nA}  gamma={args.gamma}  eps₀={args.epsilon}")

    # ---- train ----
    if args.algo == "monte_carlo":
        returns = agent.train(env, args.episodes, args.max_steps,
                              verbose_every=args.verbose_every)
    else:
        returns = agent.train(env, args.episodes, args.max_steps,
                              verbose_every=args.verbose_every)

    env.close()

    # ---- save results ----
    np.save(os.path.join(out_dir, "returns.npy"), np.array(returns))

    if hasattr(agent, "Q"):
        np.save(os.path.join(out_dir, "Q.npy"), agent.Q)
    if hasattr(agent, "V"):
        np.save(os.path.join(out_dir, "V.npy"), agent.V)

    # ---- training curve ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(returns, alpha=0.3, color="steelblue", label="raw")
    window = 50
    if len(returns) >= window:
        smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(returns)), smooth,
                color="steelblue", linewidth=2, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"{tag} on highway-v0")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "returns.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    final_avg = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
    print(f"\n=== Done.  Final avg-100 return: {final_avg:.3f} ===")
    print(f"    Results saved to {out_dir}")


if __name__ == "__main__":
    main()
