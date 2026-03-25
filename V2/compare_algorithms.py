"""Algorithm comparison: overlay training curves from multiple experiments.

Reads episode_log.json files produced by train_classical.py / train_dqn.py /
train_ppo.py and plots them on a single figure for side-by-side comparison.

Usage
-----
    python compare_algorithms.py \
        --logs outputs/dqn/double_dueling_<run>/episode_log.json \
                outputs/ppo/ppo_<run>/episode_log.json \
                outputs/classical/sarsa_sarsa_lam_backward/returns.npy \
        --labels "DQN (Double+Dueling)" "PPO" "SARSA(lam)" \
        --out outputs/comparison.png

Arguments accept both .json (from logger) and .npy (raw returns array).
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs",   nargs="+", required=True,
                   help="Paths to episode_log.json or returns.npy files")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Display name for each log (must match --logs count)")
    p.add_argument("--out",    default="outputs/comparison.png")
    p.add_argument("--window", type=int, default=20,
                   help="Moving-average smoothing window")
    p.add_argument("--title",  default="Algorithm Comparison — Episode Returns")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Truncate all curves to this many episodes")
    return p.parse_args()


def load_returns(path: str):
    """Load episode returns from a .json or .npy file."""
    if path.endswith(".npy"):
        return np.load(path).tolist()
    with open(path) as f:
        data = json.load(f)
    # Support both list-of-dicts and list-of-floats
    if isinstance(data, list) and isinstance(data[0], dict):
        return [r.get("return", r.get("episode_return", 0.0)) for r in data]
    return [float(x) for x in data]


def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    args = parse_args()

    n = len(args.logs)
    labels = args.labels or [os.path.basename(p) for p in args.logs]
    assert len(labels) == n, "Number of --labels must match --logs"

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (path, label) in enumerate(zip(args.logs, labels)):
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue

        returns = load_returns(path)
        if args.max_episodes:
            returns = returns[: args.max_episodes]

        color = colors[i % len(colors)]
        eps = list(range(len(returns)))

        ax.plot(eps, returns, alpha=0.15, color=color)

        w = args.window
        if len(returns) >= w:
            smooth = moving_average(returns, w)
            ax.plot(eps[w - 1:], smooth, color=color, linewidth=2, label=label)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Discounted Return", fontsize=12)
    ax.set_title(args.title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to {args.out}")


if __name__ == "__main__":
    main()
