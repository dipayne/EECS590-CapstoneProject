"""DQN Component Ablation Study.

Systematically isolates the contribution of each DQN enhancement:

  Config 1: DQN baseline          (no double, no dueling, no PER)
  Config 2: DQN + Double          (reduces Q-value overestimation)
  Config 3: DQN + Dueling         (separate value / advantage streams)
  Config 4: DQN + Double + Dueling
  Config 5: DQN + Double + Dueling + PER  (full D+D+PER)

All configs share identical hyperparameters. Results saved to
outputs/comparison/07_dqn_ablation.png and appended to results_summary.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.envs.highway_wrapper import make_highway_env
from src.agents.deep.dqn import DQNAgent

OUTPUT_DIR = Path("outputs/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUTPUT_DIR / "results_summary.json"

N_STEPS   = 20_000
MAX_STEPS = 200
SEED      = 42

ABLATION_CONFIGS = [
    {
        "label":   "DQN (baseline)",
        "variant": "dqn",
        "use_per": False,
        "color":   "#1b7837",
    },
    {
        "label":   "DQN + Double",
        "variant": "double",
        "use_per": False,
        "color":   "#762a83",
    },
    {
        "label":   "DQN + Dueling",
        "variant": "dueling",
        "use_per": False,
        "color":   "#e08214",
    },
    {
        "label":   "DQN + Double + Dueling",
        "variant": "double_dueling",
        "use_per": False,
        "color":   "#4393c3",
    },
    {
        "label":   "DQN + D+D+PER (full)",
        "variant": "double_dueling",
        "use_per": True,
        "color":   "#d6604d",
    },
]


def moving_average(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def train_variant(cfg: dict) -> tuple[list, float]:
    np.random.seed(SEED)
    env = make_highway_env(normalize_obs=True)
    agent = DQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        variant=cfg["variant"],
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        buffer_capacity=min(20_000, N_STEPS),
        target_update_freq=300,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=int(N_STEPS * 0.7),
        use_per=cfg["use_per"],
        train_start=500,
    )
    t0 = time.time()
    result = agent.train(env, n_steps=N_STEPS, max_ep_steps=MAX_STEPS,
                         verbose_every=5000)
    env.close()
    elapsed = time.time() - t0
    rets = result["returns"]
    avg50 = float(np.mean(rets[-50:])) if len(rets) >= 50 else float(np.mean(rets))
    print(f"  {cfg['label']:<28}  eps={len(rets):>4}  "
          f"final-avg50={avg50:+.3f}  ({elapsed:.0f}s)")
    return rets, elapsed


def plot_ablation(all_results: dict) -> None:
    window = 15
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --- left: learning curves ---
    ax = axes[0]
    for cfg in ABLATION_CONFIGS:
        label = cfg["label"]
        if label not in all_results:
            continue
        rets  = all_results[label]["returns"]
        color = cfg["color"]
        eps   = list(range(len(rets)))
        ax.plot(eps, rets, alpha=0.10, color=color)
        if len(rets) >= window:
            sm = moving_average(rets, window)
            ax.plot(eps[window - 1:], sm, color=color, linewidth=2, label=label)
        else:
            ax.plot(eps, rets, color=color, linewidth=2, label=label)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Discounted Return", fontsize=11)
    ax.set_title("DQN Ablation — Learning Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # --- right: bar chart of final-avg-50 ---
    ax2 = axes[1]
    labels   = [cfg["label"] for cfg in ABLATION_CONFIGS if cfg["label"] in all_results]
    avg50s   = [all_results[l]["final_avg50"] for l in labels]
    colors   = [cfg["color"] for cfg in ABLATION_CONFIGS if cfg["label"] in all_results]

    bars = ax2.barh(range(len(labels)), avg50s, color=colors, alpha=0.85)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Final Avg-50 Discounted Return", fontsize=11)
    ax2.set_title("DQN Ablation — Component Contributions", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    # Annotate incremental gains
    for i, (bar, val) in enumerate(zip(bars, avg50s)):
        ax2.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.2f}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "07_dqn_ablation.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    print("=" * 60)
    print("  DQN Ablation Study  (5 configs x 20,000 steps each)")
    print("=" * 60)

    all_results: dict = {}

    for cfg in ABLATION_CONFIGS:
        print(f"\n--- {cfg['label']} ---")
        rets, elapsed = train_variant(cfg)
        all_results[cfg["label"]] = {
            "returns":      rets,
            "final_avg50":  float(np.mean(rets[-50:])) if len(rets) >= 50 else float(np.mean(rets)),
            "mean_return":  float(np.mean(rets)),
            "std_return":   float(np.std(rets)),
            "n_episodes":   len(rets),
            "training_time_s": elapsed,
        }

    # Save raw returns for reproducibility
    ablation_returns_path = OUTPUT_DIR / "ablation_returns.json"
    with open(ablation_returns_path, "w") as f:
        json.dump({k: v["returns"] for k, v in all_results.items()}, f, indent=2)
    print(f"\n  Saved raw returns: {ablation_returns_path.name}")

    # Generate plots
    print("\nGenerating ablation plots ...")
    plot_ablation(all_results)

    # Append to results summary
    if SUMMARY_PATH.exists():
        summary = json.load(open(SUMMARY_PATH))
    else:
        summary = {}

    for label, data in all_results.items():
        summary[label] = {k: v for k, v in data.items() if k != "returns"}

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Updated: {SUMMARY_PATH.name}")

    # Print ranking table
    print("\nAblation Results:")
    print(f"  {'Config':<30} {'Final Avg-50':>14}  {'Episodes':>9}  {'Time(s)':>8}")
    print("  " + "-" * 68)
    for cfg in ABLATION_CONFIGS:
        lbl = cfg["label"]
        if lbl not in all_results:
            continue
        d = all_results[lbl]
        print(f"  {lbl:<30} {d['final_avg50']:>+14.3f}  {d['n_episodes']:>9}  {d['training_time_s']:>8.0f}")

    # Compute incremental gains
    print("\nIncremental component gains:")
    prev = None
    for cfg in ABLATION_CONFIGS:
        lbl = cfg["label"]
        if lbl not in all_results:
            continue
        cur = all_results[lbl]["final_avg50"]
        if prev is not None:
            gain = cur - prev
            print(f"  {lbl:<30} gain = {gain:+.3f}")
        else:
            print(f"  {lbl:<30} baseline = {cur:+.3f}")
        prev = cur

    print("\nDone. Ablation plot saved to outputs/comparison/07_dqn_ablation.png")


if __name__ == "__main__":
    main()
