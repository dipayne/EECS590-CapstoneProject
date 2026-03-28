"""Combine per-config ablation results and generate the final plot.

Run this after all 5 run_ablation_single.py jobs have finished.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR  = Path("outputs/comparison")
SUMMARY_PATH = OUTPUT_DIR / "results_summary.json"

CONFIGS = [
    {"label": "DQN (baseline)",        "color": "#1b7837"},
    {"label": "DQN + Double",           "color": "#762a83"},
    {"label": "DQN + Dueling",          "color": "#e08214"},
    {"label": "DQN + Double + Dueling", "color": "#4393c3"},
    {"label": "DQN + D+D+PER (full)",   "color": "#d6604d"},
]


def moving_average(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    # Load per-config results
    all_results = {}
    for idx, cfg in enumerate(CONFIGS):
        path = OUTPUT_DIR / f"ablation_config_{idx}.json"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping {cfg['label']}")
            continue
        data = json.load(open(path))
        all_results[cfg["label"]] = data
        print(f"  Loaded: {cfg['label']}  final-avg50={data['final_avg50']:+.3f}  "
              f"episodes={data['n_episodes']}  ({data['training_time_s']:.0f}s)")

    if not all_results:
        print("No results found. Run run_ablation_single.py for each config first.")
        return

    # Generate plots
    window = 15
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for cfg in CONFIGS:
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

    ax2 = axes[1]
    labels = [cfg["label"] for cfg in CONFIGS if cfg["label"] in all_results]
    avg50s = [all_results[l]["final_avg50"] for l in labels]
    colors = [cfg["color"] for cfg in CONFIGS if cfg["label"] in all_results]

    bars = ax2.barh(range(len(labels)), avg50s, color=colors, alpha=0.85)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Final Avg-50 Discounted Return", fontsize=11)
    ax2.set_title("DQN Ablation — Component Contributions", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, avg50s):
        ax2.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.2f}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "07_dqn_ablation.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    # Save combined raw returns
    with open(OUTPUT_DIR / "ablation_returns.json", "w") as f:
        json.dump({k: v["returns"] for k, v in all_results.items()}, f, indent=2)

    # Update results summary
    if SUMMARY_PATH.exists():
        summary = json.load(open(SUMMARY_PATH))
    else:
        summary = {}
    for label, data in all_results.items():
        summary[label] = {k: v for k, v in data.items() if k != "returns"}
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Updated: {SUMMARY_PATH.name}")

    # Print ranking
    print("\nAblation Results:")
    print(f"  {'Config':<30} {'Final Avg-50':>14}  {'Episodes':>9}")
    print("  " + "-" * 58)
    prev = None
    for cfg in CONFIGS:
        lbl = cfg["label"]
        if lbl not in all_results:
            continue
        d = all_results[lbl]
        gain = f"  gain={d['final_avg50'] - prev:+.3f}" if prev is not None else ""
        print(f"  {lbl:<30} {d['final_avg50']:>+14.3f}  {d['n_episodes']:>9}{gain}")
        prev = d["final_avg50"]


if __name__ == "__main__":
    main()
