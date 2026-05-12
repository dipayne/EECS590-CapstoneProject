"""Multi-seed robustness comparison for V3 Optuna-tuned DQN vs V2 baseline.

Re-trains the V2 default DQN(D+D+PER) config and the V3 Optuna-tuned best
config over 5 seeds each. Confirms whether the tuned gain is robust to
random initialization or just seed luck.

Outputs (under V3/outputs/robustness/):
    returns_by_seed.json     per-seed return curves for both configs
    summary.json             mean/std/min/max per config
    learning_curves.png      side-by-side curves with mean +/- std bands
    final_avg50_box.png      boxplot of final-avg-50 across seeds

Run:
    python V3/run_robustness.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
V2_DIR = THIS_DIR.parent / "V2"
sys.path.insert(0, str(V2_DIR))

from src.envs.highway_wrapper import make_highway_env   # noqa: E402
from src.agents.deep.dqn import DQNAgent                # noqa: E402


OUT_DIR = THIS_DIR / "outputs" / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STEPS = 20_000
MAX_EP_STEPS = 200
SEEDS = [0, 1, 2, 3, 4]

# V2 ablation default for the full D+D+PER config (from V2/run_ablation.py)
V2_DEFAULT_HP = {
    "gamma": 0.99,
    "lr": 1e-4,
    "batch_size": 64,
    "target_update_freq": 300,
    "eps_decay_pct": 0.7,
    "train_start": 500,
}

# V3 Optuna-tuned best (from V3/outputs/optuna/best_params.json, trial 9)
_BEST_PATH = THIS_DIR / "outputs" / "optuna" / "best_params.json"
if _BEST_PATH.exists():
    V3_TUNED_HP = json.loads(_BEST_PATH.read_text())["best_params"]
else:
    raise FileNotFoundError(
        f"{_BEST_PATH} missing — run run_optuna.py first."
    )


CONFIGS = [
    {"label": "V2 default",    "hp": V2_DEFAULT_HP, "color": "#1b7837"},
    {"label": "V3 tuned (Optuna)", "hp": V3_TUNED_HP,  "color": "#d6604d"},
]


def train_one(seed: int, hp: dict) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_highway_env(normalize_obs=True)
    agent = DQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        variant="double_dueling",
        gamma=hp["gamma"],
        lr=hp["lr"],
        batch_size=int(hp["batch_size"]),
        buffer_capacity=min(20_000, N_STEPS),
        target_update_freq=int(hp["target_update_freq"]),
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=max(1, int(N_STEPS * hp["eps_decay_pct"])),
        use_per=True,
        train_start=int(hp["train_start"]),
    )
    t0 = time.time()
    result = agent.train(env, n_steps=N_STEPS, max_ep_steps=MAX_EP_STEPS,
                         verbose_every=0)
    env.close()
    elapsed = time.time() - t0
    rets = result["returns"]
    avg50 = float(np.mean(rets[-50:])) if len(rets) >= 50 else float(np.mean(rets))
    return {
        "returns": rets,
        "final_avg50": avg50,
        "n_episodes": len(rets),
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 64)
    print(f"  Robustness study  ({len(CONFIGS)} configs x {len(SEEDS)} seeds x "
          f"{N_STEPS} steps)")
    print("=" * 64)

    all_data: dict = {}
    overall_start = time.time()

    for cfg in CONFIGS:
        label = cfg["label"]
        print(f"\n--- {label} ---")
        print(f"  HP: {cfg['hp']}")
        runs = {}
        for seed in SEEDS:
            res = train_one(seed, cfg["hp"])
            runs[str(seed)] = res
            print(f"  seed={seed}  avg50={res['final_avg50']:+.3f}  "
                  f"eps={res['n_episodes']:>4}  ({res['elapsed_s']:.0f}s)")
        avg50s = [v["final_avg50"] for v in runs.values()]
        all_data[label] = {
            "runs": runs,
            "hp": cfg["hp"],
            "color": cfg["color"],
            "mean_avg50": float(np.mean(avg50s)),
            "std_avg50": float(np.std(avg50s)),
            "min_avg50": float(np.min(avg50s)),
            "max_avg50": float(np.max(avg50s)),
        }

    total = time.time() - overall_start
    print(f"\n  Total wall: {total/60:.1f} min")

    # Save raw returns (full per-episode curves are large but useful)
    returns_path = OUT_DIR / "returns_by_seed.json"
    payload = {
        label: {seed: data["runs"][seed]["returns"]
                for seed in data["runs"]}
        for label, data in all_data.items()
    }
    returns_path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved: {returns_path.name}")

    # Save summary (without big return arrays)
    summary_path = OUT_DIR / "summary.json"
    summary = {}
    for label, data in all_data.items():
        summary[label] = {
            "hp": data["hp"],
            "mean_avg50": data["mean_avg50"],
            "std_avg50": data["std_avg50"],
            "min_avg50": data["min_avg50"],
            "max_avg50": data["max_avg50"],
            "per_seed": {seed: {"final_avg50": data["runs"][seed]["final_avg50"],
                                "n_episodes": data["runs"][seed]["n_episodes"],
                                "elapsed_s": data["runs"][seed]["elapsed_s"]}
                         for seed in data["runs"]},
        }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {summary_path.name}")

    # Plots
    plot_curves(all_data)
    plot_box(all_data)

    # Ranking print
    print("\n  --- Robustness summary ---")
    print(f"  {'Config':<25} {'mean+/-std':>16}  {'[min, max]':>18}")
    for label, data in all_data.items():
        print(f"  {label:<25}  {data['mean_avg50']:+.3f} +/- {data['std_avg50']:.3f}  "
              f"[{data['min_avg50']:+.3f}, {data['max_avg50']:+.3f}]")


def plot_curves(all_data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = 15

    def moving_average(x, w):
        x = np.asarray(x, dtype=float)
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for label, data in all_data.items():
        color = data["color"]
        all_curves = []
        for seed, run in data["runs"].items():
            rets = run["returns"]
            sm = moving_average(rets, window)
            ax.plot(range(len(sm)), sm, color=color, alpha=0.25, linewidth=1)
            all_curves.append(sm)
        # Pad to common length (min) so we can stack
        min_len = min(len(c) for c in all_curves)
        stacked = np.stack([c[:min_len] for c in all_curves])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        x = np.arange(min_len)
        ax.plot(x, mean, color=color, linewidth=2.4,
                label=f"{label}  (mean over {len(all_curves)} seeds)")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(f"Smoothed Return (window={window})", fontsize=11)
    ax.set_title("V2 default vs V3 Optuna-tuned — 5-seed learning curves",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "learning_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_box(all_data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(all_data.keys())
    data = [[run["final_avg50"] for run in all_data[lbl]["runs"].values()]
            for lbl in labels]
    colors = [all_data[lbl]["color"] for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    # Scatter raw points
    for i, (lbl, c) in enumerate(zip(labels, colors), start=1):
        vals = [run["final_avg50"] for run in all_data[lbl]["runs"].values()]
        ax.scatter([i + np.random.uniform(-0.06, 0.06) for _ in vals], vals,
                   color="black", alpha=0.7, s=22, zorder=3)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final Avg-50 Return", fontsize=11)
    ax.set_title("V3 robustness — final return across seeds",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "final_avg50_box.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


if __name__ == "__main__":
    main()
