"""Generate saliency / attribution visualisations for a trained DQN agent.

Workflow
--------
1. Train a DQN (double+dueling) agent for a short time on highway-v0.
2. Collect representative observations from an evaluation episode.
3. Apply all four attribution methods to each observation:
       - Vanilla Gradient
       - Gradient x Input
       - SmoothGrad
       - Integrated Gradients
4. Save to outputs/saliency/:
       - 07_saliency_bar_<method>.png       bar chart, top-k features
       - 08_saliency_heatmap_<method>.png   obs matrix vs attribution matrix
       - 09_saliency_methods_comparison.png 4-method side-by-side heatmap
       - 10_ig_completeness_check.png       Integrated-Gradients axiom check
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.envs.highway_wrapper import make_highway_env
from src.agents.deep.dqn import DQNAgent
from src.saliency.gradient_saliency import (
    vanilla_gradient, gradient_x_input, smoothgrad, integrated_gradients,
    check_completeness,
)
from src.saliency.visualization import (
    plot_saliency_bar, plot_obs_heatmap, rasterize,
)

import torch

OUTPUT_DIR = Path("outputs/saliency")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# highway-env feature names (5 vehicles x 5 features)
N_VEH      = 5
FEAT_NAMES = ["presence", "x", "y", "vx", "vy"]
FEAT_LABELS = [f"veh{v}_{f}" for v in range(N_VEH) for f in FEAT_NAMES]
VEH_LABELS  = [f"ego" if v == 0 else f"veh{v}" for v in range(N_VEH)]

METHODS = {
    "Vanilla Gradient":    "vanilla",
    "Gradient x Input":    "gxi",
    "SmoothGrad":          "smoothgrad",
    "Integrated Gradients":"ig",
}


# ---------------------------------------------------------------------------
# Step 1 — train a DQN agent
# ---------------------------------------------------------------------------

def train_agent(n_steps: int = 2000, max_ep_steps: int = 60):
    print(f"\n[1/3] Training DQN (double+dueling) for {n_steps} steps ...")
    env = make_highway_env(normalize_obs=True)
    obs_dim   = env.obs_dim
    n_actions = env.n_actions

    agent = DQNAgent(
        obs_dim=obs_dim, n_actions=n_actions,
        variant="double_dueling",
        gamma=0.99, lr=1e-4,
        batch_size=32, buffer_capacity=1000,
        target_update_freq=100,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=1500,
        use_per=False, train_start=100,
    )
    t0 = time.time()
    result = agent.train(env, n_steps=n_steps, max_ep_steps=max_ep_steps,
                         verbose_every=0)
    env.close()
    n_ep = len(result["returns"])
    avg  = float(np.mean(result["returns"][-10:])) if n_ep else 0.0
    print(f"    done: {n_ep} episodes  final-avg={avg:+.2f}  ({time.time()-t0:.0f}s)")
    return agent, obs_dim, n_actions


# ---------------------------------------------------------------------------
# Step 2 — collect observations from an evaluation episode
# ---------------------------------------------------------------------------

def collect_obs(agent: DQNAgent, n_obs: int = 8, max_steps: int = 60):
    print(f"\n[2/3] Collecting {n_obs} representative observations ...")
    env = make_highway_env(normalize_obs=True)
    collected = []
    actions_taken = []

    obs, _ = env.reset()
    for _ in range(max_steps):
        a = agent.select_action(obs, training=False)
        collected.append(obs.copy())
        actions_taken.append(a)
        obs, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated or len(collected) >= n_obs:
            break
    env.close()

    # Evenly spaced subset
    idxs = np.linspace(0, len(collected)-1, min(n_obs, len(collected)),
                       dtype=int)
    print(f"    collected {len(collected)} steps; using indices {idxs.tolist()}")
    return [collected[i] for i in idxs], [actions_taken[i] for i in idxs]


# ---------------------------------------------------------------------------
# Step 3 — compute all four attributions for one observation
# ---------------------------------------------------------------------------

def compute_all_attributions(network, obs: np.ndarray, action: int,
                              device: torch.device):
    attrs = {}
    attrs["Vanilla Gradient"]     = vanilla_gradient(network, obs, action, device)
    attrs["Gradient x Input"]     = gradient_x_input(network, obs, action, device)
    attrs["SmoothGrad"]           = smoothgrad(network, obs, action,
                                               n_samples=30, noise_level=0.1,
                                               device=device)
    attrs["Integrated Gradients"] = integrated_gradients(network, obs, action,
                                                          n_steps=50, device=device)
    return attrs


# ---------------------------------------------------------------------------
# Step 4 — generate all plots
# ---------------------------------------------------------------------------

def make_bar_charts(attrs: dict, action: int, step: int):
    """One bar chart per method for a single observation."""
    for method_name, attr in attrs.items():
        key = METHODS[method_name]
        fig = plot_saliency_bar(
            attr.flatten(),
            feature_labels=FEAT_LABELS,
            title=f"{method_name}  |  action={action}  step={step}",
            top_k=15,
        )
        path = str(OUTPUT_DIR / f"07_saliency_bar_{key}_step{step:02d}.png")
        rasterize(fig, path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {Path(path).name}")


def make_heatmaps(obs: np.ndarray, attrs: dict, action: int, step: int):
    """One heatmap per method for a single observation."""
    for method_name, attr in attrs.items():
        key = METHODS[method_name]
        fig = plot_obs_heatmap(
            obs, attr,
            feature_names=FEAT_NAMES,
            vehicle_labels=VEH_LABELS,
            title=f"{method_name}  |  action={action}  step={step}",
        )
        path = str(OUTPUT_DIR / f"08_saliency_heatmap_{key}_step{step:02d}.png")
        rasterize(fig, path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {Path(path).name}")


def make_comparison_grid(obs: np.ndarray, attrs: dict, action: int, step: int):
    """2x2 grid comparing all four attribution methods side by side."""
    n_feat = len(FEAT_NAMES)
    n_veh  = obs.size // n_feat

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Attribution Methods Comparison  |  action={action}  step={step}",
                 fontsize=13, fontweight="bold")

    for ax, (method_name, attr) in zip(axes.flat, attrs.items()):
        attr_2d = attr.reshape(n_veh, n_feat)
        vmax = np.abs(attr_2d).max() + 1e-8
        im = ax.imshow(attr_2d, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(method_name, fontsize=10, fontweight="bold")
        ax.set_xticks(range(n_feat))
        ax.set_xticklabels(FEAT_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(n_veh))
        ax.set_yticklabels(VEH_LABELS, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    path = str(OUTPUT_DIR / f"09_saliency_methods_comparison_step{step:02d}.png")
    rasterize(fig, path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {Path(path).name}")


def make_ig_completeness_plot(network, obs: np.ndarray, action: int,
                               step: int, device: torch.device):
    """Bar chart showing IG completeness axiom: sum(attr) vs delta_F."""
    ig_attr = integrated_gradients(network, obs, action, n_steps=50,
                                    device=device)
    result  = check_completeness(network, obs, action, ig_attr, device=device)

    fig, ax = plt.subplots(figsize=(6, 3))
    vals   = [result["sum_attr"], result["delta_F"]]
    labels = ["Sum of\nAttributions", "F(input) -\nF(baseline)"]
    colors = ["#4393c3", "#d6604d"]
    bars = ax.bar(labels, vals, color=colors, width=0.4, alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(
        f"Integrated Gradients — Completeness Axiom  (step {step})\n"
        f"Relative error = {result['error']*100:.2f}%  "
        f"({'PASS' if result['passed'] else 'FAIL'})",
        fontsize=10,
    )
    ax.set_ylabel("Value")
    plt.tight_layout()
    path = str(OUTPUT_DIR / f"10_ig_completeness_step{step:02d}.png")
    rasterize(fig, path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {Path(path).name}  completeness={'PASS' if result['passed'] else 'FAIL'}")


# ---------------------------------------------------------------------------
# Summary bar: avg attribution magnitude across all observations
# ---------------------------------------------------------------------------

def make_summary_attribution(network, all_obs: list, all_actions: list,
                              device: torch.device):
    """Average |attribution| per feature across all collected observations."""
    print("\n    Computing averaged attribution across all observations ...")
    method_totals = {m: np.zeros(len(FEAT_LABELS)) for m in METHODS}

    for obs, action in zip(all_obs, all_actions):
        attrs = compute_all_attributions(network, obs, action, device)
        for method_name, attr in attrs.items():
            method_totals[method_name] += np.abs(attr.flatten())

    n = len(all_obs)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Average |Attribution| per Feature (across episode)",
                 fontsize=13, fontweight="bold")

    for ax, (method_name, total) in zip(axes.flat, method_totals.items()):
        avg = total / n
        order = np.argsort(avg)[-15:]
        colors = ["#4393c3"] * len(order)
        ax.barh(range(len(order)), avg[order], color=colors, alpha=0.85)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([FEAT_LABELS[i] for i in order], fontsize=8)
        ax.set_title(method_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("|Attribution|")
        ax.axvline(0, color="black", linewidth=0.6)

    plt.tight_layout()
    path = str(OUTPUT_DIR / "11_avg_attribution_summary.png")
    rasterize(fig, path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {Path(path).name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    agent, obs_dim, n_actions = train_agent(n_steps=2000, max_ep_steps=60)
    network = agent.online_net
    device  = agent.device

    all_obs, all_actions = collect_obs(agent, n_obs=8)

    print(f"\n[3/3] Generating saliency plots ...")

    # Per-step plots for the first 3 observations
    for step_i, (obs, action) in enumerate(zip(all_obs[:3], all_actions[:3])):
        print(f"\n  -- step {step_i} (action={action}) --")
        attrs = compute_all_attributions(network, obs, action, device)
        make_bar_charts(attrs, action, step_i)
        make_heatmaps(obs, attrs, action, step_i)
        make_comparison_grid(obs, attrs, action, step_i)
        make_ig_completeness_plot(network, obs, action, step_i, device)

    # Summary across all 8 observations
    make_summary_attribution(network, all_obs, all_actions, device)

    # Print final file list
    files = sorted(OUTPUT_DIR.glob("*.png"))
    print(f"\n=== Saliency complete.  {len(files)} plots in {OUTPUT_DIR} ===")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
