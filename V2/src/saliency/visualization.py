"""Saliency-map visualisation utilities.

Functions
---------
plot_saliency_bar      — horizontal bar chart of per-feature attributions
plot_obs_heatmap       — 2-D heatmap overlaid on the vehicle observation matrix
plot_saliency_episode  — run an episode and save per-step saliency frames
rasterize              — save any matplotlib figure as a PNG at a given DPI
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, List


# Feature name helpers for highway-env Kinematics observations
_N_VEHICLES = 5
_FEATURES    = ["presence", "x", "y", "vx", "vy"]
_FEATURE_LABELS = [
    f"veh{v}_{f}" for v in range(_N_VEHICLES) for f in _FEATURES
]


# ---------------------------------------------------------------------------
# 1. Bar chart of per-feature attributions
# ---------------------------------------------------------------------------

def plot_saliency_bar(
    attribution: np.ndarray,
    feature_labels: Optional[List[str]] = None,
    title: str = "Feature Attribution",
    top_k: int = 20,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Horizontal bar chart of the top-k most attributed features.

    Parameters
    ----------
    attribution:
        1-D array of attribution values (same length as flattened obs).
    feature_labels:
        Optional list of feature name strings.  Defaults to highway-env
        kinematics feature names (5 vehicles × 5 features).
    title:
        Plot title.
    top_k:
        Number of top features (by |attribution|) to display.
    save_path:
        If provided, save the figure to this path.
    dpi:
        Output resolution (used if save_path is set).
    """
    flat_attr = attribution.flatten()
    labels = feature_labels or _FEATURE_LABELS[: len(flat_attr)]

    # Sort by absolute magnitude
    order = np.argsort(np.abs(flat_attr))[-top_k:]
    values = flat_attr[order]
    names  = [labels[i] if i < len(labels) else f"feat_{i}" for i in order]

    colors = ["tab:red" if v < 0 else "tab:blue" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, top_k * 0.35)))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Attribution value")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        rasterize(fig, save_path, dpi=dpi)

    return fig


# ---------------------------------------------------------------------------
# 2. 2-D heatmap of vehicle × feature attribution
# ---------------------------------------------------------------------------

def plot_obs_heatmap(
    obs: np.ndarray,
    attribution: np.ndarray,
    feature_names: Optional[List[str]] = None,
    vehicle_labels: Optional[List[str]] = None,
    title: str = "Saliency Heatmap",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Visualise attribution as a heatmap over the obs matrix.

    Expects obs and attribution to be reshapable to (n_vehicles, n_features).

    Parameters
    ----------
    obs:
        Raw observation matrix (any shape — will be reshaped).
    attribution:
        Attribution map of the same size as *obs*.
    """
    feat_names = feature_names or _FEATURES
    n_feat = len(feat_names)
    n_veh = obs.size // n_feat
    veh_labels = vehicle_labels or [f"veh{i}" for i in range(n_veh)]

    attr_2d = attribution.reshape(n_veh, n_feat)
    obs_2d  = obs.reshape(n_veh, n_feat)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(3, n_veh * 0.8)))

    # Raw observation values
    im0 = axes[0].imshow(obs_2d, aspect="auto", cmap="viridis")
    axes[0].set_title("Observation")
    axes[0].set_xticks(range(n_feat))
    axes[0].set_xticklabels(feat_names, rotation=30, ha="right", fontsize=8)
    axes[0].set_yticks(range(n_veh))
    axes[0].set_yticklabels(veh_labels, fontsize=8)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Attribution heatmap with diverging colourmap (red=negative, blue=positive)
    vmax = np.abs(attr_2d).max() + 1e-8
    im1 = axes[1].imshow(attr_2d, aspect="auto",
                          cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Attribution (saliency)")
    axes[1].set_xticks(range(n_feat))
    axes[1].set_xticklabels(feat_names, rotation=30, ha="right", fontsize=8)
    axes[1].set_yticks(range(n_veh))
    axes[1].set_yticklabels(veh_labels, fontsize=8)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        rasterize(fig, save_path, dpi=dpi)

    return fig


# ---------------------------------------------------------------------------
# 3. Episode-level saliency animation (save frames)
# ---------------------------------------------------------------------------

def plot_saliency_episode(
    network,
    env,
    attribution_fn,
    action: int,
    out_dir: str,
    max_steps: int = 100,
    dpi: int = 100,
    device=None,
) -> int:
    """Run one episode and save a saliency heatmap PNG per step.

    Parameters
    ----------
    network:
        Neural network for which attributions are computed.
    env:
        Gymnasium environment (unwrapped; observation is 2-D matrix).
    attribution_fn:
        A callable matching the signature of the functions in
        ``gradient_saliency.py``, e.g. ``integrated_gradients``.
    action:
        The action index to attribute.
    out_dir:
        Directory where per-step PNG frames are saved.
    max_steps:
        Maximum episode length.

    Returns
    -------
    int
        Number of frames saved.
    """
    import torch
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cpu")

    obs, _ = env.reset()
    frame = 0
    for step in range(max_steps):
        attr = attribution_fn(network, obs, action, device=device)
        fig = plot_obs_heatmap(
            obs, attr,
            title=f"Step {step}  action={action}",
        )
        rasterize(fig, str(out_path / f"frame_{step:04d}.png"), dpi=dpi)
        plt.close(fig)
        frame += 1

        a = int(network(
            torch.tensor(obs.flatten(), dtype=torch.float32,
                         device=device).unsqueeze(0)
        ).argmax(dim=1).item())
        obs, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            break

    return frame


# ---------------------------------------------------------------------------
# 4. Rasterization helper
# ---------------------------------------------------------------------------

def rasterize(
    fig: plt.Figure,
    path: str,
    dpi: int = 150,
    bbox_inches: str = "tight",
) -> str:
    """Save *fig* to *path* as a PNG raster image.

    Creates parent directories as needed.  Returns the absolute path.
    """
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    fig.savefig(abs_path, dpi=dpi, bbox_inches=bbox_inches)
    return abs_path
