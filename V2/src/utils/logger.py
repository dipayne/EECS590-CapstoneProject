"""Experiment logging and training-curve utilities.

ExperimentLogger
----------------
Tracks per-step and per-episode metrics, saves them to JSON, and
produces training-curve plots with optional smoothing.

Usage
-----
    logger = ExperimentLogger("outputs/logs", run_name="dqn_double_dueling")
    logger.log_hparams({"lr": 1e-4, "gamma": 0.99, "variant": "double_dueling"})

    for step, metrics in training_loop():
        logger.log_step(step, metrics)   # e.g. {"loss": 0.04, "td_error": 1.2}

    logger.log_episode(ep, {"return": 12.5, "length": 200})
    logger.plot_returns(save_path="outputs/plots/returns.png")
    logger.save()
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ExperimentLogger:
    """Structured experiment logger with JSON serialisation and plot export.

    Parameters
    ----------
    log_dir:
        Directory where logs and plots are written.
    run_name:
        Human-readable identifier for this run.  A timestamp is appended
        to make each run directory unique.
    """

    def __init__(self, log_dir: str, run_name: str = "experiment"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(log_dir) / f"{run_name}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_name  = run_name
        self.hparams: Dict[str, Any] = {}
        self.step_log:    List[dict] = []
        self.episode_log: List[dict] = []

        self._step_count    = 0
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log_hparams(self, hparams: Dict[str, Any]) -> None:
        """Persist hyperparameters to ``hparams.json``."""
        self.hparams = hparams
        with open(self.run_dir / "hparams.json", "w") as f:
            json.dump({"run_name": self.run_name, **hparams}, f, indent=2)

    def log_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Record metrics for a single gradient / environment step."""
        record = {"step": step, **metrics}
        self.step_log.append(record)
        self._step_count += 1

    def log_episode(self, episode: int, metrics: Dict[str, float]) -> None:
        """Record metrics for a completed episode (return, length, etc.)."""
        record = {"episode": episode, **metrics}
        self.episode_log.append(record)
        self._episode_count += 1

    def save(self) -> None:
        """Write all logs to JSON files."""
        with open(self.run_dir / "step_log.json", "w") as f:
            json.dump(self.step_log, f, indent=2)
        with open(self.run_dir / "episode_log.json", "w") as f:
            json.dump(self.episode_log, f, indent=2)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_returns(
        self,
        save_path: Optional[str] = None,
        window: int = 20,
        title: str = "Episode Returns",
    ) -> plt.Figure:
        """Plot raw and smoothed episode returns.

        Parameters
        ----------
        save_path:
            If provided, save the figure here.  Defaults to
            ``run_dir/returns.png``.
        window:
            Smoothing window size (moving average).
        """
        if not self.episode_log:
            print("[Logger] No episode data to plot.")
            return None

        returns = [r.get("return", r.get("episode_return", 0.0))
                   for r in self.episode_log]
        episodes = [r["episode"] for r in self.episode_log]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(episodes, returns, alpha=0.3, color="steelblue", label="raw")

        if len(returns) >= window:
            smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
            ax.plot(episodes[window - 1:], smooth,
                    color="steelblue", linewidth=2, label=f"MA-{window}")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Discounted Return")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

        path = save_path or str(self.run_dir / "returns.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    def plot_losses(
        self,
        metric: str = "loss",
        save_path: Optional[str] = None,
        window: int = 200,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a step-level metric (e.g. loss or td_error) with smoothing."""
        values = [s[metric] for s in self.step_log if metric in s]
        if not values:
            print(f"[Logger] No step data for metric '{metric}'.")
            return None

        fig, ax = plt.subplots(figsize=(10, 4))
        steps = list(range(len(values)))
        ax.plot(steps, values, alpha=0.2, color="coral", label=metric)

        if len(values) >= window:
            smooth = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1:], smooth,
                    color="coral", linewidth=2, label=f"MA-{window}")

        ax.set_xlabel("Gradient step")
        ax.set_ylabel(metric)
        ax.set_title(title or f"{metric} over training")
        ax.legend()
        plt.tight_layout()

        path = save_path or str(self.run_dir / f"{metric}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Comparison across multiple runs
    # ------------------------------------------------------------------

    @staticmethod
    def compare_returns(
        run_dirs: List[str],
        labels: List[str],
        save_path: str,
        window: int = 20,
        title: str = "Algorithm Comparison — Episode Returns",
    ) -> plt.Figure:
        """Overlay return curves from multiple experiment run directories.

        Parameters
        ----------
        run_dirs:
            List of paths to run directories (each must have
            ``episode_log.json``).
        labels:
            Display name for each run.
        save_path:
            Output PNG path.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = plt.cm.tab10.colors

        for i, (run_dir, label) in enumerate(zip(run_dirs, labels)):
            ep_log_path = Path(run_dir) / "episode_log.json"
            if not ep_log_path.exists():
                print(f"[Logger] Warning: {ep_log_path} not found, skipping.")
                continue
            with open(ep_log_path) as f:
                ep_log = json.load(f)

            returns = [r.get("return", r.get("episode_return", 0.0)) for r in ep_log]
            episodes = [r["episode"] for r in ep_log]
            color = colors[i % len(colors)]

            ax.plot(episodes, returns, alpha=0.15, color=color)
            if len(returns) >= window:
                smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
                ax.plot(episodes[window - 1:], smooth, color=color,
                        linewidth=2, label=label)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Discounted Return")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
