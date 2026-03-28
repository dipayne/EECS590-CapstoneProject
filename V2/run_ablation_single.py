"""Train one DQN ablation config and save results to a JSON file.

Usage: python run_ablation_single.py <config_index>

Config indices:
  0  DQN baseline
  1  DQN + Double
  2  DQN + Dueling
  3  DQN + Double + Dueling
  4  DQN + D+D+PER (full)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.envs.highway_wrapper import make_highway_env
from src.agents.deep.dqn import DQNAgent
from pathlib import Path

CONFIGS = [
    {"label": "DQN (baseline)",        "variant": "dqn",           "use_per": False},
    {"label": "DQN + Double",           "variant": "double",         "use_per": False},
    {"label": "DQN + Dueling",          "variant": "dueling",        "use_per": False},
    {"label": "DQN + Double + Dueling", "variant": "double_dueling", "use_per": False},
    {"label": "DQN + D+D+PER (full)",   "variant": "double_dueling", "use_per": True},
]

N_STEPS   = 20_000
MAX_STEPS = 200
SEED      = 42
OUT_DIR   = Path("outputs/comparison")


def main(idx: int):
    cfg = CONFIGS[idx]
    np.random.seed(SEED + idx)
    print(f"[Config {idx}] Starting: {cfg['label']}", flush=True)

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

    out = {
        "label":           cfg["label"],
        "returns":         rets,
        "final_avg50":     avg50,
        "mean_return":     float(np.mean(rets)),
        "std_return":      float(np.std(rets)),
        "n_episodes":      len(rets),
        "training_time_s": elapsed,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"ablation_config_{idx}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[Config {idx}] Done: {cfg['label']}  "
          f"final-avg50={avg50:+.3f}  episodes={len(rets)}  ({elapsed:.0f}s)", flush=True)
    print(f"[Config {idx}] Saved: {out_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ablation_single.py <config_index 0-4>")
        sys.exit(1)
    main(int(sys.argv[1]))
