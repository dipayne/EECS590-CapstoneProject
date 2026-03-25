"""Hyperparameter sweep runner.

Runs a grid or random search over hyperparameters and collects results
into a summary JSON for comparison.

Usage
-----
    # Grid search over DQN learning rates and buffer sizes
    python scripts/run_experiment.py \
        --algo dqn \
        --param lr 1e-4 3e-4 1e-3 \
        --param buffer_cap 10000 50000 \
        --steps 100000 \
        --n-seeds 3

    # Random search (10 configs) for PPO
    python scripts/run_experiment.py \
        --algo ppo \
        --random-search 10 \
        --steps 300000

Results are written to outputs/sweeps/<algo>_<timestamp>/sweep_results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter sweep runner")
    p.add_argument("--algo", choices=["dqn", "ppo", "q_learning", "sarsa"],
                   default="dqn")
    p.add_argument("--param", nargs="+", action="append", default=[],
                   metavar=("NAME", "VALUE"),
                   help="Sweep parameter: --param lr 1e-4 3e-4  "
                        "(can be repeated for multiple params)")
    p.add_argument("--steps",         type=int, default=100_000)
    p.add_argument("--n-seeds",       type=int, default=1)
    p.add_argument("--random-search", type=int, default=0,
                   help="If > 0, sample this many random configs instead of grid")
    p.add_argument("--python",        default=sys.executable,
                   help="Python interpreter to use for subprocesses")
    p.add_argument("--dry-run",       action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


def parse_param_grid(raw_params):
    """Convert list-of-lists into {name: [values...]} dict."""
    grid = {}
    for entry in raw_params:
        name = entry[0]
        values = entry[1:]
        # Try to convert to float/int
        parsed = []
        for v in values:
            try:
                f = float(v)
                parsed.append(int(f) if f == int(f) else f)
            except ValueError:
                parsed.append(v)
        grid[name] = parsed
    return grid


def cli_flag(name: str, value) -> list:
    """Convert a hyperparameter name and value to a CLI flag list."""
    flag = "--" + name.replace("_", "-")
    if isinstance(value, bool):
        return [flag] if value else []
    return [flag, str(value)]


def build_command(python: str, algo: str, hparams: dict, steps: int, seed: int) -> list:
    """Build the subprocess command for a training run."""
    script = {
        "dqn":       "train_dqn.py",
        "ppo":       "train_ppo.py",
        "q_learning": "train_classical.py",
        "sarsa":     "train_classical.py",
    }[algo]

    # Run from V2/ directory (parent of scripts/)
    script_path = os.path.join(os.path.dirname(__file__), "..", script)

    cmd = [python, script_path, "--steps" if algo in {"dqn", "ppo"} else "--episodes",
           str(steps), "--seed", str(seed)]

    if algo in {"q_learning", "sarsa"}:
        cmd += ["--algo", algo]

    for name, value in hparams.items():
        cmd += cli_flag(name, value)

    return cmd


def main():
    args = parse_args()
    grid = parse_param_grid(args.param)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path("outputs") / "sweeps" / f"{args.algo}_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if args.random_search > 0:
        import random
        configs = []
        for _ in range(args.random_search):
            cfg = {k: random.choice(v) for k, v in grid.items()}
            configs.append(cfg)
    else:
        # Full grid search
        names = list(grid.keys())
        value_lists = [grid[n] for n in names]
        configs = [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]

    print(f"=== Hyperparameter sweep: {args.algo} ===")
    print(f"    {len(configs)} configs × {args.n_seeds} seeds "
          f"= {len(configs) * args.n_seeds} runs")
    print(f"    Results directory: {sweep_dir}\n")

    total = len(configs) * args.n_seeds
    run_idx = 0

    for cfg in configs:
        for seed in range(args.n_seeds):
            run_idx += 1
            cmd = build_command(args.python, args.algo, cfg, args.steps, seed)
            print(f"  [{run_idx}/{total}] {cfg}  seed={seed}")
            print(f"    CMD: {' '.join(cmd)}")

            if args.dry_run:
                results.append({"config": cfg, "seed": seed, "status": "dry_run"})
                continue

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - t0

            status = "ok" if proc.returncode == 0 else "error"
            result = {
                "config": cfg,
                "seed": seed,
                "status": status,
                "elapsed_s": round(elapsed, 1),
                "returncode": proc.returncode,
            }

            if proc.returncode != 0:
                print(f"    ERROR (rc={proc.returncode}):")
                print(proc.stderr[-500:])
            else:
                print(f"    OK  ({elapsed:.0f}s)")

            results.append(result)

    # Save sweep summary
    summary_path = sweep_dir / "sweep_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            "algo":     args.algo,
            "steps":    args.steps,
            "n_seeds":  args.n_seeds,
            "grid":     grid,
            "runs":     results,
        }, f, indent=2)

    print(f"\n=== Sweep complete.  Summary: {summary_path} ===")


if __name__ == "__main__":
    main()
