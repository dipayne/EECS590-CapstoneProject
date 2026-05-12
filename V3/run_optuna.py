"""Bayesian hyperparameter tuning for DQN(Double + Dueling + PER) on highway-v0.

Uses Optuna's TPE sampler (Tree-structured Parzen Estimator) to search the
agent hyperparameter space. The best V2 config (full DQN + Double + Dueling +
PER) is the starting point; we tune around it to see how much further
returns can be pushed.

Search space (all sampled per trial):
    lr                   log-uniform [1e-5, 5e-3]
    gamma                uniform     [0.95, 0.999]
    batch_size           categorical {32, 64, 128, 256}
    target_update_freq   int         [100, 1000]
    eps_decay_pct        uniform     [0.4, 0.9]   (fraction of n_steps)
    train_start          int         [200, 2000]

Objective: maximize mean discounted return over the last 50 episodes of
training (matches the V2 ablation evaluation metric).

Outputs (under V3/outputs/optuna/):
    study.db                       SQLite store (resumable, queryable)
    optimization_history.png       trial value over time
    param_importances.png          fANOVA importance per hyperparameter
    parallel_coordinate.png        trial trajectories across the search space
    best_params.json               winning HPs and their final return
    trials.csv                     full per-trial table

Run:
    python V3/run_optuna.py --trials 30 --steps 5000
    python V3/run_optuna.py --trials 3  --steps 2000   # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler


THIS_DIR = Path(__file__).resolve().parent
V2_DIR = THIS_DIR.parent / "V2"
sys.path.insert(0, str(V2_DIR))

from src.envs.highway_wrapper import make_highway_env   # noqa: E402
from src.agents.deep.dqn import DQNAgent                # noqa: E402


OUT_DIR = THIS_DIR / "outputs" / "optuna"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EP_STEPS = 200
SEED = 42


def build_agent(env, trial: optuna.Trial, n_steps: int) -> DQNAgent:
    lr                 = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    gamma              = trial.suggest_float("gamma", 0.95, 0.999)
    batch_size         = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    target_update_freq = trial.suggest_int("target_update_freq", 100, 1000)
    eps_decay_pct      = trial.suggest_float("eps_decay_pct", 0.4, 0.9)
    train_start        = trial.suggest_int("train_start", 200, 2000)

    return DQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        variant="double_dueling",
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        buffer_capacity=min(20_000, n_steps),
        target_update_freq=target_update_freq,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=max(1, int(n_steps * eps_decay_pct)),
        use_per=True,
        train_start=train_start,
    )


def objective_factory(n_steps: int):
    def objective(trial: optuna.Trial) -> float:
        np.random.seed(SEED + trial.number)
        env = make_highway_env(normalize_obs=True)
        agent = build_agent(env, trial, n_steps)
        t0 = time.time()
        result = agent.train(
            env, n_steps=n_steps, max_ep_steps=MAX_EP_STEPS,
            verbose_every=0,
        )
        env.close()
        elapsed = time.time() - t0
        rets = result["returns"]
        if len(rets) == 0:
            return -1e6
        avg50 = float(np.mean(rets[-50:])) if len(rets) >= 50 else float(np.mean(rets))
        trial.set_user_attr("n_episodes", len(rets))
        trial.set_user_attr("elapsed_s", elapsed)
        print(
            f"  trial {trial.number:>3}  "
            f"avg50={avg50:+.3f}  eps={len(rets):>4}  ({elapsed:.0f}s)  "
            f"lr={trial.params['lr']:.2e}  bs={trial.params['batch_size']}  "
            f"gamma={trial.params['gamma']:.3f}"
        )
        return avg50
    return objective


def save_plots_and_report(study: optuna.Study) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
    )

    try:
        ax = plot_optimization_history(study)
        ax.figure.savefig(OUT_DIR / "optimization_history.png", dpi=130,
                          bbox_inches="tight")
        plt.close(ax.figure)
    except Exception as e:
        print(f"  [warn] optimization_history failed: {e}")

    if len(study.trials) >= 2:
        try:
            ax = plot_param_importances(study)
            ax.figure.savefig(OUT_DIR / "param_importances.png", dpi=130,
                              bbox_inches="tight")
            plt.close(ax.figure)
        except Exception as e:
            print(f"  [warn] param_importances failed: {e}")

        try:
            ax = plot_parallel_coordinate(study)
            ax.figure.savefig(OUT_DIR / "parallel_coordinate.png", dpi=130,
                              bbox_inches="tight")
            plt.close(ax.figure)
        except Exception as e:
            print(f"  [warn] parallel_coordinate failed: {e}")

    best_payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "n_trials": len(study.trials),
        "n_complete": sum(1 for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE),
    }
    (OUT_DIR / "best_params.json").write_text(json.dumps(best_payload, indent=2))

    df = study.trials_dataframe(attrs=("number", "value", "params",
                                       "state", "user_attrs"))
    df.to_csv(OUT_DIR / "trials.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Environment steps per trial (default: 5000). "
                             "V2 full training used 20000; reduce for speed.")
    parser.add_argument("--study-name", type=str, default="dqn_ddper_highway")
    args = parser.parse_args()

    storage = f"sqlite:///{OUT_DIR / 'study.db'}"
    print("=" * 64)
    print(f"  Optuna study: {args.study_name}")
    print(f"  Trials: {args.trials}   Steps/trial: {args.steps}")
    print(f"  Storage: {storage}")
    print("=" * 64)

    sampler = TPESampler(seed=SEED, n_startup_trials=max(5, args.trials // 6))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    t0 = time.time()
    study.optimize(objective_factory(args.steps), n_trials=args.trials,
                   show_progress_bar=False)
    total = time.time() - t0

    print("\n" + "=" * 64)
    print(f"  Done. {len(study.trials)} trials in {total/60:.1f} min")
    print(f"  Best value: {study.best_value:+.3f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k} = {v}")
    print("=" * 64)

    save_plots_and_report(study)
    print(f"  Artifacts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
