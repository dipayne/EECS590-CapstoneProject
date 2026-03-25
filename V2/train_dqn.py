"""Train a DQN agent (standard / Double / Dueling / Double-Dueling) on highway-v0.

Usage
-----
    python train_dqn.py --variant double_dueling --steps 200000
    python train_dqn.py --variant dqn --steps 200000 --no-per
    python train_dqn.py --variant double --steps 200000 --lr 3e-4

Outputs
-------
    outputs/dqn/<variant>_<run_id>/
        training_log.json   — per-episode returns + losses
        returns.png         — training curve
        hparams.json        — hyperparameters

    checkpoints/dqn/highway-v0/<run_id>/
        online_net_best.pt
        target_net_best.pt
        hparams.json

    data/replay_buffers/dqn/highway-v0/<run_id>/
        buffer.npz
        metadata.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.envs.highway_wrapper import make_highway_env
from src.agents.deep.dqn import DQNAgent
from src.checkpoints.manager import CheckpointManager
from src.replay_buffer.manager import BufferManager
from src.utils.logger import ExperimentLogger


def parse_args():
    p = argparse.ArgumentParser(description="Train DQN on highway-v0")
    p.add_argument("--variant",
                   choices=["dqn", "double", "dueling", "double_dueling"],
                   default="double_dueling")
    p.add_argument("--steps",      type=int,   default=200_000)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--buffer-cap", type=int,   default=50_000)
    p.add_argument("--target-freq",type=int,   default=500)
    p.add_argument("--eps-start",  type=float, default=1.0)
    p.add_argument("--eps-end",    type=float, default=0.05)
    p.add_argument("--eps-decay",  type=int,   default=50_000)
    p.add_argument("--train-start",type=int,   default=1_000)
    p.add_argument("--no-per",     action="store_true",
                   help="Disable Prioritized Experience Replay")
    p.add_argument("--hidden",     type=int,   nargs="+", default=[256, 256])
    p.add_argument("--device",     default="cpu")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--verbose-every", type=int, default=5000)
    p.add_argument("--max-ep-steps", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # ---- environment ----
    env = make_highway_env(normalize_obs=True)
    obs_dim  = env.obs_dim    # 25
    n_actions = env.n_actions  # 5

    hparams = {
        "variant":    args.variant,
        "lr":         args.lr,
        "gamma":      args.gamma,
        "batch_size": args.batch_size,
        "buffer_cap": args.buffer_cap,
        "target_freq": args.target_freq,
        "eps_start":  args.eps_start,
        "eps_end":    args.eps_end,
        "eps_decay":  args.eps_decay,
        "train_start": args.train_start,
        "use_per":    not args.no_per,
        "hidden":     args.hidden,
        "steps":      args.steps,
        "seed":       args.seed,
        "obs_dim":    obs_dim,
        "n_actions":  n_actions,
    }

    # ---- infrastructure ----
    logger = ExperimentLogger("outputs/dqn", run_name=args.variant)
    logger.log_hparams(hparams)

    ckpt_mgr = CheckpointManager(
        "checkpoints", algorithm="dqn", task="highway-v0",
        hparams=hparams, save_every_steps=50_000,
    )
    buf_mgr = BufferManager(
        "data/replay_buffers", algorithm="dqn", task="highway-v0",
        max_run_mb=200.0, max_total_mb=800.0,
    )
    run_dir = buf_mgr.new_run(hparams=hparams)

    # ---- agent ----
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        variant=args.variant,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_cap,
        target_update_freq=args.target_freq,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay_steps=args.eps_decay,
        hidden_sizes=tuple(args.hidden),
        use_per=not args.no_per,
        train_start=args.train_start,
        device=args.device,
    )

    print(f"\n=== DQN({args.variant})  steps={args.steps}  "
          f"PER={'yes' if not args.no_per else 'no'}  device={args.device} ===")

    # ---- training loop (manual for logging) ----
    obs, _ = env.reset()
    ep_return = 0.0
    ep_steps  = 0
    ep_count  = 0
    best_return = -float("inf")

    for step in range(args.steps):
        a = agent.select_action(obs, training=True)
        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        agent.buffer.add(obs.flatten(), a, r, obs2.flatten(), terminated)

        metrics = agent.update()
        if metrics:
            logger.log_step(step, metrics)
            ckpt_mgr.save_periodic(agent, agent._grad_step)

        # eps linear decay
        agent.epsilon = max(agent.epsilon_end, agent.epsilon - agent.epsilon_decay)

        ep_return += r
        ep_steps  += 1
        obs = obs2

        if done or ep_steps >= args.max_ep_steps:
            ep_count += 1
            logger.log_episode(ep_count, {
                "return": ep_return,
                "length": ep_steps,
                "epsilon": agent.epsilon,
            })

            # Save best checkpoint
            if ep_return > best_return:
                best_return = ep_return
                ckpt_mgr.save_best(agent, ep_return)

            obs, _ = env.reset()
            ep_return = 0.0
            ep_steps  = 0

        if args.verbose_every and (step + 1) % args.verbose_every == 0:
            recent = [e["return"] for e in logger.episode_log[-20:]]
            avg = np.mean(recent) if recent else 0.0
            print(
                f"  step {step+1:>7}/{args.steps}  ep={ep_count}  "
                f"avg_ret={avg:.3f}  eps={agent.epsilon:.4f}  "
                f"loss={np.mean(agent.losses[-100:]):.4f}"
                if agent.losses else
                f"  step {step+1:>7}/{args.steps}  ep={ep_count}  "
                f"avg_ret={avg:.3f}  eps={agent.epsilon:.4f}"
            )

    env.close()

    # ---- save artifacts ----
    logger.save()
    logger.plot_returns(window=20)
    if agent.losses:
        logger.plot_losses(metric="loss", window=200)

    # Save replay buffer
    buf_mgr.save_buffer(agent.buffer, run_dir)
    buf_mgr.enforce_size_limit()

    print(f"\n=== Training complete.  Best return: {best_return:.3f} ===")
    print(f"    Logs: {logger.run_dir}")
    print(f"    Checkpoints: {ckpt_mgr.run_dir}")


if __name__ == "__main__":
    main()
