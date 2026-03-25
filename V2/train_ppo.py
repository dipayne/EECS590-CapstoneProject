"""Train a PPO agent on highway-v0.

PPO is the *second* deep RL algorithm implemented in V2, chosen because:
  - It is on-policy (complementary to DQN's off-policy approach)
  - It is sample-efficient relative to vanilla policy gradient
  - Its clipped surrogate objective provides stable, monotonic improvement
  - It is state-of-the-art on many discrete-action tasks

Usage
-----
    python train_ppo.py --steps 500000
    python train_ppo.py --steps 500000 --n-steps 2048 --n-epochs 10

Outputs
-------
    outputs/ppo/<run_id>/
        training_log.json
        returns.png
        pg_loss.png   vf_loss.png

    checkpoints/ppo/highway-v0/<run_id>/
        actor_best.pt   critic_best.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.envs.highway_wrapper import make_highway_env
from src.agents.deep.ppo import PPOAgent
from src.checkpoints.manager import CheckpointManager
from src.utils.logger import ExperimentLogger


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on highway-v0")
    p.add_argument("--steps",       type=int,   default=500_000)
    p.add_argument("--n-steps",     type=int,   default=2048,
                   help="Rollout length before each update")
    p.add_argument("--n-epochs",    type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--lam",         type=float, default=0.95)
    p.add_argument("--clip-eps",    type=float, default=0.2)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--vf-coef",     type=float, default=0.5)
    p.add_argument("--ent-coef",    type=float, default=0.01)
    p.add_argument("--max-grad",    type=float, default=0.5)
    p.add_argument("--no-norm-obs", action="store_true")
    p.add_argument("--hidden",      type=int,   nargs="+", default=[256, 256])
    p.add_argument("--device",      default="cpu")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--verbose-every", type=int, default=10)
    p.add_argument("--max-ep-steps", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    env = make_highway_env(normalize_obs=not args.no_norm_obs)
    obs_dim   = env.obs_dim
    n_actions = env.n_actions

    hparams = {
        "algorithm":   "ppo",
        "steps":       args.steps,
        "n_steps":     args.n_steps,
        "n_epochs":    args.n_epochs,
        "batch_size":  args.batch_size,
        "gamma":       args.gamma,
        "lam":         args.lam,
        "clip_eps":    args.clip_eps,
        "lr":          args.lr,
        "vf_coef":     args.vf_coef,
        "ent_coef":    args.ent_coef,
        "max_grad":    args.max_grad,
        "normalize_obs": not args.no_norm_obs,
        "hidden":      args.hidden,
        "obs_dim":     obs_dim,
        "n_actions":   n_actions,
        "seed":        args.seed,
    }

    logger = ExperimentLogger("outputs/ppo", run_name="ppo")
    logger.log_hparams(hparams)

    ckpt_mgr = CheckpointManager(
        "checkpoints", algorithm="ppo", task="highway-v0",
        hparams=hparams, save_every_steps=0,
    )

    agent = PPOAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lam=args.lam,
        clip_eps=args.clip_eps,
        lr=args.lr,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad,
        normalize_obs=not args.no_norm_obs,
        hidden_sizes=tuple(args.hidden),
        device=args.device,
    )

    print(f"\n=== PPO  total_steps={args.steps}  n_steps={args.n_steps}  "
          f"n_epochs={args.n_epochs}  device={args.device} ===")

    # ---- manual training loop for fine-grained logging ----
    obs, _ = env.reset()
    ep_return = 0.0
    ep_steps  = 0
    ep_count  = 0
    best_return = -float("inf")
    update_count = 0

    step = 0
    while step < args.steps:
        for _ in range(args.n_steps):
            if step >= args.steps:
                break

            a, val, lp = agent._get_action_value_logprob(obs)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            obs_proc = agent._process_obs(obs)
            agent.rollout.add(obs_proc, a, r, val, lp, done)

            ep_return += r
            ep_steps  += 1
            step += 1
            obs = obs2

            if done or ep_steps >= args.max_ep_steps:
                ep_count += 1
                logger.log_episode(ep_count, {"return": ep_return, "length": ep_steps})
                if ep_return > best_return:
                    best_return = ep_return
                    ckpt_mgr.save_best(agent, ep_return)
                obs, _ = env.reset()
                ep_return = 0.0
                ep_steps  = 0

        if agent.rollout._full:
            metrics = agent.update(obs, ep_steps == 0)
            update_count += 1
            logger.log_step(update_count, metrics)

            if args.verbose_every and update_count % args.verbose_every == 0:
                recent = [e["return"] for e in logger.episode_log[-20:]]
                avg = np.mean(recent) if recent else 0.0
                print(
                    f"  PPO update {update_count:>4}  step={step:>7}/{args.steps}  "
                    f"ep={ep_count}  avg_ret={avg:.3f}  "
                    f"pg={metrics['pg_loss']:.4f}  "
                    f"vf={metrics['vf_loss']:.4f}  "
                    f"ent={metrics['ent_loss']:.4f}"
                )

    env.close()

    logger.save()
    logger.plot_returns(window=20)
    logger.plot_losses(metric="pg_loss", window=10, title="PPO Policy Loss")
    logger.plot_losses(metric="vf_loss", window=10, title="PPO Value Loss")

    print(f"\n=== PPO training complete.  Best return: {best_return:.3f} ===")
    print(f"    Logs: {logger.run_dir}")
    print(f"    Checkpoints: {ckpt_mgr.run_dir}")


if __name__ == "__main__":
    main()
