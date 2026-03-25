"""Evaluate a trained agent on highway-v0 and optionally generate saliency maps.

Usage
-----
    # Evaluate best DQN checkpoint
    python eval.py --algo dqn --variant double_dueling \
        --ckpt checkpoints/dqn/highway-v0/<run_id>/

    # Evaluate best PPO checkpoint
    python eval.py --algo ppo \
        --ckpt checkpoints/ppo/highway-v0/<run_id>/

    # Evaluate classical Q-Learning (Q-table)
    python eval.py --algo q_learning \
        --q-table outputs/classical/q_learning_q_learning/Q.npy

    # Generate saliency maps during evaluation
    python eval.py --algo dqn --variant double_dueling \
        --ckpt checkpoints/dqn/highway-v0/<run_id>/ \
        --saliency integrated_gradients --saliency-out outputs/saliency/dqn/

Outputs
-------
    eval_results.json   — per-episode returns / lengths / success rate
    saliency/           — PNG saliency frames (if --saliency is set)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
V1_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, V1_SRC)

from src.envs.highway_wrapper import make_highway_env


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained agents on highway-v0")
    p.add_argument("--algo",    choices=["dqn", "ppo", "q_learning", "sarsa",
                                         "monte_carlo", "td"],
                   required=True)
    p.add_argument("--variant", default="double_dueling",
                   help="DQN variant (dqn/double/dueling/double_dueling)")
    p.add_argument("--ckpt",    default=None,
                   help="Path to checkpoint directory (deep RL agents)")
    p.add_argument("--q-table", default=None,
                   help="Path to .npy Q-table file (classical agents)")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--render",   action="store_true")
    p.add_argument("--saliency", choices=["vanilla", "smoothgrad",
                                          "integrated_gradients", "grad_x_input"],
                   default=None)
    p.add_argument("--saliency-out", default="outputs/saliency")
    p.add_argument("--saliency-steps", type=int, default=1,
                   help="Save saliency every N steps")
    p.add_argument("--device",  default="cpu")
    p.add_argument("--seed",    type=int, default=0)
    return p.parse_args()


def load_deep_agent(args, obs_dim, n_actions):
    """Load a trained DQN or PPO agent from checkpoint directory."""
    import torch
    from pathlib import Path

    ckpt_dir = Path(args.ckpt)

    if args.algo == "dqn":
        from src.agents.deep.dqn import DQNAgent
        agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions,
                         variant=args.variant, device=args.device)
        online_path = ckpt_dir / "online_net_best.pt"
        target_path = ckpt_dir / "target_net_best.pt"
        if online_path.exists():
            agent.online_net.load_state_dict(
                torch.load(online_path, map_location=args.device))
            agent.target_net.load_state_dict(
                torch.load(target_path, map_location=args.device))
            print(f"  Loaded DQN checkpoint from {ckpt_dir}")
        else:
            print(f"  WARNING: checkpoint not found at {online_path}")
        return agent

    elif args.algo == "ppo":
        from src.agents.deep.ppo import PPOAgent
        agent = PPOAgent(obs_dim=obs_dim, n_actions=n_actions, device=args.device)
        actor_path = ckpt_dir / "actor_best.pt"
        critic_path = ckpt_dir / "critic_best.pt"
        if actor_path.exists():
            agent.actor.load_state_dict(
                torch.load(actor_path, map_location=args.device))
            agent.critic.load_state_dict(
                torch.load(critic_path, map_location=args.device))
            print(f"  Loaded PPO checkpoint from {ckpt_dir}")
        else:
            print(f"  WARNING: checkpoint not found at {actor_path}")
        return agent

    raise ValueError(f"Unknown deep algo: {args.algo}")


def load_classical_agent(args, env):
    """Load a classical agent from saved Q-table / V-table."""
    nS = env.n_tabular_states
    nA = env.n_actions

    if args.q_table and os.path.exists(args.q_table):
        Q = np.load(args.q_table)
        print(f"  Loaded Q-table from {args.q_table}")
    else:
        Q = np.zeros((nS, nA))
        print("  WARNING: Q-table not found; using zero table.")

    obs_to_state = env.tabular_state
    return Q, obs_to_state


def get_saliency_fn(name: str):
    from src.saliency.gradient_saliency import (
        vanilla_gradient, smoothgrad, integrated_gradients, gradient_x_input,
    )
    mapping = {
        "vanilla":            vanilla_gradient,
        "smoothgrad":         smoothgrad,
        "integrated_gradients": integrated_gradients,
        "grad_x_input":       gradient_x_input,
    }
    return mapping[name]


def eval_deep(agent, env, args):
    """Evaluate a deep RL agent for args.episodes episodes."""
    import torch
    device = torch.device(args.device)

    saliency_fn = get_saliency_fn(args.saliency) if args.saliency else None

    # Determine which network to use for saliency
    if hasattr(agent, "online_net"):
        sal_network = agent.online_net
    elif hasattr(agent, "actor"):
        sal_network = agent.actor
    else:
        sal_network = None

    results = []
    for ep in range(args.episodes):
        obs_raw, _ = env.reset()
        obs = obs_raw  # HighwayWrapper already flattens

        ep_return = 0.0
        ep_steps  = 0
        sal_frames = []

        for step in range(args.max_steps):
            a = agent.select_action(obs, training=False)

            # Saliency
            if saliency_fn and sal_network and (step % args.saliency_steps == 0):
                try:
                    attr = saliency_fn(sal_network, obs, a, device=device)
                    sal_frames.append((step, obs.copy(), attr, a))
                except Exception as e:
                    pass  # saliency is best-effort during eval

            obs2, r, terminated, truncated, _ = env.step(a)
            ep_return += r
            ep_steps  += 1
            obs = obs2
            if terminated or truncated:
                break

        results.append({"episode": ep, "return": ep_return, "length": ep_steps})

        # Save saliency frames
        if sal_frames and args.saliency_out:
            from src.saliency.visualization import plot_obs_heatmap, rasterize
            ep_dir = os.path.join(args.saliency_out, args.algo, f"ep{ep:04d}")
            os.makedirs(ep_dir, exist_ok=True)
            for step_i, obs_i, attr_i, a_i in sal_frames:
                fig = plot_obs_heatmap(
                    obs_i.reshape(5, 5), attr_i.reshape(5, 5),
                    title=f"ep={ep}  step={step_i}  action={a_i}  "
                          f"method={args.saliency}",
                )
                rasterize(fig, os.path.join(ep_dir, f"step_{step_i:04d}.png"))
                import matplotlib.pyplot as plt
                plt.close(fig)

        print(f"  ep {ep+1:>3}/{args.episodes}  return={ep_return:.3f}  "
              f"length={ep_steps}")

    return results


def eval_classical(Q, obs_to_state, env, args):
    """Evaluate a classical tabular agent."""
    results = []
    for ep in range(args.episodes):
        obs_raw, _ = env.reset()
        ep_return = 0.0
        ep_steps  = 0

        for _ in range(args.max_steps):
            s = obs_to_state(obs_raw)
            a = int(np.argmax(Q[s]))
            obs_raw, r, terminated, truncated, _ = env.step(a)
            ep_return += r
            ep_steps  += 1
            if terminated or truncated:
                break

        results.append({"episode": ep, "return": ep_return, "length": ep_steps})
        print(f"  ep {ep+1:>3}/{args.episodes}  return={ep_return:.3f}  "
              f"length={ep_steps}")

    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)

    render_mode = "human" if args.render else None
    env = make_highway_env(render_mode=render_mode, normalize_obs=True)

    print(f"\n=== Evaluating {args.algo} for {args.episodes} episodes ===")

    if args.algo in {"dqn", "ppo"}:
        agent = load_deep_agent(args, env.obs_dim, env.n_actions)
        results = eval_deep(agent, env, args)
    else:
        Q, obs_to_state = load_classical_agent(args, env)
        results = eval_classical(Q, obs_to_state, env, args)

    env.close()

    # ---- summary ----
    returns = [r["return"] for r in results]
    print(f"\n=== Eval summary ===")
    print(f"  Mean return: {np.mean(returns):.3f} ± {np.std(returns):.3f}")
    print(f"  Max return:  {np.max(returns):.3f}")
    print(f"  Min return:  {np.min(returns):.3f}")

    out_path = f"outputs/eval_{args.algo}_{args.variant}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "algo": args.algo,
            "variant": args.variant,
            "mean_return": float(np.mean(returns)),
            "std_return":  float(np.std(returns)),
            "episodes": results,
        }, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
