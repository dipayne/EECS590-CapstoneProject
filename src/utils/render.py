"""
Human render utilities — raster_mode=human

Provides live, step-by-step visualization of an agent operating in either:
  - The custom LogisticsGridMDP (rendered with matplotlib + pygame backend)
  - Any Gymnasium environment (rendered with render_mode='human')

Usage
-----
# Watch trained DPAgent on the Logistics Grid
python -m src.utils.render --env logistics --load --episodes 3

# Watch a random policy on the Logistics Grid
python -m src.utils.render --env logistics --episodes 3

# Watch a random policy on the highway-v0 foundation environment
python -m src.utils.render --env highway-v0 --episodes 2
"""
from __future__ import annotations
import argparse
import random
import time
import json
import numpy as np


# ---------------------------------------------------------------------------
# Gymnasium foundation environment — render_mode='human'
# ---------------------------------------------------------------------------

def run_human_render(env_id: str, policy_fn=None,
                     episodes: int = 3, max_steps: int = 500,
                     fps: int = 10) -> None:
    """
    Run episodes on a Gymnasium environment with render_mode='human'.

    Parameters
    ----------
    env_id     : Gymnasium environment ID (e.g. 'highway-v0')
    policy_fn  : callable(obs) -> action, or None for a random policy
    episodes   : number of episodes to render
    max_steps  : max steps per episode
    fps        : target frames per second (controls sleep between steps)
    """
    from src.envs.make_env import make_env

    env = make_env(env_id, render_mode="human")

    for ep in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        for t in range(max_steps):
            a = policy_fn(obs) if policy_fn is not None else env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(a)
            total += float(r)
            time.sleep(1.0 / fps)
            if terminated or truncated:
                break
        print(f"Episode {ep + 1}/{episodes}: return={total:.3f}  steps={t + 1}")

    env.close()


# ---------------------------------------------------------------------------
# LogisticsGridMDP — step-by-step matplotlib renderer
# ---------------------------------------------------------------------------

_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←", 4: "•"}

_COLORS = {
    "blocked": "#2b2b2b",
    "goal":    "#a8d5a2",
    "depot":   "#fff7cc",
    "empty":   "#f5f5f5",
    "agent":   "#4472c4",
    "agent_done": "#ffd700",
}


def render_logistics_grid(mdp, policy: np.ndarray | None = None,
                          episodes: int = 3, max_steps: int = 500,
                          delay: float = 0.35) -> None:
    """
    Step-by-step matplotlib render of the LogisticsGridMDP (raster_mode=human).

    Parameters
    ----------
    mdp        : LogisticsGridMDP instance
    policy     : np.ndarray[nS] of action indices, or None for random policy
    episodes   : number of episodes to render
    max_steps  : max steps per episode before giving up
    delay      : seconds to pause between frames
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("Logistics Grid — raster_mode=human")

    for ep in range(episodes):
        s = mdp.start_state()
        total = 0.0
        done = False

        for t in range(max_steps):
            _draw_frame(ax, mdp, s, ep + 1, episodes, t, total, done=False)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(delay)

            # Choose and execute action
            if policy is not None:
                a = int(policy[s])
            else:
                acts = mdp.actions(s)
                a = random.choice(acts) if acts else 0

            trans = mdp.transitions(s, a)
            s_next = random.choices(list(trans.keys()), weights=list(trans.values()))[0]
            total += mdp.reward(s, a, s_next)
            s = s_next

            if mdp.is_terminal(s):
                done = True
                break

        # Final frame
        _draw_frame(ax, mdp, s, ep + 1, episodes, t + 1, total, done=done)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(delay * 4)

        status = "DELIVERED" if done else "TIMEOUT"
        print(f"Episode {ep + 1}/{episodes}: {status}  return={total:.3f}  steps={t + 1}")

    plt.ioff()
    plt.show()


def _draw_frame(ax, mdp, agent_state: int, ep: int, total_eps: int,
                step: int, total_return: float, done: bool) -> None:
    """Render one frame of the logistics grid."""
    import matplotlib.patches as mpatches

    ax.clear()
    ax.set_xlim(-0.5, mdp.cols - 0.5)
    ax.set_ylim(mdp.rows - 0.5, -0.5)
    ax.set_xticks(range(mdp.cols))
    ax.set_yticks(range(mdp.rows))
    ax.grid(True, color="#cccccc", linewidth=0.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    title = f"Episode {ep}/{total_eps}  |  Step {step}  |  Return {total_return:.2f}"
    if done:
        title += "  ✓ DELIVERED"
    ax.set_title(title, fontsize=11)

    for cell_s in range(mdp.nS):
        r, c = divmod(cell_s, mdp.cols)
        if cell_s in mdp._blocked_states:
            color = _COLORS["blocked"]
            label = ""
        elif cell_s == mdp.goal_state():
            color = _COLORS["goal"]
            label = "G"
        elif cell_s == mdp.start_state():
            color = _COLORS["depot"]
            label = "D"
        else:
            color = _COLORS["empty"]
            label = ""

        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.48, r - 0.48), 0.96, 0.96,
            boxstyle="round,pad=0.02", facecolor=color,
            edgecolor="none", zorder=1
        ))
        if label:
            ax.text(c, r, label, ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#444444", zorder=2)

    # Draw agent
    ar, ac = divmod(agent_state, mdp.cols)
    agent_color = _COLORS["agent_done"] if done else _COLORS["agent"]
    ax.add_patch(mpatches.Circle(
        (ac, ar), 0.28, color=agent_color, zorder=5
    ))
    ax.text(ac, ar, "A", ha="center", va="center",
            fontsize=9, fontweight="bold",
            color="black" if done else "white", zorder=6)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="raster_mode=human — watch an agent run step by step"
    )
    parser.add_argument("--env", type=str, default="logistics",
                        help="'logistics' for the custom MDP, or a Gymnasium env id")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--delay", type=float, default=0.35,
                        help="Seconds between frames (logistics only)")
    parser.add_argument("--fps", type=int, default=8,
                        help="Frames per second (Gymnasium envs only)")
    parser.add_argument("--load", action="store_true",
                        help="Load trained policy from outputs/ (logistics only)")
    parser.add_argument("--random", action="store_true",
                        help="Force random policy even if --load is set")
    args = parser.parse_args()

    if args.env == "logistics":
        from src.envs.logistics_grid_mdp import LogisticsGridMDP
        mdp = LogisticsGridMDP()

        policy = None
        if args.load and not args.random:
            with open("outputs/policies/best_policy.json", "r", encoding="utf-8") as f:
                policy = np.array(json.load(f), dtype=int)
            print("Loaded trained policy from outputs/policies/best_policy.json")
        else:
            print("Using random policy.")

        render_logistics_grid(mdp, policy=policy,
                              episodes=args.episodes,
                              max_steps=args.max_steps,
                              delay=args.delay)
    else:
        print(f"Launching '{args.env}' with render_mode='human'.")
        run_human_render(args.env, policy_fn=None,
                         episodes=args.episodes,
                         max_steps=args.max_steps,
                         fps=args.fps)


if __name__ == "__main__":
    main()
