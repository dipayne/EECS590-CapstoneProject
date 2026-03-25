"""REINFORCE — Monte Carlo Policy Gradient (Williams, 1992).

REINFORCE is the foundational policy-gradient algorithm.  At the end of
each episode it computes the full Monte Carlo return G_t and updates the
policy parameters in the direction that increases the log-probability of
actions proportional to their returns:

    ∇J(theta) ≈ Σ_t G_t ∇log π_theta(a_t | s_t)

Two variants are implemented:
  - ``"vanilla"``   — plain REINFORCE (high variance)
  - ``"baseline"``  — REINFORCE with a learned value-function baseline
                       (reduces variance, equivalent to A≈G_t − V(s_t))

Reference: Williams, 1992. "Simple statistical gradient-following algorithms."
           Sutton & Barto, Chapter 13.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict

from src.agents.base import BaseAgent
from src.networks.mlp import ActorMLP, CriticMLP


class REINFORCEAgent(BaseAgent):
    """REINFORCE with optional learned value-function baseline.

    Parameters
    ----------
    obs_dim:
        Flattened observation size.
    n_actions:
        Number of discrete actions.
    gamma:
        Discount factor.
    lr_actor:
        Learning rate for the policy network.
    lr_critic:
        Learning rate for the value baseline (used only when
        ``use_baseline=True``).
    use_baseline:
        If True, subtract V(s) from the return to reduce variance.
    hidden_sizes:
        Hidden layer widths for actor (and baseline) networks.
    device:
        Torch device string.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        use_baseline: bool = True,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.device = torch.device(device)

        self.actor  = ActorMLP(obs_dim, n_actions, hidden_sizes,
                               activation="tanh").to(self.device)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic: CriticMLP | None = None
        if use_baseline:
            self.critic  = CriticMLP(obs_dim, hidden_sizes,
                                     activation="tanh").to(self.device)
            self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs, training: bool = True) -> int:
        obs_t = torch.tensor(obs.flatten(), dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor.get_dist(obs_t)
            if training:
                action = dist.sample()
            else:
                action = dist.logits.argmax(dim=1)
        return int(action.item())

    def update(self, episode: List[Tuple]) -> Dict:  # type: ignore[override]
        """Update policy from a completed episode.

        Parameters
        ----------
        episode:
            List of (obs, action, reward) tuples.

        Returns
        -------
        dict with ``"pg_loss"`` (and ``"vf_loss"`` if baseline is used).
        """
        T = len(episode)
        obs_list   = [torch.tensor(o.flatten(), dtype=torch.float32,
                                   device=self.device) for o, _, _ in episode]
        acts_list  = [a for _, a, _ in episode]
        rewards    = [r for _, _, r in episode]

        # Compute discounted returns G_t
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        obs_t  = torch.stack(obs_list)               # (T, D)
        acts_t = torch.tensor(acts_list, dtype=torch.long, device=self.device)

        # ---- baseline ----
        if self.use_baseline and self.critic is not None:
            values = self.critic(obs_t)               # (T,)
            advantages = (returns_t - values).detach()

            # Value function update (MSE)
            vf_loss = nn.functional.mse_loss(values, returns_t)
            self.opt_critic.zero_grad()
            vf_loss.backward()
            self.opt_critic.step()
            vf_loss_val = float(vf_loss.item())
        else:
            advantages = returns_t
            # Normalise returns to reduce variance (when no baseline)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            vf_loss_val = 0.0

        # ---- policy gradient update ----
        log_probs, entropy = self.actor.evaluate_actions(obs_t, acts_t)
        pg_loss = -(log_probs * advantages).mean() - 1e-3 * entropy.mean()

        self.opt_actor.zero_grad()
        pg_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.opt_actor.step()

        return {
            "pg_loss": float(pg_loss.item()),
            "vf_loss": vf_loss_val,
            "ep_return": float(returns[0]),
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env, n_episodes: int, max_steps: int = 500,
              verbose_every: int = 50) -> dict:
        """Run REINFORCE for ``n_episodes`` complete episodes.

        Returns dict with ``"returns"``, ``"pg_losses"``.
        """
        all_returns: List[float] = []
        pg_losses:   List[float] = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            episode = []
            for _ in range(max_steps):
                a = self.select_action(obs, training=True)
                obs2, r, terminated, truncated, _ = env.step(a)
                episode.append((obs.copy(), a, r))
                obs = obs2
                if terminated or truncated:
                    break

            metrics = self.update(episode)
            ep_return = sum(self.gamma ** t * episode[t][2]
                            for t in range(len(episode)))
            all_returns.append(ep_return)
            pg_losses.append(metrics["pg_loss"])

            if verbose_every and (ep + 1) % verbose_every == 0:
                avg = np.mean(all_returns[-verbose_every:])
                print(
                    f"  REINFORCE ep {ep+1:>5}/{n_episodes}  "
                    f"avg_return={avg:.3f}  "
                    f"pg_loss={metrics['pg_loss']:.4f}"
                )

        return {"returns": all_returns, "pg_losses": pg_losses}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        ckpt = {"actor": self.actor.state_dict()}
        if self.critic is not None:
            ckpt["critic"] = self.critic.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        if self.critic is not None and "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
