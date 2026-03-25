"""A2C — Synchronous Advantage Actor-Critic (Mnih et al., 2016).

A2C is the synchronous, deterministic version of A3C.  At each step it:
  1. Collects n_steps of experience with a single worker (synchronous).
  2. Computes n-step returns and advantages A(s,a) = R_t − V(s_t).
  3. Updates the actor with ∇log π(a|s) · A(s,a) and the critic with MSE.

Relationship to A3C:
  - A3C runs multiple asynchronous workers in parallel threads.
  - A2C replaces that parallelism with a single synchronous rollout,
    giving identical theoretical properties with simpler code.

Relationship to PPO:
  - A2C performs a single gradient step per rollout.
  - PPO performs K epochs with a clipped surrogate — more data-efficient.

This implementation supports both **shared** and **separate** networks
for actor and critic (controlled by ``shared_backbone=True/False``).

Reference: Mnih et al., 2016. "Asynchronous Methods for Deep RL."
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple

from src.agents.base import BaseAgent
from src.networks.mlp import ActorMLP, CriticMLP, MLP


# ---------------------------------------------------------------------------
# Optional shared-backbone A2C network
# ---------------------------------------------------------------------------

class SharedBackboneA2C(nn.Module):
    """Single shared feature extractor with separate actor/critic heads."""

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.shared = nn.Sequential(*layers)
        self.actor_head  = nn.Linear(in_dim, n_actions)
        self.critic_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor):
        feat = self.shared(x.float().flatten(start_dim=1) if x.dim() > 2 else x.float())
        logits = self.actor_head(feat)
        value  = self.critic_head(feat).squeeze(-1)
        return logits, value


# ---------------------------------------------------------------------------
# A2C Agent
# ---------------------------------------------------------------------------

class A2CAgent(BaseAgent):
    """Advantage Actor-Critic (synchronous A2C) for discrete actions.

    Parameters
    ----------
    obs_dim:
        Flattened observation size.
    n_actions:
        Number of discrete actions.
    n_steps:
        Number of environment steps per rollout before an update.
    gamma:
        Discount factor.
    lr:
        Learning rate (shared optimiser).
    vf_coef:
        Coefficient c₁ for the value-loss term.
    ent_coef:
        Coefficient c₂ for the entropy bonus.
    max_grad_norm:
        Global gradient-norm clipping.
    shared_backbone:
        If True, actor and critic share a feature extractor.
        If False, two independent networks are used.
    hidden_sizes:
        Hidden layer widths.
    device:
        Torch device string.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_steps: int = 5,
        gamma: float = 0.99,
        lr: float = 7e-4,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        shared_backbone: bool = True,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.shared_backbone = shared_backbone

        if shared_backbone:
            self.network = SharedBackboneA2C(obs_dim, n_actions,
                                             hidden_sizes).to(self.device)
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=lr, eps=1e-5)
        else:
            self.actor  = ActorMLP(obs_dim, n_actions, hidden_sizes,
                                   activation="tanh").to(self.device)
            self.critic = CriticMLP(obs_dim, hidden_sizes,
                                    activation="tanh").to(self.device)
            params = list(self.actor.parameters()) + list(self.critic.parameters())
            self.optimizer = optim.RMSprop(params, lr=lr, eps=1e-5)

        self._total_steps = 0
        self.episode_returns: List[float] = []

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward(self, obs_t: torch.Tensor):
        """Return (logits, value) regardless of backbone style."""
        if self.shared_backbone:
            return self.network(obs_t)
        else:
            logits = self.actor(obs_t)
            value  = self.critic(obs_t)
            return logits, value

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs, training: bool = True) -> int:
        obs_t = torch.tensor(obs.flatten(), dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample() if training else logits.argmax(dim=1)
        return int(action.item())

    def update(self, rollout: dict) -> Dict:  # type: ignore[override]
        """Compute n-step A2C loss and perform one gradient step.

        Parameters
        ----------
        rollout:
            dict with keys ``obs``, ``actions``, ``rewards``,
            ``dones``, ``last_obs``, ``last_done``.
        """
        obs_t     = torch.tensor(rollout["obs"],     dtype=torch.float32, device=self.device)
        acts_t    = torch.tensor(rollout["actions"], dtype=torch.long,    device=self.device)
        rewards   = rollout["rewards"]
        dones     = rollout["dones"]

        # ---- bootstrap value for last step ----
        with torch.no_grad():
            last_obs_t = torch.tensor(rollout["last_obs"], dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
            _, last_val = self._forward(last_obs_t)
            last_val = 0.0 if rollout["last_done"] else float(last_val.item())

        # ---- n-step returns ----
        returns = []
        R = last_val
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # ---- forward pass ----
        logits, values = self._forward(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(acts_t)
        entropy   = dist.entropy().mean()

        # ---- losses ----
        advantages = (returns_t - values).detach()
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss = -(log_probs * advantages).mean()
        vf_loss = nn.functional.mse_loss(values, returns_t)
        total_loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self.network.parameters() if self.shared_backbone
            else list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "pg_loss":  float(pg_loss.item()),
            "vf_loss":  float(vf_loss.item()),
            "entropy":  float(entropy.item()),
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env, total_steps: int, max_ep_steps: int = 500,
              verbose_every: int = 5000) -> dict:
        """Collect n_steps rollouts and update for ``total_steps`` steps.

        Returns dict with ``"returns"``.
        """
        all_returns: List[float] = []
        all_pg_losses: List[float] = []

        obs, _ = env.reset()
        ep_return = 0.0
        ep_steps  = 0
        step = 0

        while step < total_steps:
            # ---- collect rollout ----
            rollout: dict = {
                "obs":     [],
                "actions": [],
                "rewards": [],
                "dones":   [],
            }
            last_done = False

            for _ in range(self.n_steps):
                if step >= total_steps:
                    break

                a = self.select_action(obs, training=True)
                obs2, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated

                rollout["obs"].append(obs.flatten())
                rollout["actions"].append(a)
                rollout["rewards"].append(r)
                rollout["dones"].append(float(done))

                ep_return += r
                ep_steps  += 1
                self._total_steps += 1
                step += 1
                obs = obs2

                if done or ep_steps >= max_ep_steps:
                    all_returns.append(ep_return)
                    obs, _ = env.reset()
                    ep_return = 0.0
                    ep_steps  = 0
                    last_done = True
                else:
                    last_done = False

            rollout["obs"]     = np.array(rollout["obs"],     dtype=np.float32)
            rollout["actions"] = np.array(rollout["actions"], dtype=np.int64)
            rollout["rewards"] = rollout["rewards"]
            rollout["dones"]   = rollout["dones"]
            rollout["last_obs"]  = obs.flatten()
            rollout["last_done"] = last_done

            if rollout["obs"].shape[0] > 0:
                metrics = self.update(rollout)
                all_pg_losses.append(metrics["pg_loss"])

            if verbose_every and step % verbose_every == 0:
                avg = np.mean(all_returns[-20:]) if all_returns else 0.0
                print(
                    f"  A2C step {step:>7}/{total_steps}  "
                    f"ep={len(all_returns)}  avg_ret={avg:.3f}  "
                    f"pg={metrics.get('pg_loss', 0):.4f}"
                )

        return {"returns": all_returns, "pg_losses": all_pg_losses}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        state = {}
        if self.shared_backbone:
            state["network"] = self.network.state_dict()
        else:
            state["actor"]  = self.actor.state_dict()
            state["critic"] = self.critic.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if self.shared_backbone:
            self.network.load_state_dict(ckpt["network"])
        else:
            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
