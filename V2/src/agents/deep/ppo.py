"""Proximal Policy Optimisation (PPO) agent for discrete action spaces.

Implementation follows the original Schulman et al. (2017) paper with
Generalized Advantage Estimation (GAE, Schulman et al. 2016) and the
practical tricks from the "Implementation Matters" paper (Engstrom et al.):

  - Orthogonal weight initialisation
  - Observation normalisation  (running mean/std)
  - Value-function clipping
  - Entropy bonus for exploration
  - Global gradient-norm clipping
  - Separate actor / critic networks (or optionally shared)

This implementation uses *on-policy rollouts*: it collects a fixed number
of steps with the current policy, computes advantages, then performs
multiple epochs of mini-batch gradient updates before discarding the data.

References
----------
Schulman et al., 2017. "Proximal Policy Optimization Algorithms."
Schulman et al., 2016. "High-Dimensional Continuous Control Using GAE."
Engstrom et al., 2020. "Implementation Matters in Deep RL."
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List

from src.agents.base import BaseAgent
from src.networks.mlp import ActorMLP, CriticMLP


# ---------------------------------------------------------------------------
# Running observation normaliser
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Incremental online mean/variance estimator (Welford's algorithm)."""

    def __init__(self, shape: tuple):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # avoid div-by-zero

    def update(self, batch: np.ndarray) -> None:
        batch = batch.reshape(-1, *self.mean.shape)
        m_b = batch.mean(0)
        v_b = batch.var(0)
        n_b = batch.shape[0]
        n   = self.count + n_b
        delta = m_b - self.mean
        self.mean  = self.mean + delta * n_b / n
        self.var   = (self.var * self.count + v_b * n_b + delta ** 2 * self.count * n_b / n) / n
        self.count = n

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -clip, clip)


# ---------------------------------------------------------------------------
# Rollout buffer — stores one full batch of on-policy experience
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-size buffer for on-policy PPO trajectories."""

    def __init__(self, n_steps: int, obs_dim: int, gamma: float, lam: float,
                 device: torch.device):
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.device = device

        self.obs     = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.values  = np.zeros(n_steps + 1, dtype=np.float32)  # +1 for bootstrap
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.dones   = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns    = np.zeros(n_steps, dtype=np.float32)

        self._ptr = 0
        self._full = False

    def add(self, obs, action, reward, value, log_prob, done) -> None:
        t = self._ptr
        self.obs[t]      = obs
        self.actions[t]  = action
        self.rewards[t]  = reward
        self.values[t]   = value
        self.log_probs[t] = log_prob
        self.dones[t]    = float(done)
        self._ptr += 1
        if self._ptr == self.n_steps:
            self._full = True

    def compute_gae(self, last_value: float) -> None:
        """Compute GAE advantages and discounted returns in-place."""
        self.values[self.n_steps] = last_value
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            delta = (self.rewards[t]
                     + self.gamma * self.values[t + 1] * (1 - self.dones[t])
                     - self.values[t])
            gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values[:self.n_steps]

    def get_batches(self, batch_size: int):
        """Yield mini-batches of tensors for PPO epochs."""
        assert self._full, "Buffer not full yet."
        idxs = np.random.permutation(self.n_steps)
        # Normalise advantages
        adv = self.advantages[idxs]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, self.n_steps, batch_size):
            b = idxs[start: start + batch_size]
            yield (
                torch.tensor(self.obs[b],       device=self.device),
                torch.tensor(self.actions[b],   device=self.device),
                torch.tensor(self.log_probs[b], device=self.device),
                torch.tensor(self.returns[b],   device=self.device),
                torch.tensor(adv[start: start + batch_size], device=self.device),
            )

    def reset(self) -> None:
        self._ptr = 0
        self._full = False


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent(BaseAgent):
    """On-policy Proximal Policy Optimisation for discrete actions.

    Parameters
    ----------
    obs_dim:
        Flattened observation size.
    n_actions:
        Number of discrete actions.
    n_steps:
        Number of environment steps per rollout (before each update).
    n_epochs:
        Number of gradient epochs over each rollout batch.
    batch_size:
        Mini-batch size within each epoch.
    gamma:
        Discount factor.
    lam:
        GAE lambda.
    clip_eps:
        PPO clipping parameter eps.
    lr:
        Adam learning rate (shared for actor and critic).
    vf_coef:
        Value-loss coefficient c₁.
    ent_coef:
        Entropy bonus coefficient c₂.
    max_grad_norm:
        Global gradient-norm clip threshold.
    clip_vf:
        Whether to clip the value-function loss (PPO implementation detail).
    normalize_obs:
        If True, maintain running obs normalisation.
    hidden_sizes:
        Hidden layer sizes for both actor and critic networks.
    device:
        Torch device string.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        clip_vf: bool = True,
        normalize_obs: bool = True,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.clip_vf = clip_vf
        self.device = torch.device(device)

        # ---- networks ----
        self.actor  = ActorMLP(obs_dim, n_actions, hidden_sizes, activation="tanh").to(self.device)
        self.critic = CriticMLP(obs_dim, hidden_sizes, activation="tanh").to(self.device)

        # Shared optimiser (common in PPO implementations)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(params, lr=lr, eps=1e-5)

        # ---- rollout buffer ----
        self.rollout = RolloutBuffer(n_steps, obs_dim, gamma, lam, self.device)

        # ---- obs normalisation ----
        self.normalize_obs = normalize_obs
        self.obs_rms = RunningMeanStd((obs_dim,)) if normalize_obs else None

        self._total_steps = 0
        self._update_count = 0
        self.episode_returns: List[float] = []

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs, training: bool = True) -> int:
        obs_proc = self._process_obs(obs)
        obs_t = torch.tensor(obs_proc, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor.get_dist(obs_t)
            if training:
                action = dist.sample()
            else:
                action = dist.logits.argmax(dim=1)
        return int(action.item())

    def _get_action_value_logprob(self, obs: np.ndarray):
        """Returns (action, value, log_prob) for environment stepping."""
        obs_proc = self._process_obs(obs)
        obs_t = torch.tensor(obs_proc, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor.get_dist(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(obs_t)
        return int(action.item()), float(value.item()), float(log_prob.item())

    def update(self, last_obs: np.ndarray, last_done: bool) -> dict:  # type: ignore[override]
        """Compute GAE and perform PPO update epochs over current rollout.

        Call this after the rollout buffer is full.
        """
        # Bootstrap value for GAE
        obs_proc = self._process_obs(last_obs)
        obs_t = torch.tensor(obs_proc, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_val = 0.0 if last_done else float(self.critic(obs_t).item())

        self.rollout.compute_gae(last_val)

        pg_losses, vf_losses, ent_losses = [], [], []

        # Old value predictions for clipped VF loss
        old_values = torch.tensor(self.rollout.values[:self.n_steps],
                                  device=self.device)

        for _ in range(self.n_epochs):
            for obs_b, acts_b, old_lp_b, returns_b, adv_b in \
                    self.rollout.get_batches(self.batch_size):

                # ---- actor loss (clipped surrogate) ----
                log_probs, entropy = self.actor.evaluate_actions(obs_b, acts_b)
                ratio = torch.exp(log_probs - old_lp_b)
                clip_adv = torch.clamp(ratio, 1 - self.clip_eps,
                                              1 + self.clip_eps) * adv_b
                pg_loss = -torch.min(ratio * adv_b, clip_adv).mean()

                # ---- critic loss ----
                values_pred = self.critic(obs_b)
                if self.clip_vf:
                    # Clip value update around old value to prevent large jumps
                    start_idx = 0  # mini-batch slice (approx)
                    values_clipped = old_values[:len(obs_b)] + torch.clamp(
                        values_pred - old_values[:len(obs_b)],
                        -self.clip_eps, self.clip_eps
                    )
                    vf_loss = torch.max(
                        nn.functional.mse_loss(values_pred, returns_b),
                        nn.functional.mse_loss(values_clipped, returns_b),
                    )
                else:
                    vf_loss = nn.functional.mse_loss(values_pred, returns_b)

                # ---- entropy bonus ----
                ent_loss = -entropy.mean()

                # ---- combined loss ----
                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                pg_losses.append(float(pg_loss.item()))
                vf_losses.append(float(vf_loss.item()))
                ent_losses.append(float(ent_loss.item()))

        self.rollout.reset()
        self._update_count += 1

        return {
            "pg_loss":  float(np.mean(pg_losses)),
            "vf_loss":  float(np.mean(vf_losses)),
            "ent_loss": float(np.mean(ent_losses)),
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env, total_steps: int, max_ep_steps: int = 500,
              verbose_every: int = 10) -> dict:
        """Collect rollouts and update policy for ``total_steps`` env steps.

        Returns dict with ``"returns"`` and training-loss lists.
        """
        all_returns: List[float] = []
        pg_losses, vf_losses = [], []

        obs, _ = env.reset()
        ep_return = 0.0
        ep_steps = 0
        last_done = False

        step = 0
        while step < total_steps:
            # ---- collect rollout ----
            for _ in range(self.n_steps):
                if step >= total_steps:
                    break
                a, val, lp = self._get_action_value_logprob(obs)
                obs2, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated

                obs_proc = self._process_obs(obs)
                self.rollout.add(obs_proc, a, r, val, lp, done)

                ep_return += r
                ep_steps += 1
                self._total_steps += 1
                step += 1

                obs = obs2
                if done or ep_steps >= max_ep_steps:
                    all_returns.append(ep_return)
                    obs, _ = env.reset()
                    ep_return = 0.0
                    ep_steps = 0
                    last_done = True
                else:
                    last_done = False

            if self.rollout._full:
                metrics = self.update(obs, last_done)
                pg_losses.append(metrics["pg_loss"])
                vf_losses.append(metrics["vf_loss"])

                update_n = self._update_count
                if verbose_every and update_n % verbose_every == 0:
                    avg_ret = np.mean(all_returns[-20:]) if all_returns else 0.0
                    print(
                        f"  PPO update {update_n:>4}  step={step:>7}/{total_steps}  "
                        f"avg_ret={avg_ret:.3f}  pg={metrics['pg_loss']:.4f}  "
                        f"vf={metrics['vf_loss']:.4f}  ent={metrics['ent_loss']:.4f}"
                    )

        return {"returns": all_returns, "pg_losses": pg_losses, "vf_losses": vf_losses}

    # ------------------------------------------------------------------
    # Obs processing
    # ------------------------------------------------------------------

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        flat = obs.flatten().astype(np.float32)
        if self.normalize_obs and self.obs_rms is not None:
            self.obs_rms.update(flat[np.newaxis])
            flat = self.obs_rms.normalize(flat).astype(np.float32)
        return flat

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor":     self.actor.state_dict(),
                "critic":    self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps":     self._total_steps,
                "updates":   self._update_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._total_steps  = ckpt.get("steps", 0)
        self._update_count = ckpt.get("updates", 0)
