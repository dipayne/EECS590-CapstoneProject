"""Deep Q-Network agent with optional Double-DQN and Dueling-DQN.

Three DQN variants are selectable via the ``variant`` parameter:
  - ``"dqn"``         — original DQN (Mnih et al., 2015)
  - ``"double"``      — Double DQN   (van Hasselt et al., 2016)
  - ``"dueling"``     — Dueling DQN  (Wang et al., 2016)
  - ``"double_dueling"`` — Double + Dueling combined

All variants use:
  - Replay buffer (standard or Prioritized PER)
  - Fixed target network with periodic hard copy
  - eps-greedy exploration with exponential decay
  - Huber loss (smooth L1) for stable gradient magnitudes

References
----------
Mnih et al., 2015. "Human-level control through deep reinforcement learning."
van Hasselt et al., 2016. "Deep Reinforcement Learning with Double Q-learning."
Wang et al., 2016. "Dueling Network Architectures for Deep RL."
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple

from src.agents.base import BaseAgent
from src.networks.mlp import MLP
from src.networks.dueling import DuelingQNetwork
from src.replay_buffer.buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent(BaseAgent):
    """DQN family agent for discrete-action environments.

    Parameters
    ----------
    obs_dim:
        Flattened observation size (e.g. 25 for highway-env).
    n_actions:
        Number of discrete actions.
    variant:
        One of ``"dqn"``, ``"double"``, ``"dueling"``, ``"double_dueling"``.
    gamma:
        Discount factor.
    lr:
        Adam learning rate.
    batch_size:
        Mini-batch size for gradient updates.
    buffer_capacity:
        Maximum replay buffer size.
    target_update_freq:
        Number of gradient steps between target-network copies.
    epsilon_start, epsilon_end, epsilon_decay_steps:
        eps schedule: eps decays linearly from start to end over decay_steps.
    hidden_sizes:
        Hidden layer widths for the MLP backbone.
    use_per:
        If True, use Prioritized Experience Replay.
    train_start:
        Number of transitions to collect before starting gradient updates.
    device:
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        variant: str = "double_dueling",
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update_freq: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        use_per: bool = False,
        train_start: int = 1_000,
        device: str = "cpu",
    ):
        valid = {"dqn", "double", "dueling", "double_dueling"}
        assert variant in valid, f"variant must be one of {valid}"

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.variant = variant
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.train_start = train_start
        self.device = torch.device(device)

        # ---- networks ----
        use_dueling = "dueling" in variant
        if use_dueling:
            self.online_net = DuelingQNetwork(obs_dim, n_actions,
                                              shared_hidden=(256,),
                                              stream_hidden=128).to(self.device)
        else:
            self.online_net = MLP(obs_dim, n_actions,
                                  hidden_sizes=hidden_sizes).to(self.device)

        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # ---- replay buffer ----
        obs_shape = (obs_dim,)
        if use_per:
            self.buffer = PrioritizedReplayBuffer(buffer_capacity, obs_shape)
        else:
            self.buffer = ReplayBuffer(buffer_capacity, obs_shape)
        self.use_per = use_per

        self._step = 0          # total environment steps
        self._grad_step = 0     # total gradient updates

        # For external tracking
        self.losses: list = []

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        obs_t = torch.tensor(obs.flatten(), dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online_net(obs_t)
        return int(q.argmax(dim=1).item())

    def update(self) -> dict:  # type: ignore[override]
        """Sample from replay buffer and perform one gradient step.

        Returns dict with ``"loss"`` and ``"td_error"`` for logging.
        """
        if len(self.buffer) < self.train_start:
            return {}

        if self.use_per:
            obs, acts, rews, next_obs, dones, weights, tree_idxs = \
                self.buffer.sample(self.batch_size)
            weights_t = torch.tensor(weights, device=self.device)
        else:
            obs, acts, rews, next_obs, dones = self.buffer.sample(self.batch_size)
            weights_t = None
            tree_idxs = None

        obs_t      = torch.tensor(obs,      device=self.device)
        acts_t     = torch.tensor(acts,     device=self.device)
        rews_t     = torch.tensor(rews,     device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        dones_t    = torch.tensor(dones,    device=self.device)

        # ---- compute targets ----
        with torch.no_grad():
            if "double" in self.variant:
                # Double DQN: select action with online net, evaluate with target
                next_q_online = self.online_net(next_obs_t)
                next_acts = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.target_net(next_obs_t).gather(1, next_acts).squeeze(1)
            else:
                # Standard DQN: greedy with target net
                next_q_target = self.target_net(next_obs_t).max(dim=1).values

            targets = rews_t + self.gamma * next_q_target * (1.0 - dones_t)

        # ---- compute current Q values ----
        q_vals = self.online_net(obs_t).gather(1, acts_t.unsqueeze(1)).squeeze(1)

        # ---- Huber loss ----
        td_errors = (targets - q_vals).detach()
        if weights_t is not None:
            element_loss = nn.functional.huber_loss(q_vals, targets, reduction="none")
            loss = (weights_t * element_loss).mean()
        else:
            loss = nn.functional.huber_loss(q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ---- update PER priorities ----
        if self.use_per and tree_idxs is not None:
            self.buffer.update_priorities(tree_idxs, td_errors.cpu().numpy())

        # ---- periodic target network update ----
        self._grad_step += 1
        if self._grad_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return {"loss": loss_val, "td_error": float(td_errors.abs().mean().item())}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        n_steps: int,
        max_ep_steps: int = 500,
        verbose_every: int = 5000,
    ) -> dict:
        """Train for ``n_steps`` environment steps.

        Returns
        -------
        dict with ``"returns"`` (per-episode) and ``"losses"`` lists.
        """
        all_returns: list = []
        all_losses: list = []

        obs, _ = env.reset()
        ep_return = 0.0
        ep_steps = 0

        for step in range(n_steps):
            a = self.select_action(obs, training=True)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            self.buffer.add(obs.flatten(), a, r, obs2.flatten(), terminated)

            ep_return += r
            ep_steps += 1
            self._step += 1

            metrics = self.update()
            if metrics:
                all_losses.append(metrics["loss"])

            # eps linear decay
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

            obs = obs2
            if done or ep_steps >= max_ep_steps:
                all_returns.append(ep_return)
                obs, _ = env.reset()
                ep_return = 0.0
                ep_steps = 0

            if verbose_every and (step + 1) % verbose_every == 0:
                avg_ret = np.mean(all_returns[-20:]) if all_returns else 0.0
                avg_loss = np.mean(all_losses[-100:]) if all_losses else 0.0
                print(
                    f"  DQN({self.variant}) step {step+1:>7}/{n_steps}  "
                    f"avg_ret={avg_ret:.3f}  loss={avg_loss:.4f}  eps={self.epsilon:.4f}"
                )

        return {"returns": all_returns, "losses": all_losses}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step": self._step,
                "grad_step": self._grad_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self._step = ckpt.get("step", 0)
        self._grad_step = ckpt.get("grad_step", 0)
