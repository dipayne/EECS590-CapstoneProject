"""Monte Carlo prediction and control algorithms.

Implements:
  - First-visit MC prediction  (value estimation)
  - Every-visit  MC prediction
  - First-visit MC control     (on-policy, eps-greedy)

All algorithms are model-free; they require only the ability to run
episodes in a Gym-compatible environment together with a state abstraction
function ``obs_to_state(obs) -> int``.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Callable, Optional

from src.agents.base import BaseAgent


class MonteCarloAgent(BaseAgent):
    """On-policy Monte Carlo control with eps-greedy exploration.

    Supports both *first-visit* and *every-visit* variants and
    exposes the V / Q tables after each episode for inspection.

    Parameters
    ----------
    nS, nA:
        Number of states / actions.
    obs_to_state:
        Maps a raw environment observation to a tabular state index.
    gamma:
        Discount factor.
    epsilon:
        Initial eps for eps-greedy exploration.
    epsilon_decay:
        Multiplicative decay applied to eps after every episode.
    epsilon_min:
        Floor on eps.
    first_visit:
        If True use first-visit MC; otherwise every-visit.
    """

    def __init__(
        self,
        nS: int,
        nA: int,
        obs_to_state: Callable,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05,
        first_visit: bool = True,
    ):
        self.nS = nS
        self.nA = nA
        self.obs_to_state = obs_to_state
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.first_visit = first_visit

        # Q-table and incremental return tracking
        self.Q = np.zeros((nS, nA), dtype=np.float64)
        self._returns_sum = np.zeros((nS, nA), dtype=np.float64)
        self._returns_cnt = np.zeros((nS, nA), dtype=np.int64)

        # Value function (derived from Q after each episode)
        self.V = np.zeros(nS, dtype=np.float64)

        self._episode_count = 0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs, training: bool = True) -> int:
        s = self.obs_to_state(obs)
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.nA))
        return int(np.argmax(self.Q[s]))

    def update(self, episode: list) -> dict:
        """Process one complete episode and update Q.

        Parameters
        ----------
        episode:
            List of (obs, action, reward) tuples for the full episode.

        Returns
        -------
        dict with keys ``"episode_return"`` and ``"epsilon"``.
        """
        # Convert raw observations to state indices
        states = [self.obs_to_state(obs) for obs, _, _ in episode]
        actions = [a for _, a, _ in episode]
        rewards = [r for _, _, r in episode]

        T = len(episode)
        G = 0.0
        visited = set()

        for t in reversed(range(T)):
            G = self.gamma * G + rewards[t]
            sa = (states[t], actions[t])

            if self.first_visit and sa in visited:
                continue
            visited.add(sa)

            s, a = sa
            self._returns_sum[s, a] += G
            self._returns_cnt[s, a] += 1
            self.Q[s, a] = self._returns_sum[s, a] / self._returns_cnt[s, a]

        # Sync V
        self.V = np.max(self.Q, axis=1)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episode_count += 1

        ep_return = sum(
            self.gamma ** t * rewards[t] for t in range(T)
        )
        return {"episode_return": ep_return, "epsilon": self.epsilon}

    # ------------------------------------------------------------------
    # Convenience: run a full training loop
    # ------------------------------------------------------------------

    def train(self, env, n_episodes: int, max_steps: int = 1000,
              verbose_every: int = 100) -> list:
        """Run ``n_episodes`` of MC control.

        Returns a list of episode returns for plotting.
        """
        returns = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            episode = []
            for _ in range(max_steps):
                a = self.select_action(obs, training=True)
                obs2, r, terminated, truncated, _ = env.step(a)
                episode.append((obs, a, r))
                obs = obs2
                if terminated or truncated:
                    break

            metrics = self.update(episode)
            returns.append(metrics["episode_return"])

            if verbose_every and (ep + 1) % verbose_every == 0:
                avg = np.mean(returns[-verbose_every:])
                print(
                    f"  MC ep {ep+1:>5}/{n_episodes}  "
                    f"avg_return={avg:.3f}  eps={self.epsilon:.4f}"
                )

        return returns

    # ------------------------------------------------------------------
    # MC Prediction (separate from control — for value estimation only)
    # ------------------------------------------------------------------

    def predict(self, env, policy: np.ndarray, n_episodes: int,
                max_steps: int = 1000) -> np.ndarray:
        """First-visit / every-visit MC *prediction* for a fixed policy.

        Parameters
        ----------
        policy:
            Deterministic policy array of shape (nS,) with action indices.

        Returns
        -------
        V  — estimated value function of shape (nS,).
        """
        returns_sum = np.zeros(self.nS, dtype=np.float64)
        returns_cnt = np.zeros(self.nS, dtype=np.int64)

        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode = []
            for _ in range(max_steps):
                s = self.obs_to_state(obs)
                a = int(policy[s])
                obs2, r, terminated, truncated, _ = env.step(a)
                episode.append((s, r))
                obs = obs2
                if terminated or truncated:
                    break

            T = len(episode)
            G = 0.0
            visited = set()
            for t in reversed(range(T)):
                s, r = episode[t]
                G = self.gamma * G + r
                if self.first_visit and s in visited:
                    continue
                visited.add(s)
                returns_sum[s] += G
                returns_cnt[s] += 1

        V = np.where(returns_cnt > 0, returns_sum / returns_cnt, 0.0)
        return V
