"""Q-Learning and Double Q-Learning agents.

Implements:
  - Standard Q-Learning    (off-policy TD control, Watkins 1989)
  - Double Q-Learning      (reduces maximization bias, van Hasselt 2010)

Both are off-policy: the target uses the *greedy* next action, not
the action actually taken, decoupling behaviour and target policies.

Reference: Sutton & Barto, Chapter 6.5; van Hasselt 2010.
"""
from __future__ import annotations

import numpy as np
from typing import Callable

from src.agents.base import BaseAgent


class QLearningAgent(BaseAgent):
    """Q-Learning (standard and double) with eps-greedy exploration.

    Parameters
    ----------
    nS, nA:
        Tabular state / action counts.
    obs_to_state:
        Raw-obs → integer-state mapper.
    gamma:
        Discount factor.
    alpha:
        Step-size (learning rate).
    epsilon, epsilon_decay, epsilon_min:
        eps-greedy exploration schedule.
    double:
        If True, use Double Q-Learning (two separate Q-tables).
    """

    def __init__(
        self,
        nS: int,
        nA: int,
        obs_to_state: Callable,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05,
        double: bool = False,
    ):
        self.nS = nS
        self.nA = nA
        self.obs_to_state = obs_to_state
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.double = double

        self.Q = np.zeros((nS, nA), dtype=np.float64)
        # Second Q-table used only in Double Q-Learning
        self.Q2 = np.zeros((nS, nA), dtype=np.float64) if double else None

        self._step_count = 0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs, training: bool = True) -> int:
        s = self.obs_to_state(obs)
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.nA))
        # Off-policy: greedy w.r.t. average Q for double, Q for standard
        q_vals = (self.Q[s] + self.Q2[s]) / 2 if self.double else self.Q[s]
        return int(np.argmax(q_vals))

    def update(self, obs, action: int, reward: float, obs2,
               terminated: bool) -> dict:
        """Single-step update.  Call after every env.step().

        Returns dict with ``"td_error"`` for logging.
        """
        s = self.obs_to_state(obs)
        s2 = self.obs_to_state(obs2)

        if self.double:
            td_err = self._double_update(s, action, reward, s2, terminated)
        else:
            td_err = self._standard_update(s, action, reward, s2, terminated)

        self._step_count += 1
        return {"td_error": td_err}

    # ------------------------------------------------------------------
    # Standard Q-Learning update
    # ------------------------------------------------------------------

    def _standard_update(self, s, a, r, s2, terminated) -> float:
        """Q[s,a] ← Q[s,a] + alpha (r + gamma max_a' Q[s',a'] − Q[s,a])"""
        q_next = 0.0 if terminated else float(np.max(self.Q[s2]))
        target = r + self.gamma * q_next
        td_err = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_err
        return td_err

    # ------------------------------------------------------------------
    # Double Q-Learning update  (van Hasselt 2010)
    # ------------------------------------------------------------------

    def _double_update(self, s, a, r, s2, terminated) -> float:
        """Randomly assign to Q1 or Q2 to decouple selection / evaluation."""
        if np.random.rand() < 0.5:
            # Update Q1: select with Q1, evaluate with Q2
            a_star = int(np.argmax(self.Q[s2]))
            q_next = 0.0 if terminated else float(self.Q2[s2, a_star])
            target = r + self.gamma * q_next
            td_err = target - self.Q[s, a]
            self.Q[s, a] += self.alpha * td_err
        else:
            # Update Q2: select with Q2, evaluate with Q1
            a_star = int(np.argmax(self.Q2[s2]))
            q_next = 0.0 if terminated else float(self.Q[s2, a_star])
            target = r + self.gamma * q_next
            td_err = target - self.Q2[s, a]
            self.Q2[s, a] += self.alpha * td_err

        return td_err

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env, n_episodes: int, max_steps: int = 1000,
              verbose_every: int = 100) -> list:
        """Run Q-Learning for ``n_episodes``.

        Returns list of episode returns.
        """
        label = "Double-Q" if self.double else "Q-Learning"
        returns = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_return = 0.0
            gamma_t = 1.0

            for _ in range(max_steps):
                a = self.select_action(obs, training=True)
                obs2, r, terminated, truncated, _ = env.step(a)
                self.update(obs, a, r, obs2, terminated)

                ep_return += gamma_t * r
                gamma_t *= self.gamma
                obs = obs2
                if terminated or truncated:
                    break

            returns.append(ep_return)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose_every and (ep + 1) % verbose_every == 0:
                avg = np.mean(returns[-verbose_every:])
                print(
                    f"  {label} ep {ep+1:>5}/{n_episodes}  "
                    f"avg_return={avg:.3f}  eps={self.epsilon:.4f}"
                )

        return returns

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {"Q": self.Q}
        if self.double:
            data["Q2"] = self.Q2
        np.savez(path, **data)

    def load(self, path: str) -> None:
        data = np.load(path + ".npz")
        self.Q = data["Q"]
        if self.double and "Q2" in data:
            self.Q2 = data["Q2"]
