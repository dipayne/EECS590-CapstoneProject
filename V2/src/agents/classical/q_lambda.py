"""Q(lam) — Q-Learning with eligibility traces (Watkins' Q(lam)).

Watkins' Q(lam) is the off-policy analog of SARSA(lam).  The key difference:
  - SARSA(lam): traces are maintained for the full episode (on-policy).
  - Watkins' Q(lam): traces are **cut to zero** whenever a non-greedy action
    is taken.  This preserves the off-policy nature of Q-learning.

This "trace-cutting" rule makes Watkins' Q(lam) theoretically sound as an
off-policy method, at the cost of resetting credit flow on exploratory steps.

Reference: Watkins & Dayan, 1992. "Q-learning."
           Sutton & Barto, Chapter 12.7.
"""
from __future__ import annotations

import numpy as np
from typing import Callable

from src.agents.base import BaseAgent


class QLambdaAgent(BaseAgent):
    """Watkins' Q(lam) agent with eps-greedy exploration.

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
    lam:
        Eligibility trace decay.  0 → Q(0) ≡ Q-Learning, 1 → MC-like.
    epsilon, epsilon_decay, epsilon_min:
        eps-greedy exploration schedule.
    trace_type:
        ``"accumulating"`` or ``"replacing"`` eligibility traces.
        Watkins' Q(lam) also supports cutting, applied automatically when
        a non-greedy action is taken.
    """

    def __init__(
        self,
        nS: int,
        nA: int,
        obs_to_state: Callable,
        gamma: float = 0.99,
        alpha: float = 0.1,
        lam: float = 0.8,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05,
        trace_type: str = "replacing",
    ):
        self.nS = nS
        self.nA = nA
        self.obs_to_state = obs_to_state
        self.gamma = gamma
        self.alpha = alpha
        self.lam = lam
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.trace_type = trace_type

        self.Q = np.zeros((nS, nA), dtype=np.float64)
        self._E = np.zeros((nS, nA), dtype=np.float64)  # eligibility traces

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs, training: bool = True) -> int:
        s = self.obs_to_state(obs)
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.nA))
        return int(np.argmax(self.Q[s]))

    def _is_greedy(self, s: int, a: int) -> bool:
        """Return True if action a is greedy w.r.t. Q[s]."""
        return a == int(np.argmax(self.Q[s]))

    def update(self, obs, action: int, reward: float, obs2,
               terminated: bool) -> dict:
        """Single Watkins Q(lam) step — call after every env.step()."""
        s  = self.obs_to_state(obs)
        s2 = self.obs_to_state(obs2)

        # Off-policy target: greedy Q-value of next state
        q_next = 0.0 if terminated else float(np.max(self.Q[s2]))
        td_err = reward + self.gamma * q_next - self.Q[s, action]

        # Eligibility trace update
        if self.trace_type == "replacing":
            self._E[s, action] = 1.0
        else:
            self._E[s, action] += 1.0

        # Global update
        self.Q  += self.alpha * td_err * self._E

        # Watkins' trace-cutting: reset traces if non-greedy action was taken
        if not self._is_greedy(s, action):
            self._E[:] = 0.0
        else:
            self._E *= self.gamma * self.lam

        return {"td_error": td_err}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env, n_episodes: int, max_steps: int = 1000,
              verbose_every: int = 100) -> list:
        returns = []
        for ep in range(n_episodes):
            self._E[:] = 0.0
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
                    f"  Q(lam) ep {ep+1:>5}/{n_episodes}  "
                    f"avg_return={avg:.3f}  eps={self.epsilon:.4f}"
                )
        return returns

    def save(self, path: str) -> None:
        np.savez(path, Q=self.Q)

    def load(self, path: str) -> None:
        self.Q = np.load(path + ".npz")["Q"]
