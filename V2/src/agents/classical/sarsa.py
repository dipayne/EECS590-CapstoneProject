"""SARSA family of on-policy TD control algorithms.

Implements:
  - SARSA(0)         — one-step on-policy TD control
  - SARSA(n)         — n-step on-policy control (forward view)
  - SARSA(lam) forward — offline lam-return SARSA
  - SARSA(lam) backward— online eligibility-trace SARSA (accumulating / replacing)
  - n-cutoff variant is parameterised via ``n`` / ``n_cutoff``.

Reference: Sutton & Barto, Chapter 7 & 12.
"""
from __future__ import annotations

import numpy as np
from typing import Callable

from src.agents.base import BaseAgent


class SARSAAgent(BaseAgent):
    """On-policy SARSA family agent with eps-greedy exploration.

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
    n:
        n-step return depth for SARSA(n).
    lam:
        lam for SARSA(lam).
    mode:
        One of ``"sarsa0"``, ``"sarsan"``, ``"sarsa_lam_forward"``,
        ``"sarsa_lam_backward"``.
    epsilon, epsilon_decay, epsilon_min:
        eps-greedy schedule.
    trace_type:
        ``"accumulating"`` or ``"replacing"`` traces.
    n_cutoff:
        Truncation depth for forward-view lam-return.
    """

    def __init__(
        self,
        nS: int,
        nA: int,
        obs_to_state: Callable,
        gamma: float = 0.99,
        alpha: float = 0.1,
        n: int = 5,
        lam: float = 0.9,
        mode: str = "sarsa_lam_backward",
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05,
        trace_type: str = "accumulating",
        n_cutoff: int = 50,
    ):
        valid_modes = {"sarsa0", "sarsan", "sarsa_lam_forward", "sarsa_lam_backward"}
        assert mode in valid_modes, f"mode must be one of {valid_modes}"

        self.nS = nS
        self.nA = nA
        self.obs_to_state = obs_to_state
        self.gamma = gamma
        self.alpha = alpha
        self.n = n
        self.lam = lam
        self.mode = mode
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.trace_type = trace_type
        self.n_cutoff = n_cutoff

        self.Q = np.zeros((nS, nA), dtype=np.float64)
        # Eligibility traces (state-action level)
        self._E = np.zeros((nS, nA), dtype=np.float64)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs, training: bool = True) -> int:
        s = self.obs_to_state(obs)
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.nA))
        return int(np.argmax(self.Q[s]))

    def update(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Use train() for SARSA agents.")

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    def train(self, env, n_episodes: int, max_steps: int = 1000,
              verbose_every: int = 100) -> list:
        dispatch = {
            "sarsa0": self._run_sarsa0,
            "sarsan": self._run_sarsan,
            "sarsa_lam_forward": self._run_sarsa_lam_forward,
            "sarsa_lam_backward": self._run_sarsa_lam_backward,
        }
        runner = dispatch[self.mode]
        returns = []

        for ep in range(n_episodes):
            ep_return = runner(env, max_steps)
            returns.append(ep_return)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose_every and (ep + 1) % verbose_every == 0:
                avg = np.mean(returns[-verbose_every:])
                print(
                    f"  SARSA({self.mode}) ep {ep+1:>5}/{n_episodes}  "
                    f"avg_return={avg:.3f}  eps={self.epsilon:.4f}"
                )

        return returns

    # ------------------------------------------------------------------
    # SARSA(0)
    # ------------------------------------------------------------------

    def _run_sarsa0(self, env, max_steps: int) -> float:
        obs, _ = env.reset()
        s = self.obs_to_state(obs)
        a = self.select_action(obs, training=True)
        total = 0.0
        gamma_t = 1.0

        for _ in range(max_steps):
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = self.obs_to_state(obs2)
            a2 = self.select_action(obs2, training=True)

            # SARSA update: on-policy (use next action a2, not greedy)
            q_next = 0.0 if terminated else self.Q[s2, a2]
            td_err = r + self.gamma * q_next - self.Q[s, a]
            self.Q[s, a] += self.alpha * td_err

            total += gamma_t * r
            gamma_t *= self.gamma
            s, a, obs = s2, a2, obs2

            if terminated or truncated:
                break

        return total

    # ------------------------------------------------------------------
    # SARSA(n) — n-step on-policy return (forward view)
    # ------------------------------------------------------------------

    def _run_sarsan(self, env, max_steps: int) -> float:
        """n-step SARSA: store episode, then apply n-step TD targets."""
        obs, _ = env.reset()
        states = [self.obs_to_state(obs)]
        actions = [self.select_action(obs, training=True)]
        rewards = []
        obs_list = [obs]
        total = 0.0
        gamma_t = 1.0

        for _ in range(max_steps):
            obs2, r, terminated, truncated, _ = env.step(actions[-1])
            s2 = self.obs_to_state(obs2)
            a2 = self.select_action(obs2, training=True)
            rewards.append(r)
            states.append(s2)
            actions.append(a2)
            obs_list.append(obs2)
            total += gamma_t * r
            gamma_t *= self.gamma
            if terminated or truncated:
                break

        T = len(rewards)
        for t in range(T):
            # n-step return G_{t:t+n}
            G = sum(
                self.gamma ** k * rewards[t + k]
                for k in range(min(self.n, T - t))
            )
            t_n = t + self.n
            if t_n < T:
                G += self.gamma ** self.n * self.Q[states[t_n], actions[t_n]]
            self.Q[states[t], actions[t]] += self.alpha * (
                G - self.Q[states[t], actions[t]]
            )

        return total

    # ------------------------------------------------------------------
    # SARSA(lam) forward view — offline
    # ------------------------------------------------------------------

    def _run_sarsa_lam_forward(self, env, max_steps: int) -> float:
        """Offline lam-return SARSA (forward view, truncated at n_cutoff)."""
        obs, _ = env.reset()
        states = [self.obs_to_state(obs)]
        actions = [self.select_action(obs, training=True)]
        rewards = []
        total = 0.0
        gamma_t = 1.0

        for _ in range(max_steps):
            obs2, r, terminated, truncated, _ = env.step(actions[-1])
            s2 = self.obs_to_state(obs2)
            a2 = self.select_action(obs2, training=True)
            rewards.append(r)
            states.append(s2)
            actions.append(a2)
            total += gamma_t * r
            gamma_t *= self.gamma
            if terminated or truncated:
                break

        T = len(rewards)

        def G_n(t, n):
            g = sum(self.gamma ** k * rewards[t + k]
                    for k in range(min(n, T - t)))
            t_n = t + n
            if t_n < T:
                g += self.gamma ** n * self.Q[states[t_n], actions[t_n]]
            return g

        for t in range(T):
            max_n = min(self.n_cutoff, T - t)
            lam_return = 0.0
            for nk in range(1, max_n):
                lam_return += (1 - self.lam) * self.lam ** (nk - 1) * G_n(t, nk)
            lam_return += self.lam ** (max_n - 1) * G_n(t, max_n)
            self.Q[states[t], actions[t]] += self.alpha * (
                lam_return - self.Q[states[t], actions[t]]
            )

        return total

    # ------------------------------------------------------------------
    # SARSA(lam) backward view — online with eligibility traces
    # ------------------------------------------------------------------

    def _run_sarsa_lam_backward(self, env, max_steps: int) -> float:
        """Online SARSA(lam) with accumulating or replacing eligibility traces."""
        self._E[:] = 0.0
        obs, _ = env.reset()
        s = self.obs_to_state(obs)
        a = self.select_action(obs, training=True)
        total = 0.0
        gamma_t = 1.0

        for _ in range(max_steps):
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = self.obs_to_state(obs2)
            a2 = self.select_action(obs2, training=True)

            q_next = 0.0 if terminated else self.Q[s2, a2]
            td_err = r + self.gamma * q_next - self.Q[s, a]

            # Eligibility trace
            if self.trace_type == "accumulating":
                self._E[s, a] += 1.0
            else:  # replacing
                self._E[s, a] = 1.0

            # Global Q and E update
            self.Q += self.alpha * td_err * self._E
            self._E *= self.gamma * self.lam

            total += gamma_t * r
            gamma_t *= self.gamma
            s, a, obs = s2, a2, obs2

            if terminated or truncated:
                break

        return total
