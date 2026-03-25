"""Temporal-Difference learning agents.

Implements:
  - TD(0)  — one-step TD prediction
  - TD(n)  — n-step return prediction (forward view)
  - TD(lam)  — forward view  (truncated lam-return)
  - TD(lam)  — backward view (online eligibility-trace update)

All methods operate on a *state-value function* V and accept an
external eps-greedy policy for action selection (or random exploration).

Reference: Sutton & Barto, Chapters 6–7.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from src.agents.base import BaseAgent


class TDAgent(BaseAgent):
    """TD-family value-estimation agent.

    Parameters
    ----------
    nS, nA:
        Number of states / actions.
    obs_to_state:
        Maps raw observation -> integer state index.
    gamma:
        Discount factor.
    alpha:
        Step-size (learning rate).
    n:
        n-step return length  (used by TD(n); ignored for TD(0) and TD(lam)).
    lam:
        lam value for TD(lam)  (0 → TD(0),  1 → MC-like).
    mode:
        One of ``"td0"``, ``"tdn"``, ``"tdlam_forward"``,
        ``"tdlam_backward"``.
    epsilon:
        eps for eps-greedy action selection during training.
    epsilon_decay, epsilon_min:
        Decay schedule for eps.
    trace_type:
        ``"accumulating"`` or ``"replacing"`` eligibility traces.
    n_cutoff:
        Truncation depth for the forward lam-return (ignored in backward).
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
        mode: str = "tdlam_backward",
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01,
        trace_type: str = "accumulating",
        n_cutoff: int = 50,
    ):
        assert mode in {"td0", "tdn", "tdlam_forward", "tdlam_backward"}
        assert trace_type in {"accumulating", "replacing"}

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

        self.V = np.zeros(nS, dtype=np.float64)
        # Eligibility traces for backward view
        self._E = np.zeros(nS, dtype=np.float64)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.nA))
        # Value-based action selection (not Q-based; random tie-break)
        return int(np.random.randint(self.nA))

    def update(self, *args, **kwargs) -> dict:
        """Not used directly — use ``train()`` which dispatches by mode."""
        raise NotImplementedError("Call train() for TD agents.")

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    def train(self, env, n_episodes: int, max_steps: int = 1000,
              verbose_every: int = 100) -> list:
        """Run the selected TD variant for ``n_episodes``.

        Returns list of episode returns.
        """
        dispatch = {
            "td0": self._run_td0,
            "tdn": self._run_tdn,
            "tdlam_forward": self._run_tdlam_forward,
            "tdlam_backward": self._run_tdlam_backward,
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
                    f"  TD({self.mode}) ep {ep+1:>5}/{n_episodes}  "
                    f"avg_return={avg:.3f}  eps={self.epsilon:.4f}"
                )

        return returns

    # ------------------------------------------------------------------
    # TD(0)
    # ------------------------------------------------------------------

    def _run_td0(self, env, max_steps: int) -> float:
        obs, _ = env.reset()
        s = self.obs_to_state(obs)
        total = 0.0
        gamma_t = 1.0

        for _ in range(max_steps):
            a = self.select_action(obs, training=True)
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = self.obs_to_state(obs2)

            # TD(0) update
            v_next = 0.0 if terminated else self.V[s2]
            td_err = r + self.gamma * v_next - self.V[s]
            self.V[s] += self.alpha * td_err

            total += gamma_t * r
            gamma_t *= self.gamma
            s, obs = s2, obs2

            if terminated or truncated:
                break

        return total

    # ------------------------------------------------------------------
    # TD(n) — forward n-step return
    # ------------------------------------------------------------------

    def _run_tdn(self, env, max_steps: int) -> float:
        """n-step TD prediction (forward view)."""
        # Store entire episode then update offline using n-step returns
        obs, _ = env.reset()
        states, rewards = [self.obs_to_state(obs)], []
        obs_list = [obs]
        done = False
        total = 0.0
        gamma_t = 1.0

        for _ in range(max_steps):
            a = self.select_action(obs_list[-1], training=True)
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = self.obs_to_state(obs2)
            rewards.append(r)
            states.append(s2)
            obs_list.append(obs2)
            total += gamma_t * r
            gamma_t *= self.gamma
            done = terminated or truncated
            if done:
                break

        T = len(rewards)
        for t in range(T):
            # G_{t:t+n}
            G = sum(
                self.gamma ** k * rewards[t + k]
                for k in range(min(self.n, T - t))
            )
            t_n = t + self.n
            if t_n < T:
                G += self.gamma ** self.n * self.V[states[t_n]]
            # Update
            self.V[states[t]] += self.alpha * (G - self.V[states[t]])

        return total

    # ------------------------------------------------------------------
    # TD(lam) forward view — offline lam-return
    # ------------------------------------------------------------------

    def _run_tdlam_forward(self, env, max_steps: int) -> float:
        """Off-line lam-return TD(lam) (forward view, truncated at n_cutoff)."""
        obs, _ = env.reset()
        traj_s, traj_r = [self.obs_to_state(obs)], []
        total = 0.0
        gamma_t = 1.0

        for _ in range(max_steps):
            a = self.select_action(obs, training=True)
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = self.obs_to_state(obs2)
            traj_s.append(s2)
            traj_r.append(r)
            total += gamma_t * r
            gamma_t *= self.gamma
            obs = obs2
            if terminated or truncated:
                break

        T = len(traj_r)

        def G_n(t, n):
            g = sum(self.gamma ** k * traj_r[t + k]
                    for k in range(min(n, T - t)))
            t_n = t + n
            if t_n < T:
                g += self.gamma ** n * self.V[traj_s[t_n]]
            return g

        for t in range(T):
            max_n = min(self.n_cutoff, T - t)
            # lam-return: (1-lam) Σ_{n=1}^{T-t-1} lam^{n-1} G_n  +  lam^{T-t-1} G_{T-t}
            lam_return = 0.0
            for n in range(1, max_n):
                lam_return += (1 - self.lam) * self.lam ** (n - 1) * G_n(t, n)
            lam_return += self.lam ** (max_n - 1) * G_n(t, max_n)
            self.V[traj_s[t]] += self.alpha * (lam_return - self.V[traj_s[t]])

        return total

    # ------------------------------------------------------------------
    # TD(lam) backward view — online with eligibility traces
    # ------------------------------------------------------------------

    def _run_tdlam_backward(self, env, max_steps: int) -> float:
        """Online TD(lam) backward view with eligibility traces."""
        self._E[:] = 0.0
        obs, _ = env.reset()
        s = self.obs_to_state(obs)
        total = 0.0
        gamma_t = 1.0

        for _ in range(max_steps):
            a = self.select_action(obs, training=True)
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = self.obs_to_state(obs2)

            td_err = r + self.gamma * (0.0 if terminated else self.V[s2]) - self.V[s]

            # Eligibility trace update
            if self.trace_type == "accumulating":
                self._E[s] += 1.0
            else:  # replacing
                self._E[s] = 1.0

            # Global V and E update
            self.V += self.alpha * td_err * self._E
            self._E *= self.gamma * self.lam

            total += gamma_t * r
            gamma_t *= self.gamma
            s, obs = s2, obs2

            if terminated or truncated:
                break

        return total
