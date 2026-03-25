"""Linear Function Approximation + Semi-Gradient TD methods.

Implements Q-value approximation using a linear model:

    Q(s, a; w) ≈ w_a^T φ(s)

where φ(s) is a hand-crafted feature vector derived from the tabular
state representation.  This bridges tabular methods and deep RL.

Algorithms
----------
1. **Semi-gradient SARSA** with linear FA (Sutton & Barto §10.1)
2. **Gradient TD(0)** (semi-gradient TD) for prediction with linear FA

Feature engineering for highway-v0
-----------------------------------
The V1 tabular state is encoded as (lane, speed_bin, front_bin, rear_bin).
We expand this into a one-hot feature vector of length nS, making the
linear approximation equivalent to the tabular case.  Additional hand-crafted
features (tile coding, polynomial) can be plugged in via ``feature_fn``.

Reference: Sutton & Barto, Chapter 9–10.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from src.agents.base import BaseAgent


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def one_hot_features(state: int, nS: int) -> np.ndarray:
    """One-hot encoding: equivalent to tabular representation."""
    phi = np.zeros(nS, dtype=np.float64)
    phi[state] = 1.0
    return phi


def polynomial_features(state: int, nS: int, degree: int = 2) -> np.ndarray:
    """Polynomial features over normalized state index [0, 1].

    For a 1-D state index this is a simple basis expansion.
    """
    x = state / max(nS - 1, 1)  # normalize to [0, 1]
    return np.array([x ** d for d in range(degree + 1)], dtype=np.float64)


def tile_coding_features(state: int, nS: int,
                         n_tilings: int = 8,
                         tiles_per_tiling: int = 16) -> np.ndarray:
    """Simple tile-coding feature vector (1-D state).

    Each tiling partitions the [0, nS] range differently via an offset.
    Returns a binary vector of length n_tilings * tiles_per_tiling.
    """
    feat_dim = n_tilings * tiles_per_tiling
    phi = np.zeros(feat_dim, dtype=np.float64)
    x = float(state)
    for t in range(n_tilings):
        offset = t * (nS / (n_tilings * tiles_per_tiling))
        tile_idx = int((x + offset) / nS * tiles_per_tiling) % tiles_per_tiling
        phi[t * tiles_per_tiling + tile_idx] = 1.0
    return phi


# ---------------------------------------------------------------------------
# Linear FA Agent (Semi-Gradient SARSA + Gradient TD)
# ---------------------------------------------------------------------------

class LinearFAAgent(BaseAgent):
    """Semi-gradient SARSA and Gradient TD(0) with linear Q-function approximation.

    Parameters
    ----------
    nS, nA:
        Tabular state / action counts.
    obs_to_state:
        Raw-obs → integer-state mapper.
    feature_fn:
        Function ``(state: int, nS: int) -> np.ndarray`` producing φ(s).
        Defaults to one-hot encoding (equivalent to tabular).
    gamma:
        Discount factor.
    alpha:
        Step-size (learning rate) — critical for convergence with linear FA.
    method:
        ``"sarsa"`` (on-policy control) or ``"td"`` (prediction).
    epsilon, epsilon_decay, epsilon_min:
        eps-greedy schedule.
    """

    def __init__(
        self,
        nS: int,
        nA: int,
        obs_to_state: Callable,
        feature_fn: Optional[Callable] = None,
        gamma: float = 0.99,
        alpha: float = 0.01,
        method: str = "sarsa",
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05,
    ):
        assert method in {"sarsa", "td"}
        self.nS = nS
        self.nA = nA
        self.obs_to_state = obs_to_state
        self.gamma = gamma
        self.alpha = alpha
        self.method = method
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self._feature_fn = feature_fn or one_hot_features

        # Probe feature dimension
        sample_phi = self._feature_fn(0, nS)
        self.feat_dim = len(sample_phi)

        # Weight matrix: one weight vector per action
        self.W = np.zeros((nA, self.feat_dim), dtype=np.float64)

    # ------------------------------------------------------------------
    # Q-value computation
    # ------------------------------------------------------------------

    def _phi(self, s: int) -> np.ndarray:
        return self._feature_fn(s, self.nS)

    def _q(self, s: int, a: int) -> float:
        return float(self.W[a] @ self._phi(s))

    def _q_all(self, s: int) -> np.ndarray:
        phi = self._phi(s)
        return self.W @ phi  # (nA,)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs, training: bool = True) -> int:
        s = self.obs_to_state(obs)
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.nA))
        return int(np.argmax(self._q_all(s)))

    def update(self, obs, action: int, reward: float, obs2,
               terminated: bool, action2: Optional[int] = None) -> dict:
        """Semi-gradient TD update for one step.

        Parameters
        ----------
        action2:
            Next action (required for SARSA; ignored for TD prediction).
        """
        s  = self.obs_to_state(obs)
        s2 = self.obs_to_state(obs2)
        phi_s = self._phi(s)

        if self.method == "sarsa":
            q_next = 0.0 if terminated else self._q(s2, action2 or 0)
            td_err = reward + self.gamma * q_next - self._q(s, action)
            # Semi-gradient: gradient of Q(s,a;w) w.r.t. w_a is φ(s)
            self.W[action] += self.alpha * td_err * phi_s
        else:
            # TD(0) prediction: we don't have per-action weights in value mode
            # Use action=0 as proxy (pure prediction — not control)
            q_next = 0.0 if terminated else float(np.max(self._q_all(s2)))
            td_err = reward + self.gamma * q_next - self._q(s, action)
            self.W[action] += self.alpha * td_err * phi_s

        return {"td_error": td_err}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env, n_episodes: int, max_steps: int = 1000,
              verbose_every: int = 100) -> list:
        """Semi-gradient SARSA or Gradient TD(0) control loop."""
        returns = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            a = self.select_action(obs, training=True)
            ep_return = 0.0
            gamma_t = 1.0

            for _ in range(max_steps):
                obs2, r, terminated, truncated, _ = env.step(a)
                a2 = self.select_action(obs2, training=True)
                self.update(obs, a, r, obs2, terminated, action2=a2)

                ep_return += gamma_t * r
                gamma_t *= self.gamma
                obs, a = obs2, a2
                if terminated or truncated:
                    break

            returns.append(ep_return)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose_every and (ep + 1) % verbose_every == 0:
                avg = np.mean(returns[-verbose_every:])
                feat_name = getattr(self._feature_fn, "__name__", "custom")
                print(
                    f"  LinearFA({self.method}, {feat_name}) "
                    f"ep {ep+1:>5}/{n_episodes}  "
                    f"avg_return={avg:.3f}  eps={self.epsilon:.4f}"
                )
        return returns

    # ------------------------------------------------------------------
    # Derived Q / V tables (for comparison with tabular agents)
    # ------------------------------------------------------------------

    def to_Q_table(self) -> np.ndarray:
        """Materialise a (nS, nA) Q-table from the linear weights."""
        Q = np.zeros((self.nS, self.nA), dtype=np.float64)
        for s in range(self.nS):
            Q[s] = self._q_all(s)
        return Q
