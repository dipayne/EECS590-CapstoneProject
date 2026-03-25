"""Replay buffer implementations.

Provides:
  - ReplayBuffer              — standard uniform circular replay buffer
  - PrioritizedReplayBuffer   — proportional PER (Schaul et al., 2016)

Both store (obs, action, reward, next_obs, done) transitions in
pre-allocated numpy arrays for memory efficiency.

Reference:
  Schaul et al., "Prioritized Experience Replay," ICLR 2016.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Standard replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular experience replay buffer with uniform sampling.

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored.
    obs_shape:
        Shape of a single observation (e.g. (25,) for flattened highway-env).
    """

    def __init__(self, capacity: int, obs_shape: tuple):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self._ptr = 0     # write pointer
        self._size = 0    # current fill level

        # Pre-allocate storage arrays
        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition."""
        self.obs[self._ptr]      = obs.flatten()
        self.next_obs[self._ptr] = next_obs.flatten()
        self.actions[self._ptr]  = action
        self.rewards[self._ptr]  = reward
        self.dones[self._ptr]    = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Return a random mini-batch of transitions."""
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist buffer to a compressed .npz file."""
        np.savez_compressed(
            path,
            obs=self.obs[: self._size],
            next_obs=self.next_obs[: self._size],
            actions=self.actions[: self._size],
            rewards=self.rewards[: self._size],
            dones=self.dones[: self._size],
            ptr=np.array([self._ptr]),
            size=np.array([self._size]),
        )

    def load(self, path: str) -> None:
        """Restore buffer state from a .npz file."""
        data = np.load(path + ".npz")
        n = int(data["size"][0])
        self.obs[:n]      = data["obs"]
        self.next_obs[:n] = data["next_obs"]
        self.actions[:n]  = data["actions"]
        self.rewards[:n]  = data["rewards"]
        self.dones[:n]    = data["dones"]
        self._ptr  = int(data["ptr"][0])
        self._size = n


# ---------------------------------------------------------------------------
# Prioritized experience replay (PER) — proportional variant
# ---------------------------------------------------------------------------

class _SumTree:
    """Binary sum-tree for O(log n) priority sampling and update."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._data_ptr = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self._tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, idx: int, priority: float) -> None:
        tree_idx = idx + self.capacity - 1
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def get(self, s: float) -> Tuple[int, int, float]:
        """Find leaf index, data index, and priority for sample value s."""
        node = 0
        while node < self.capacity - 1:
            left = 2 * node + 1
            if s <= self._tree[left]:
                node = left
            else:
                s -= self._tree[left]
                node = left + 1
        data_idx = node - self.capacity + 1
        return node, data_idx, self._tree[node]

    @property
    def total(self) -> float:
        return self._tree[0]


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (PER).

    Parameters
    ----------
    capacity:
        Maximum number of stored transitions.
    obs_shape:
        Shape of a single observation.
    alpha:
        Priority exponent  (0 → uniform, 1 → full prioritisation).
    beta_start:
        Initial importance-sampling correction exponent.
    beta_end:
        Final β (annealed towards 1 over training).
    beta_steps:
        Number of steps over which to anneal β.
    eps:
        Small constant added to TD-errors before computing priorities.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100_000,
        eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_inc = (beta_end - beta_start) / beta_steps
        self.eps = eps

        self._tree = _SumTree(capacity)
        self._ptr = 0
        self._size = 0
        self._max_priority = 1.0

        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self._ptr]      = obs.flatten()
        self.next_obs[self._ptr] = next_obs.flatten()
        self.actions[self._ptr]  = action
        self.rewards[self._ptr]  = reward
        self.dones[self._ptr]    = float(done)

        # New transitions get max priority
        self._tree.update(self._ptr, self._max_priority ** self.alpha)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """Sample a batch; returns (obs, acts, rews, next_obs, dones, weights, idxs)."""
        idxs, tree_idxs, priorities = [], [], []
        segment = self._tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            tree_idx, data_idx, pri = self._tree.get(s)
            idxs.append(data_idx)
            tree_idxs.append(tree_idx)
            priorities.append(pri)

        idxs = np.array(idxs)
        priorities = np.array(priorities, dtype=np.float64)

        # Importance-sampling weights
        probs = priorities / self._tree.total
        weights = (self._size * probs) ** (-self.beta)
        weights /= weights.max()  # normalise

        # Anneal beta
        self.beta = min(self.beta_end, self.beta + self.beta_inc)

        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs],
            weights.astype(np.float32),
            np.array(tree_idxs),
        )

    def update_priorities(self, tree_idxs: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities from new TD-errors after gradient step.

        tree_idxs are the raw tree node indices returned by sample().
        Convert to data indices before calling _tree.update().
        """
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for ti, p in zip(tree_idxs, priorities):
            data_idx = int(ti) - self._tree.capacity + 1
            self._tree.update(data_idx, p)
            self._max_priority = max(self._max_priority, p)

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size
