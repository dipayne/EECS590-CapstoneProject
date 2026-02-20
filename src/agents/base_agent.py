from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    def __init__(self, nS: int, nA: int, gamma: float = 0.95):
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.policy = np.zeros(nS, dtype=int)
        self.V = np.zeros(nS, dtype=float)
        self.Q = np.zeros((nS, nA), dtype=float)

    @abstractmethod
    def train(self, **kwargs):
        """Train the agent."""

    @abstractmethod
    def act(self, state: int) -> int:
        """Select an action for the given state."""

    @abstractmethod
    def evaluate(self, episodes: int, max_steps: int) -> list:
        """Evaluate the current policy and return per-episode returns."""

    def save_policy(self, path: str) -> None:
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.policy.tolist(), f, indent=2)

    def load_policy(self, path: str) -> None:
        import json
        with open(path, "r", encoding="utf-8") as f:
            self.policy = np.array(json.load(f), dtype=int)

    def save_values(self, v_path: str, q_path: str) -> None:
        import os
        os.makedirs(os.path.dirname(v_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(q_path) or ".", exist_ok=True)
        np.save(v_path, self.V)
        np.save(q_path, self.Q)

    def load_values(self, v_path: str, q_path: str) -> None:
        self.V = np.load(v_path)
        self.Q = np.load(q_path)
