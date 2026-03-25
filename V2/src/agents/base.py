"""Base agent interface shared by all classical and deep RL agents."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Common interface for all agents in this framework."""

    @abstractmethod
    def select_action(self, obs: Any, training: bool = True) -> int:
        """Return an action given an observation.

        Parameters
        ----------
        obs:
            Raw observation from the environment.
        training:
            If True the agent may explore (e.g. eps-greedy).
            If False use the greedy / deterministic policy.
        """

    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """Perform one learning update and return a dict of scalar metrics."""

    def save(self, path: str) -> None:
        """Persist agent state to *path*.  Override in subclasses."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Restore agent state from *path*.  Override in subclasses."""
        raise NotImplementedError
