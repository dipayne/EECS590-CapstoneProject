"""Highway-env observation wrapper for deep RL agents.

Provides a thin ``gymnasium.Wrapper`` around ``highway-v0`` that:
  1. Flattens the (N_veh × N_feat) kinematics observation to a 1-D vector.
  2. Optionally scales features to [0, 1] via known domain bounds.
  3. Exposes ``obs_dim`` and ``n_actions`` as attributes so agent
     constructors can query them without hard-coding magic numbers.

The default configuration uses *Kinematics* observations with 5 vehicles
and 5 features each (presence, x, y, vx, vy), giving obs_dim = 25.

For classical RL (tabular agents) use the ``obs_to_tabular_state``
helper imported from the V1 abstraction module.
"""
from __future__ import annotations

import sys
import os
import numpy as np
import gymnasium as gym

# Make V1 tabular abstraction importable
_V1_SRC = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "src"
)
if _V1_SRC not in sys.path:
    sys.path.insert(0, _V1_SRC)

try:
    from mdp.abstractions.highway_tabular import (
        obs_to_tabular_state,
        tabular_state_space_size,
    )
    _TABULAR_AVAILABLE = True
except ImportError:
    _TABULAR_AVAILABLE = False

import highway_env  # registers highway-* envs


# ---------------------------------------------------------------------------
# Highway-env feature bounds (for optional normalisation)
# ---------------------------------------------------------------------------

_BOUNDS = {
    "presence": (0.0, 1.0),
    "x":        (-200.0, 200.0),
    "y":        (-4.0, 4.0),
    "vx":       (0.0, 40.0),
    "vy":       (-5.0, 5.0),
}

_FEATURE_ORDER = ["presence", "x", "y", "vx", "vy"]
_LOW  = np.array([_BOUNDS[f][0] for f in _FEATURE_ORDER], dtype=np.float32)
_HIGH = np.array([_BOUNDS[f][1] for f in _FEATURE_ORDER], dtype=np.float32)


# ---------------------------------------------------------------------------
# Default highway-v0 config used throughout V2
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": False,
        "absolute": False,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 10,
    "duration": 40,            # episode length (seconds)
    "reward_speed_range": [20, 30],
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "collision_reward": -1.0,
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "lane_change_reward": 0.0,
}


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class HighwayWrapper(gym.ObservationWrapper):
    """Flattens highway-env kinematics observations to 1-D numpy arrays.

    Parameters
    ----------
    env:
        Underlying highway-env Gymnasium environment.
    normalize:
        If True, scale each feature to [0, 1] using domain bounds.
    """

    def __init__(self, env: gym.Env, normalize: bool = False):
        super().__init__(env)
        self._normalize = normalize
        # Read config from unwrapped env to bypass OrderEnforcing wrapper
        base = env.unwrapped if hasattr(env, "unwrapped") else env
        obs_cfg = base.config["observation"]
        n_veh = obs_cfg["vehicles_count"]
        n_feat = len(obs_cfg["features"])
        self.obs_dim = n_veh * n_feat
        self.n_actions = env.action_space.n

        low  = np.full(self.obs_dim, -1.0 if normalize else -np.inf, dtype=np.float32)
        high = np.full(self.obs_dim,  1.0 if normalize else  np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        flat = obs.flatten().astype(np.float32)
        if self._normalize:
            # Repeat bounds for each vehicle row
            n_veh = obs.shape[0]
            lo = np.tile(_LOW, n_veh)
            hi = np.tile(_HIGH, n_veh)
            flat = (flat - lo) / (hi - lo + 1e-8)
            flat = np.clip(flat, 0.0, 1.0)
        return flat

    # ------------------------------------------------------------------
    # Tabular state (for classical RL agents)
    # ------------------------------------------------------------------

    def tabular_state(self, obs: np.ndarray) -> int:
        """Convert kinematics obs (flat or 2-D) to integer tabular state."""
        if not _TABULAR_AVAILABLE:
            raise RuntimeError(
                "V1 tabular abstraction not found.  "
                "Ensure the V1 src/ directory is accessible."
            )
        # HighwayWrapper returns flat (25,) obs; reshape to (n_veh, n_feat) for V1 fn
        if obs.ndim == 1:
            base = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            n_feat = len(base.config["observation"]["features"])
            obs = obs.reshape(-1, n_feat)
        return obs_to_tabular_state(obs)

    @property
    def n_tabular_states(self) -> int:
        if not _TABULAR_AVAILABLE:
            raise RuntimeError("V1 tabular abstraction not found.")
        return tabular_state_space_size()


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_highway_env(
    env_id: str = "highway-v0",
    render_mode: str | None = None,
    normalize_obs: bool = False,
    config_overrides: dict | None = None,
) -> HighwayWrapper:
    """Create and wrap a highway-env environment.

    Parameters
    ----------
    env_id:
        Gymnasium environment ID (default ``"highway-v0"``).
    render_mode:
        ``"human"`` for on-screen rendering, ``None`` for headless.
    normalize_obs:
        Whether to normalize observations to [0, 1].
    config_overrides:
        Dict of key-value pairs to update in ``DEFAULT_CONFIG``.

    Returns
    -------
    HighwayWrapper
        Wrapped environment with flat observation space.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config_overrides:
        cfg.update(config_overrides)

    env = gym.make(env_id, render_mode=render_mode)
    # gym.make() wraps with OrderEnforcing; unwrap to reach the base env for configure()
    base_env = env.unwrapped
    base_env.configure(cfg)
    env.reset()  # apply config so observation_space reflects new settings

    return HighwayWrapper(env, normalize=normalize_obs)
