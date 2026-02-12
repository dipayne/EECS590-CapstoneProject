import gymnasium as gym
import highway_env  # registers highway-* envs

def make_env(env_id: str = "highway-v0", render_mode=None, **kwargs):
    """
    Central factory for creating the foundational environment.
    Keeping env creation in one place makes algorithms env-agnostic.
    """
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    return env
