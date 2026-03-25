import numpy as np
import gymnasium as gym
import highway_env

from src.envs.make_env import make_env
from src.mdp.abstractions.highway_tabular import obs_to_tabular_state, tabular_state_space_size
from src.mdp.model_learning import estimate_mdp
from src.dp.policy_iteration import policy_iteration

def main():
    env = make_env("highway-v0")
    nA = env.action_space.n
    nS = tabular_state_space_size(max_lanes=5)

    print("Learning approximate MDP model from foundation env...")
    P, R = estimate_mdp(env, obs_to_tabular_state, nS=nS, nA=nA, episodes=200, max_steps=300)

    # valid actions are all discrete actions for every abstract state
    valid_actions = [list(range(nA)) for _ in range(nS)]
    terminal_mask = np.zeros(nS, dtype=bool)  # we treat termination as part of reward signal here

    states = list(range(nS))
    result = policy_iteration(
        states=states,
        nS=nS,
        nA=nA,
        valid_actions=valid_actions,
        P=P,
        R=R,
        gamma=0.95,
        terminal_mask=terminal_mask,
        max_outer_iter=50
    )

    print("DP on learned model completed in", result.iterations, "iterations.")
    print("Sample of policy actions:", result.policy[:20])
    env.close()

if __name__ == "__main__":
    main()
