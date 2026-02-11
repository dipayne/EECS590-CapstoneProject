import os
import json
import numpy as np

from src.envs.logistics_grid_mdp import LogisticsGridMDP
from src.dp.policy_iteration import policy_iteration


def ensure_dirs():
    os.makedirs("outputs/policies", exist_ok=True)
    os.makedirs("outputs/values", exist_ok=True)
    os.makedirs("outputs/qvalues", exist_ok=True)


def save_policy(policy: np.ndarray, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(policy.tolist(), f, indent=2)


def main():
    ensure_dirs()

    mdp = LogisticsGridMDP()
    states = mdp.states()

    valid_actions = [[] for _ in range(mdp.nS)]
    for s in range(mdp.nS):
        valid_actions[s] = mdp.actions(s)

    terminal_mask = np.zeros(mdp.nS, dtype=bool)
    for s in range(mdp.nS):
        terminal_mask[s] = mdp.is_terminal(s)

    result = policy_iteration(
        states=states,
        nS=mdp.nS,
        nA=mdp.nA,
        valid_actions=valid_actions,
        P=mdp.P,
        R=mdp.R,
        gamma=mdp.gamma,
        terminal_mask=terminal_mask
    )

    print(f"Policy Iteration completed in {result.iterations} iterations.")
    print("Value at depot:", result.V[mdp.start_state()])
    print("Value at customer:", result.V[mdp.goal_state()])

    save_policy(result.policy, "outputs/policies/best_policy.json")
    np.save("outputs/values/V.npy", result.V)
    np.save("outputs/qvalues/Q.npy", result.Q)

    print("Saved outputs to outputs/.")


if __name__ == "__main__":
    main()
