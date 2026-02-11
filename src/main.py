import os
import json
import numpy as np
from src.utils.viz import policy_to_grid, print_policy_grid, save_policy_map_png, save_value_heatmap_png

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
    
        # ---- Visualization ----
    # blocked + terminal sets (for display)
    blocked_states = getattr(mdp, "_blocked_states", set())
    terminal_states = getattr(mdp, "_terminal_states", set())

    grid = policy_to_grid(
        policy=result.policy,
        rows=mdp.rows,
        cols=mdp.cols,
        blocked_states=blocked_states,
        terminal_states=terminal_states
    )

    print("\nPolicy Map (terminal view):")
    print_policy_grid(
        grid,
        depot_rc=mdp.spec.depot,
        goal_rc=mdp.spec.customer
    )

    save_policy_map_png(grid, mdp.spec.depot, mdp.spec.customer, "outputs/plots/policy_map.png")
    save_value_heatmap_png(result.V, mdp.rows, mdp.cols, blocked_states, "outputs/plots/value_heatmap.png")

    print("\nSaved plots:")
    print(" - outputs/plots/policy_map.png")
    print(" - outputs/plots/value_heatmap.png")

    print("Saved outputs to outputs/.")


if __name__ == "__main__":
    main()
