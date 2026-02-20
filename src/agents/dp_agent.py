from __future__ import annotations
import argparse
import random
import numpy as np

from src.agents.base_agent import BaseAgent
from src.dp.policy_iteration import policy_iteration


class DPAgent(BaseAgent):
    """
    Policy Iteration agent for tabular MDPs.

    Hyperparameters
    ---------------
    gamma       : float  -- discount factor
    theta       : float  -- convergence threshold for policy evaluation
    max_outer_iter : int -- max policy iteration sweeps
    """

    def __init__(self, mdp, gamma: float = 0.95,
                 theta: float = 1e-8, max_outer_iter: int = 1000):
        super().__init__(nS=mdp.nS, nA=mdp.nA, gamma=gamma)
        self.mdp = mdp
        self.theta = theta
        self.max_outer_iter = max_outer_iter

        self.valid_actions = [mdp.actions(s) for s in range(mdp.nS)]
        self.terminal_mask = np.array(
            [mdp.is_terminal(s) for s in range(mdp.nS)], dtype=bool
        )
        self._trained = False

    def train(self, **kwargs):
        """Run policy iteration on the MDP and store the converged policy, V, and Q."""
        result = policy_iteration(
            states=self.mdp.states(),
            nS=self.mdp.nS,
            nA=self.mdp.nA,
            valid_actions=self.valid_actions,
            P=self.mdp.P,
            R=self.mdp.R,
            gamma=self.gamma,
            terminal_mask=self.terminal_mask,
            max_outer_iter=self.max_outer_iter,
        )
        self.policy = result.policy
        self.V = result.V
        self.Q = result.Q
        self._trained = True
        print(f"[DPAgent] Policy iteration converged in {result.iterations} iterations.")
        return result

    def act(self, state: int) -> int:
        """Return the greedy action for the given state."""
        return int(self.policy[state])

    def evaluate(self, episodes: int = 10, max_steps: int = 500) -> list:
        """
        Simulate episodes using the current policy and the MDP transition model.
        Returns a list of discounted returns, one per episode.
        """
        if not self._trained:
            print("[DPAgent] Warning: policy has not been trained yet. Using default policy.")

        mdp = self.mdp
        returns = []

        for ep in range(episodes):
            s = mdp.start_state()
            total = 0.0
            for t in range(max_steps):
                a = self.act(s)
                trans = mdp.transitions(s, a)
                s_next = random.choices(
                    list(trans.keys()), weights=list(trans.values())
                )[0]
                r = mdp.reward(s, a, s_next)
                total += (self.gamma ** t) * r
                s = s_next
                if mdp.is_terminal(s):
                    break
            returns.append(total)
            print(f"  Episode {ep + 1}/{episodes}: discounted return = {total:.3f}")

        print(f"[DPAgent] Mean return over {episodes} episodes: {float(np.mean(returns)):.3f}")
        return returns


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate DPAgent on the Logistics Grid MDP"
    )
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both",
                        help="Whether to train, evaluate, or do both (default: both)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--theta", type=float, default=1e-8,
                        help="Convergence threshold for policy evaluation")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Max policy iteration sweeps")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--save", action="store_true",
                        help="Save policy, V, and Q after training")
    parser.add_argument("--load", action="store_true",
                        help="Load a pre-trained policy from outputs/ before evaluating")
    args = parser.parse_args()

    from src.envs.logistics_grid_mdp import LogisticsGridMDP
    mdp = LogisticsGridMDP()
    agent = DPAgent(mdp, gamma=args.gamma, theta=args.theta, max_outer_iter=args.max_iter)

    if args.load:
        agent.load_policy("outputs/policies/best_policy.json")
        agent.load_values("outputs/values/V.npy", "outputs/qvalues/Q.npy")
        agent._trained = True
        print("[DPAgent] Loaded pre-trained policy and values from outputs/.")

    if args.mode in ("train", "both"):
        print("Training...")
        agent.train()
        if args.save:
            agent.save_policy("outputs/policies/best_policy.json")
            agent.save_values("outputs/values/V.npy", "outputs/qvalues/Q.npy")
            print("[DPAgent] Saved policy and values to outputs/.")

    if args.mode in ("eval", "both"):
        print("Evaluating...")
        agent.evaluate(episodes=args.episodes, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
