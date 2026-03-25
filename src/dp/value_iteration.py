"""Value Iteration — Dynamic Programming (Bellman optimality backup).

Value Iteration directly solves the Bellman optimality equations:

    V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V*(s')]

by iterating this backup until convergence (|V_new - V_old|_∞ < θ).
Policy extraction is then a one-shot greedy step over the optimal V*.

Difference from Policy Iteration:
  - Policy Iteration alternates policy evaluation (full convergence) and
    policy improvement.
  - Value Iteration combines both into a single backup step per iteration,
    which is often faster in practice.

Reference: Sutton & Barto, Chapter 4.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ValueIterationResult:
    policy: np.ndarray
    V: np.ndarray
    Q: np.ndarray
    iterations: int


def value_iteration(
    states: List[int],
    nS: int,
    nA: int,
    valid_actions: List[List[int]],
    P,
    R,
    gamma: float,
    theta: float = 1e-8,
    max_iter: int = 10_000,
    terminal_mask: Optional[np.ndarray] = None,
) -> ValueIterationResult:
    """Run Value Iteration to convergence.

    Parameters
    ----------
    states:
        List of state indices to iterate over.
    nS, nA:
        Number of states / actions.
    valid_actions:
        ``valid_actions[s]`` — list of valid action indices for state s.
    P:
        Transition function.  ``P[s][a]`` → dict ``{s': probability}``.
    R:
        Reward function.  ``R[s][a]`` → dict ``{s': expected_reward}``.
    gamma:
        Discount factor.
    theta:
        Convergence threshold.
    max_iter:
        Maximum number of sweeps.
    terminal_mask:
        Boolean array of length nS; True entries are skipped.

    Returns
    -------
    ValueIterationResult
        Contains optimal policy, V, Q, and iteration count.
    """
    V = np.zeros(nS, dtype=float)

    for it in range(max_iter):
        delta = 0.0
        for s in states:
            if terminal_mask is not None and terminal_mask[s]:
                continue
            acts = valid_actions[s]
            if not acts:
                continue

            v_old = V[s]
            # Bellman optimality backup: take max over actions
            best = -1e30
            for a in acts:
                val = 0.0
                for s_next, p in P[s][a].items():
                    val += p * (R[s][a][s_next] + gamma * V[s_next])
                best = max(best, val)
            V[s] = best
            delta = max(delta, abs(v_old - V[s]))

        if delta < theta:
            # Converged — extract greedy policy and Q
            Q = _compute_Q(states, nS, nA, valid_actions, P, R, V, gamma,
                           terminal_mask)
            policy = _extract_policy(nS, valid_actions, Q, terminal_mask)
            return ValueIterationResult(policy=policy, V=V, Q=Q, iterations=it + 1)

    Q = _compute_Q(states, nS, nA, valid_actions, P, R, V, gamma, terminal_mask)
    policy = _extract_policy(nS, valid_actions, Q, terminal_mask)
    return ValueIterationResult(policy=policy, V=V, Q=Q, iterations=max_iter)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_Q(states, nS, nA, valid_actions, P, R, V, gamma,
               terminal_mask) -> np.ndarray:
    Q = np.zeros((nS, nA), dtype=float)
    for s in states:
        if terminal_mask is not None and terminal_mask[s]:
            continue
        for a in valid_actions[s]:
            q = 0.0
            for s_next, p in P[s][a].items():
                q += p * (R[s][a][s_next] + gamma * V[s_next])
            Q[s, a] = q
    return Q


def _extract_policy(nS, valid_actions, Q, terminal_mask) -> np.ndarray:
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        acts = valid_actions[s]
        if not acts:
            continue
        if terminal_mask is not None and terminal_mask[s]:
            policy[s] = acts[0]
            continue
        policy[s] = max(acts, key=lambda a: Q[s, a])
    return policy
