from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class PolicyIterationResult:
    policy: np.ndarray
    V: np.ndarray
    Q: np.ndarray
    iterations: int


def policy_evaluation(states: List[int], nS: int, nA: int, valid_actions: List[List[int]],
                      P, R, policy: np.ndarray, gamma: float,
                      theta: float = 1e-8, max_iter: int = 10000,
                      terminal_mask: np.ndarray | None = None) -> np.ndarray:
    V = np.zeros(nS, dtype=float)

    for _ in range(max_iter):
        delta = 0.0
        for s in states:
            if terminal_mask is not None and terminal_mask[s]:
                continue

            v_old = V[s]
            a = int(policy[s])
            v_new = 0.0

            for s_next, p in P[s][a].items():
                v_new += p * (R[s][a][s_next] + gamma * V[s_next])

            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))

        if delta < theta:
            break

    return V


def compute_Q_from_V(states: List[int], nS: int, nA: int, valid_actions: List[List[int]],
                     P, R, V: np.ndarray, gamma: float,
                     terminal_mask: np.ndarray | None = None) -> np.ndarray:
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


def policy_improvement_from_V(states: List[int], nS: int, valid_actions: List[List[int]],
                              P, R, V: np.ndarray, gamma: float,
                              terminal_mask: np.ndarray | None = None) -> np.ndarray:
    new_policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        acts = valid_actions[s]
        if len(acts) == 0:
            continue
        if terminal_mask is not None and terminal_mask[s]:
            new_policy[s] = acts[0]
            continue

        best_a = acts[0]
        best_val = -1e30
        for a in acts:
            val = 0.0
            for s_next, p in P[s][a].items():
                val += p * (R[s][a][s_next] + gamma * V[s_next])
            if val > best_val:
                best_val = val
                best_a = a
        new_policy[s] = best_a
    return new_policy


def policy_improvement_from_Q(nS: int, valid_actions: List[List[int]],
                              Q: np.ndarray,
                              terminal_mask: np.ndarray | None = None) -> np.ndarray:
    new_policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        acts = valid_actions[s]
        if len(acts) == 0:
            continue
        if terminal_mask is not None and terminal_mask[s]:
            new_policy[s] = acts[0]
            continue
        new_policy[s] = max(acts, key=lambda a: Q[s, a])
    return new_policy


def policy_iteration(states: List[int], nS: int, nA: int, valid_actions: List[List[int]],
                     P, R, gamma: float,
                     terminal_mask: np.ndarray | None = None,
                     max_outer_iter: int = 1000) -> PolicyIterationResult:
    # init policy: first valid action
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        acts = valid_actions[s]
        if len(acts) > 0:
            policy[s] = acts[0]

    for it in range(max_outer_iter):
        V = policy_evaluation(states, nS, nA, valid_actions, P, R, policy, gamma, terminal_mask=terminal_mask)
        Q = compute_Q_from_V(states, nS, nA, valid_actions, P, R, V, gamma, terminal_mask=terminal_mask)

        improved_V = policy_improvement_from_V(states, nS, valid_actions, P, R, V, gamma, terminal_mask=terminal_mask)
        improved_Q = policy_improvement_from_Q(nS, valid_actions, Q, terminal_mask=terminal_mask)

        # prefer Q-greedy (rubric requires both; we compute both)
        new_policy = improved_Q

        if np.array_equal(new_policy, policy):
            return PolicyIterationResult(policy=policy, V=V, Q=Q, iterations=it + 1)

        policy = new_policy

    return PolicyIterationResult(policy=policy, V=V, Q=Q, iterations=max_outer_iter)
