import numpy as np
from collections import defaultdict
from tqdm import trange


def estimate_mdp(env, obs_to_state_fn, nS: int, nA: int,
                 episodes: int = 200, max_steps: int = 1000, gamma: float = 0.95, desc: str = "Estimating MDP"):
    """
    Estimate a tabular MDP model (P and expected R) from the foundation environment by sampling.
    Returns:
      P[s][a] -> dict(s_next -> prob)
      R[s][a] -> dict(s_next -> expected_reward)

    tqdm progress bar tracks episodes.
    """
    trans_counts = [[defaultdict(int) for _ in range(nA)] for _ in range(nS)]
    reward_sums = [[defaultdict(float) for _ in range(nA)] for _ in range(nS)]

    for ep in trange(episodes, desc=desc):
        obs, info = env.reset()
        s = obs_to_state_fn(obs)

        for t in range(max_steps):
            a = env.action_space.sample()
            obs2, r, terminated, truncated, info = env.step(a)
            s2 = obs_to_state_fn(obs2)

            trans_counts[s][a][s2] += 1
            reward_sums[s][a][s2] += float(r)

            s = s2
            if terminated or truncated:
                break

    P = [[{} for _ in range(nA)] for _ in range(nS)]
    R = [[{} for _ in range(nA)] for _ in range(nS)]

    for s in range(nS):
        for a in range(nA):
            total = sum(trans_counts[s][a].values())
            if total == 0:
                continue
            for s2, c in trans_counts[s][a].items():
                P[s][a][s2] = c / total
                R[s][a][s2] = reward_sums[s][a][s2] / c

    return P, R
