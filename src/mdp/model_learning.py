import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import trange


def estimate_mdp(env, obs_to_state_fn, nS: int, nA: int,
                 episodes: int = 200, max_steps: int = 1000,
                 gamma: float = 0.95, desc: str = "Estimating MDP",
                 counts_path: str = None):
    """
    Estimate a tabular MDP model (P and expected R) from the foundation environment
    by sampling random trajectories.

    If counts_path is provided and the file exists, prior transition counts are
    loaded and updated incrementally (model-based belief updating). The updated
    counts are saved back to counts_path after sampling.

    Returns
    -------
    P : list[list[dict]]  -- P[s][a] -> {s_next: probability}
    R : list[list[dict]]  -- R[s][a] -> {s_next: expected_reward}
    """
    if counts_path and os.path.exists(counts_path):
        trans_counts, reward_sums = _load_counts(counts_path)
        print(f"[model_learning] Loaded prior belief counts from '{counts_path}'.")
    else:
        trans_counts = [[defaultdict(int) for _ in range(nA)] for _ in range(nS)]
        reward_sums = [[defaultdict(float) for _ in range(nA)] for _ in range(nS)]

    for _ in trange(episodes, desc=desc):
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

    if counts_path:
        _save_counts(trans_counts, reward_sums, counts_path)
        print(f"[model_learning] Saved updated belief counts to '{counts_path}'.")

    return _counts_to_model(trans_counts, reward_sums, nS, nA)


def save_model(P, R, path: str) -> None:
    """Serialize and save an estimated MDP model (P, R) to disk."""
    _ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump({"P": P, "R": R}, f)
    print(f"[model_learning] Model saved to '{path}'.")


def load_model(path: str):
    """Load a previously saved MDP model from disk.

    Returns
    -------
    P, R : same format as estimate_mdp output
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[model_learning] Model loaded from '{path}'.")
    return data["P"], data["R"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _counts_to_model(trans_counts, reward_sums, nS: int, nA: int):
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


def _save_counts(trans_counts, reward_sums, path: str) -> None:
    _ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump({"trans_counts": trans_counts, "reward_sums": reward_sums}, f)


def _load_counts(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["trans_counts"], data["reward_sums"]


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
