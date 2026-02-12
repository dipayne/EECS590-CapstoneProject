import argparse
import numpy as np
from src.envs.make_env import make_env

def run_random_policy(env_id: str, episodes: int = 5, max_steps: int = 500):
    env = make_env(env_id)
    returns = []

    for ep in range(episodes):
        obs, info = env.reset()
        total = 0.0
        for t in range(max_steps):
            a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
        print(f"Episode {ep+1}/{episodes}: return={total:.3f}, steps={t+1}")

    env.close()
    print("Avg return:", float(np.mean(returns)))
    return returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="highway-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    run_random_policy(args.env, args.episodes, args.max_steps)

if __name__ == "__main__":
    main()
