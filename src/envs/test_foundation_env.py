import gymnasium as gym
import highway_env  # registers envs

def main():
    env = gym.make("highway-v0")
    obs, info = env.reset()
    print("reset ok, obs type:", type(obs), "obs shape:", getattr(obs, "shape", None))

    for t in range(10):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        print(t, "reward:", reward, "term:", terminated, "trunc:", truncated)
        if terminated or truncated:
            env.reset()

    env.close()
    print("done")

if __name__ == "__main__":
    main()
