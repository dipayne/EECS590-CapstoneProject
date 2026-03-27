"""Train all agents with CheckpointManager and ExperimentLogger wired in.

Produces
--------
checkpoints/
    <algo>/highway-v0/<run_id>/
        hparams.json          -- hyperparameters used
        online_net_best.pt    -- best DQN online network (by episode return)
        target_net_best.pt    -- corresponding target network
        actor_best.pt         -- best PPO / REINFORCE actor
        critic_best.pt        -- best PPO / REINFORCE / A2C critic
        optimizer_best.pt     -- optimizer state
        training_log.json     -- per-episode metrics

logs/
    <algo>_highway-v0_<timestamp>/
        hparams.json          -- hyperparameters
        episode_log.json      -- per-episode {episode, return, length}
        step_log.json         -- per-step {step, loss, ...}  (deep agents)
        returns.png           -- smoothed return curve
        loss.png              -- loss curve (deep agents)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.envs.highway_wrapper import make_highway_env
from src.checkpoints.manager import CheckpointManager
from src.utils.logger import ExperimentLogger

TASK          = "highway-v0"
CKPT_ROOT     = "checkpoints"
LOG_ROOT      = "logs"
GAMMA         = 0.99
MAX_EP_STEPS  = 60
N_CLASSICAL   = 25       # episodes for classical agents
N_DEEP        = 2_000    # env steps for deep agents (fast but meaningful)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discounted_return(rewards, gamma=GAMMA):
    G, ret = 0.0, 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G


def _sanitize(hparams: dict) -> dict:
    """Convert numpy scalars to Python natives for JSON serialisation."""
    out = {}
    for k, v in hparams.items():
        if hasattr(v, "item"):        # numpy scalar
            out[k] = v.item()
        elif isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)           # fallback
    return out


def init_managers(algo: str, hparams: dict):
    safe = _sanitize(hparams)
    ckpt = CheckpointManager(
        root=CKPT_ROOT, algorithm=algo, task=TASK,
        hparams=safe, save_every_steps=0,
    )
    log = ExperimentLogger(log_dir=LOG_ROOT, run_name=f"{algo}_highway-v0")
    log.log_hparams(safe)
    return ckpt, log


# ---------------------------------------------------------------------------
# Classical agents
# ---------------------------------------------------------------------------

def run_classical(algo_name, AgentClass, kwargs):
    print(f"\n[Classical] {algo_name}")
    env = make_highway_env()
    nS, nA = env.n_tabular_states, env.n_actions

    hparams = {k: (int(v) if hasattr(v, "item") else v)
               for k, v in kwargs.items()
               if not callable(v)}
    hparams.update({"nS": int(nS), "nA": int(nA), "n_episodes": N_CLASSICAL})

    algo_key = algo_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    ckpt, log = init_managers(algo_key, hparams)

    agent = AgentClass(**kwargs)
    t0 = time.time()

    # All classical agents support train() returning a list of episode returns
    result = agent.train(env, N_CLASSICAL, MAX_EP_STEPS, verbose_every=0)
    returns = result if isinstance(result, list) else result.get("returns", [])

    for ep, ep_return in enumerate(returns):
        log.log_episode(ep, {"return": float(ep_return)})
        ckpt.log_metrics({"episode": ep, "return": float(ep_return)})

    log.save()
    log.plot_returns(title=f"{algo_name} - {TASK}")

    env.close()
    elapsed = time.time() - t0
    avg = float(np.mean(returns[-10:])) if returns else 0.0
    print(f"  done: {len(returns)} ep  avg={avg:+.2f}  ({elapsed:.0f}s)")
    print(f"  logs  -> {log.run_dir}")
    print(f"  ckpts -> {ckpt.run_dir}")


# ---------------------------------------------------------------------------
# REINFORCE
# ---------------------------------------------------------------------------

def run_reinforce(obs_dim, n_actions):
    from src.agents.deep.reinforce import REINFORCEAgent
    algo = "reinforce"
    hparams = dict(obs_dim=obs_dim, n_actions=n_actions, gamma=GAMMA,
                   lr_actor=5e-4, use_baseline=True, n_episodes=N_CLASSICAL)
    ckpt, log = init_managers(algo, hparams)

    print(f"\n[Deep] REINFORCE")
    env   = make_highway_env(normalize_obs=True)
    agent = REINFORCEAgent(obs_dim=obs_dim, n_actions=n_actions,
                           gamma=GAMMA, lr_actor=5e-4, use_baseline=True)
    t0 = time.time()
    best = -float("inf")

    for ep in range(N_CLASSICAL):
        obs, _ = env.reset()
        episode = []
        for _ in range(MAX_EP_STEPS):
            a = agent.select_action(obs, training=True)
            obs2, r, te, tr, _ = env.step(a)
            episode.append((obs.copy(), a, r))
            obs = obs2
            if te or tr:
                break

        metrics = agent.update(episode)
        ep_return = discounted_return([r for _, _, r in episode])

        log.log_episode(ep, {"return": ep_return, "length": len(episode)})
        log.log_step(ep, {"pg_loss": metrics["pg_loss"],
                          "vf_loss": metrics["vf_loss"]})
        ckpt.log_metrics({"episode": ep, "return": ep_return,
                          "pg_loss": metrics["pg_loss"]})

        if ep_return > best:
            best = ep_return
            ckpt.save_best(agent, ep_return)

    log.save()
    log.plot_returns(title=f"REINFORCE - {TASK}")
    log.plot_losses(metric="pg_loss", title="REINFORCE PG Loss")
    env.close()

    elapsed = time.time() - t0
    avg = float(np.mean([r["return"] for r in log.episode_log[-10:]]))
    print(f"  done: {N_CLASSICAL} ep  avg={avg:+.2f}  best={best:+.2f}  ({elapsed:.0f}s)")
    print(f"  logs  -> {log.run_dir}")
    print(f"  ckpts -> {ckpt.run_dir}")


# ---------------------------------------------------------------------------
# A2C
# ---------------------------------------------------------------------------

def run_a2c(obs_dim, n_actions):
    from src.agents.deep.a2c import A2CAgent
    algo = "a2c"
    hparams = dict(obs_dim=obs_dim, n_actions=n_actions, n_steps=5,
                   gamma=GAMMA, lr=7e-4, total_steps=N_DEEP)
    ckpt, log = init_managers(algo, hparams)

    print(f"\n[Deep] A2C")
    env   = make_highway_env(normalize_obs=True)
    agent = A2CAgent(obs_dim=obs_dim, n_actions=n_actions,
                     n_steps=5, gamma=GAMMA, lr=7e-4)
    t0    = time.time()
    obs, _ = env.reset()
    ep_return, ep_len, ep_num = 0.0, 0, 0
    best  = -float("inf")

    step = 0
    while step < N_DEEP:
        r_obs, r_acts, r_rews, r_dones = [], [], [], []
        last_obs_snap, last_done_snap = obs.copy(), False
        for _ in range(5):
            if step >= N_DEEP:
                break
            a = agent.select_action(obs, training=True)
            obs2, r, te, tr, _ = env.step(a)
            r_obs.append(obs.flatten().astype("float32"))
            r_acts.append(a); r_rews.append(r); r_dones.append(float(te or tr))
            ep_return += r; ep_len += 1; step += 1
            last_obs_snap = obs2.copy(); last_done_snap = bool(te or tr)
            obs = obs2
            if te or tr:
                log.log_episode(ep_num, {"return": ep_return, "length": ep_len})
                ckpt.log_metrics({"episode": ep_num, "return": ep_return})
                if ep_return > best:
                    best = ep_return
                    ckpt.save_best(agent, ep_return)
                ep_num += 1
                ep_return, ep_len = 0.0, 0
                obs, _ = env.reset()

        if r_obs:
            rollout = {
                "obs":       np.array(r_obs),
                "actions":   np.array(r_acts),
                "rewards":   r_rews,
                "dones":     r_dones,
                "last_obs":  last_obs_snap.flatten().astype("float32"),
                "last_done": last_done_snap,
            }
            metrics = agent.update(rollout)
            log.log_step(step, {"pg_loss": metrics.get("pg_loss", 0.0),
                                 "vf_loss": metrics.get("vf_loss", 0.0)})

    log.save()
    log.plot_returns(title=f"A2C - {TASK}")
    env.close()

    elapsed = time.time() - t0
    recent = [r["return"] for r in log.episode_log[-10:]]
    avg = float(np.mean(recent)) if recent else 0.0
    print(f"  done: {ep_num} ep  avg={avg:+.2f}  best={best:+.2f}  ({elapsed:.0f}s)")
    print(f"  logs  -> {log.run_dir}")
    print(f"  ckpts -> {ckpt.run_dir}")


# ---------------------------------------------------------------------------
# DQN variants
# ---------------------------------------------------------------------------

def run_dqn(variant: str, use_per: bool, label: str):
    from src.agents.deep.dqn import DQNAgent
    env_tmp = make_highway_env(normalize_obs=True)
    obs_dim, n_actions = env_tmp.obs_dim, env_tmp.n_actions
    env_tmp.close()

    algo = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
    hparams = dict(variant=variant, use_per=use_per, gamma=GAMMA, lr=1e-4,
                   batch_size=32, buffer_capacity=800, target_update_freq=80,
                   epsilon_start=1.0, epsilon_end=0.1,
                   epsilon_decay_steps=1_000, train_start=80,
                   total_steps=N_DEEP)
    ckpt, log = init_managers(algo, hparams)

    print(f"\n[Deep] {label}")
    env   = make_highway_env(normalize_obs=True)
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions,
                     variant=variant, gamma=GAMMA, lr=1e-4,
                     batch_size=32, buffer_capacity=800,
                     target_update_freq=80,
                     epsilon_start=1.0, epsilon_end=0.1,
                     epsilon_decay_steps=1_000,
                     use_per=use_per, train_start=80)
    t0    = time.time()
    obs, _ = env.reset()
    ep_return, ep_len, ep_num = 0.0, 0, 0
    best  = -float("inf")

    for step in range(N_DEEP):
        a = agent.select_action(obs, training=True)
        obs2, r, te, tr, _ = env.step(a)
        agent.buffer.add(obs, a, r, obs2, te or tr)
        ep_return += r; ep_len += 1
        obs = obs2

        if len(agent.buffer) >= agent.train_start:
            metrics = agent.update()
            if metrics:
                log.log_step(step, {"loss": metrics.get("loss", 0.0),
                                    "epsilon": agent.epsilon})

        if te or tr or ep_len >= MAX_EP_STEPS:
            log.log_episode(ep_num, {"return": ep_return, "length": ep_len})
            ckpt.log_metrics({"episode": ep_num, "return": ep_return,
                              "epsilon": agent.epsilon})
            if ep_return > best:
                best = ep_return
                ckpt.save_best(agent, ep_return)
            ep_num += 1
            ep_return, ep_len = 0.0, 0
            obs, _ = env.reset()

    log.save()
    log.plot_returns(title=f"{label} - {TASK}")
    log.plot_losses(metric="loss", title=f"{label} Loss")
    env.close()

    elapsed = time.time() - t0
    recent = [r["return"] for r in log.episode_log[-10:]]
    avg = float(np.mean(recent)) if recent else 0.0
    print(f"  done: {ep_num} ep  avg={avg:+.2f}  best={best:+.2f}  ({elapsed:.0f}s)")
    print(f"  logs  -> {log.run_dir}")
    print(f"  ckpts -> {ckpt.run_dir}")


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def run_ppo(obs_dim, n_actions):
    from src.agents.deep.ppo import PPOAgent
    algo = "ppo"
    hparams = dict(obs_dim=obs_dim, n_actions=n_actions, n_steps=128,
                   n_epochs=3, batch_size=32, gamma=GAMMA, lam=0.95,
                   clip_eps=0.2, lr=3e-4, ent_coef=0.01,
                   normalize_obs=True, total_steps=N_DEEP)
    ckpt, log = init_managers(algo, hparams)

    print(f"\n[Deep] PPO")
    env   = make_highway_env(normalize_obs=True)
    agent = PPOAgent(obs_dim=obs_dim, n_actions=n_actions,
                     n_steps=128, n_epochs=3, batch_size=32,
                     gamma=GAMMA, lam=0.95, clip_eps=0.2,
                     lr=3e-4, ent_coef=0.01, normalize_obs=True)
    t0    = time.time()
    obs, _ = env.reset()
    ep_return, ep_len, ep_num = 0.0, 0, 0
    best  = -float("inf")
    step  = 0
    last_done = False

    while step < N_DEEP:
        for _ in range(agent.n_steps):
            if step >= N_DEEP:
                break
            a, val, lp = agent._get_action_value_logprob(obs)
            obs2, r, te, tr, _ = env.step(a)
            done = te or tr
            agent.rollout.add(agent._process_obs(obs), a, r, val, lp, done)
            ep_return += r; ep_len += 1; step += 1
            obs = obs2
            if done or ep_len >= MAX_EP_STEPS:
                log.log_episode(ep_num, {"return": ep_return, "length": ep_len})
                ckpt.log_metrics({"episode": ep_num, "return": ep_return})
                if ep_return > best:
                    best = ep_return
                    ckpt.save_best(agent, ep_return)
                ep_num += 1
                ep_return, ep_len = 0.0, 0
                obs, _ = env.reset()
                last_done = True
            else:
                last_done = False

        if agent.rollout._full:
            metrics = agent.update(obs, last_done)
            log.log_step(step, {"pg_loss": metrics["pg_loss"],
                                 "vf_loss": metrics["vf_loss"],
                                 "ent_loss": metrics["ent_loss"]})

    log.save()
    log.plot_returns(title=f"PPO - {TASK}")
    log.plot_losses(metric="pg_loss", title="PPO Policy Loss")
    env.close()

    elapsed = time.time() - t0
    recent = [r["return"] for r in log.episode_log[-10:]]
    avg = float(np.mean(recent)) if recent else 0.0
    print(f"  done: {ep_num} ep  avg={avg:+.2f}  best={best:+.2f}  ({elapsed:.0f}s)")
    print(f"  logs  -> {log.run_dir}")
    print(f"  ckpts -> {ckpt.run_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("  Training all agents with checkpointing + logging")
    print("=" * 60)

    # ---- env dims ----
    env_tmp = make_highway_env(normalize_obs=True)
    obs_dim, n_actions = env_tmp.obs_dim, env_tmp.n_actions
    env_tmp.close()

    env_tab = make_highway_env()
    nS, nA  = env_tab.n_tabular_states, env_tab.n_actions
    obs_fn  = env_tab.tabular_state
    common  = dict(nS=nS, nA=nA, obs_to_state=obs_fn, gamma=GAMMA,
                   epsilon=1.0, epsilon_decay=0.97, epsilon_min=0.1)
    env_tab.close()

    # ---- classical ----
    from src.agents.classical.monte_carlo import MonteCarloAgent
    from src.agents.classical.td import TDAgent
    from src.agents.classical.sarsa import SARSAAgent
    from src.agents.classical.q_learning import QLearningAgent
    from src.agents.classical.q_lambda import QLambdaAgent
    from src.agents.classical.linear_fa import LinearFAAgent

    run_classical("monte_carlo",  MonteCarloAgent, {**common})
    run_classical("td_lambda",    TDAgent,
                  {**common, "alpha": 0.1, "lam": 0.9, "mode": "tdlam_backward"})
    run_classical("sarsa_lambda", SARSAAgent,
                  {**common, "alpha": 0.1, "lam": 0.9, "mode": "sarsa_lam_backward"})
    run_classical("q_learning",   QLearningAgent, {**common, "alpha": 0.1})
    run_classical("q_lambda",     QLambdaAgent,
                  {**common, "alpha": 0.1, "lam": 0.8})
    run_classical("linear_fa",    LinearFAAgent,
                  {**common, "alpha": 0.02, "method": "sarsa"})

    # ---- deep ----
    run_reinforce(obs_dim, n_actions)
    run_a2c(obs_dim, n_actions)
    run_dqn("dqn",            False, "DQN")
    run_dqn("double",         False, "Double_DQN")
    run_dqn("dueling",        False, "Dueling_DQN")
    run_dqn("double_dueling", True,  "DQN_D+D+PER")
    run_ppo(obs_dim, n_actions)

    # ---- summary ----
    print("\n" + "=" * 60)
    print("  Done. Directory layout:")
    print("=" * 60)
    for d in ["checkpoints", "logs"]:
        for root, dirs, files in os.walk(d):
            depth = root.replace(d, "").count(os.sep)
            indent = "  " * depth
            print(f"{indent}{os.path.basename(root)}/")
            if depth >= 2:
                for f in sorted(files):
                    print(f"{indent}  {f}")


if __name__ == "__main__":
    main()
