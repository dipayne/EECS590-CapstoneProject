"""Checkpoint management for deep RL agents.

Directory layout
----------------
    checkpoints/
    └── <algorithm>/
        └── <task>/
            └── <run_id>/
                ├── hparams.json          # hyperparameters for this run
                ├── actor_best.pt         # best actor checkpoint (PPO)
                ├── critic_best.pt        # best critic checkpoint (PPO)
                ├── online_net_best.pt    # best online Q-net (DQN)
                ├── target_net_best.pt    # corresponding target net (DQN)
                ├── checkpoint_step_*.pt  # periodic checkpoints
                └── training_log.json     # episode returns / losses per update

Supports
--------
- Saving best checkpoint by episode return
- Periodic checkpoint saving every N steps
- Restoring agent from best checkpoint
- Listing all checkpoints for a run
- Hyperparameter versioning (separate sub-dirs per hparam config)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class CheckpointManager:
    """Manages saving / loading checkpoints for one agent run.

    Parameters
    ----------
    root:
        Top-level checkpoints directory (e.g. ``"checkpoints"``).
    algorithm:
        Agent identifier, e.g. ``"dqn"``, ``"ppo"``, ``"sarsa"``.
    task:
        Environment ID, e.g. ``"highway-v0"``.
    hparams:
        Hyperparameter dict persisted to ``hparams.json``.
    save_every_steps:
        If > 0, save a periodic checkpoint every N gradient steps.
    """

    def __init__(
        self,
        root: str,
        algorithm: str,
        task: str,
        hparams: Optional[Dict[str, Any]] = None,
        save_every_steps: int = 0,
    ):
        self.root = Path(root)
        self.algorithm = algorithm
        self.task = task
        self.hparams = hparams or {}
        self.save_every_steps = save_every_steps

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.root / algorithm / task / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Persist hyperparameters immediately
        with open(self.run_dir / "hparams.json", "w") as f:
            json.dump({"run_id": run_id, **self.hparams}, f, indent=2)

        self._best_return = -float("inf")
        self._log: List[dict] = []

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_best(self, agent, episode_return: float) -> bool:
        """Save checkpoint if *episode_return* beats the current best.

        Returns True if a new best was saved.
        """
        if episode_return <= self._best_return:
            return False

        self._best_return = episode_return
        self._save_agent(agent, tag="best")
        print(
            f"  [Ckpt] New best return={episode_return:.3f}  "
            f"saved to {self.run_dir}"
        )
        return True

    def save_periodic(self, agent, step: int) -> None:
        """Save a periodic checkpoint at gradient step *step*."""
        if self.save_every_steps <= 0:
            return
        if step % self.save_every_steps == 0:
            self._save_agent(agent, tag=f"step_{step:08d}")

    def log_metrics(self, metrics: dict) -> None:
        """Append a metrics dict to the training log."""
        self._log.append(metrics)
        with open(self.run_dir / "training_log.json", "w") as f:
            json.dump(self._log, f, indent=2)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_best(self, agent) -> bool:
        """Load best checkpoint into *agent*.  Returns True if found."""
        return self._load_agent(agent, tag="best")

    def load_step(self, agent, step: int) -> bool:
        """Load periodic checkpoint at *step*."""
        return self._load_agent(agent, tag=f"step_{step:08d}")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[str]:
        """Return sorted list of checkpoint file names in run_dir."""
        return sorted(
            f.name for f in self.run_dir.iterdir()
            if f.suffix == ".pt"
        )

    def best_return(self) -> float:
        return self._best_return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_agent(self, agent, tag: str) -> None:
        """Dispatch to agent.save() for each network component."""
        import torch
        # Agents expose .save(path) but some (PPO/REINFORCE) have actor/critic separately
        if hasattr(agent, "actor") and hasattr(agent, "critic"):
            # PPO / REINFORCE style: actor + critic networks
            torch.save(agent.actor.state_dict(),
                       self.run_dir / f"actor_{tag}.pt")
            torch.save(agent.critic.state_dict(),
                       self.run_dir / f"critic_{tag}.pt")
            # Optimizer: PPO uses .optimizer, REINFORCE uses .opt_actor
            if hasattr(agent, "optimizer"):
                torch.save(agent.optimizer.state_dict(),
                           self.run_dir / f"optimizer_{tag}.pt")
            elif hasattr(agent, "opt_actor"):
                torch.save(agent.opt_actor.state_dict(),
                           self.run_dir / f"opt_actor_{tag}.pt")
                if hasattr(agent, "opt_critic") and agent.opt_critic is not None:
                    torch.save(agent.opt_critic.state_dict(),
                               self.run_dir / f"opt_critic_{tag}.pt")
        elif hasattr(agent, "online_net") and hasattr(agent, "target_net"):
            # DQN-style: online + target nets
            import torch
            torch.save(agent.online_net.state_dict(),
                       self.run_dir / f"online_net_{tag}.pt")
            torch.save(agent.target_net.state_dict(),
                       self.run_dir / f"target_net_{tag}.pt")
            torch.save(agent.optimizer.state_dict(),
                       self.run_dir / f"optimizer_{tag}.pt")
        else:
            # Generic: use agent's own .save()
            agent.save(str(self.run_dir / f"agent_{tag}"))

    def _load_agent(self, agent, tag: str) -> bool:
        import torch
        if hasattr(agent, "actor") and hasattr(agent, "critic"):
            actor_path = self.run_dir / f"actor_{tag}.pt"
            critic_path = self.run_dir / f"critic_{tag}.pt"
            if not actor_path.exists():
                return False
            agent.actor.load_state_dict(
                torch.load(actor_path, map_location=agent.device))
            agent.critic.load_state_dict(
                torch.load(critic_path, map_location=agent.device))
            return True
        elif hasattr(agent, "online_net") and hasattr(agent, "target_net"):
            online_path = self.run_dir / f"online_net_{tag}.pt"
            if not online_path.exists():
                return False
            agent.online_net.load_state_dict(
                torch.load(online_path, map_location=agent.device))
            agent.target_net.load_state_dict(
                torch.load(self.run_dir / f"target_net_{tag}.pt",
                           map_location=agent.device))
            return True
        else:
            ckpt_path = str(self.run_dir / f"agent_{tag}")
            if not Path(ckpt_path).exists():
                return False
            agent.load(ckpt_path)
            return True
