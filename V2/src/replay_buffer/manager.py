"""Replay-buffer disk management.

Responsibilities
----------------
1. Organise replay-buffer files under a structured directory tree:
       data/replay_buffers/<algorithm>/<task>/<run_id>/
2. Assert maximum storage size per run and across all runs for an
   algorithm+task pair (preventing unbounded disk growth).
3. Implement a dynamic *replacement* policy: when the aggregate size
   exceeds ``max_total_mb``, delete the oldest run(s) first.
4. Track per-run metadata (timestamp, hyperparams, episode count, size)
   in ``metadata.json`` inside each run directory.

Usage (called from training scripts)
-------------------------------------
    mgr = BufferManager("data/replay_buffers", algorithm="dqn", task="highway-v0")
    run_dir = mgr.new_run(hparams={"lr": 1e-3, "gamma": 0.99})
    ...
    mgr.save_buffer(buf, run_dir)
    mgr.enforce_size_limit()
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_KB = 1024
_MB = _KB * _KB


class BufferManager:
    """Manages persisted replay buffers for one (algorithm, task) pair.

    Parameters
    ----------
    root:
        Top-level data directory (e.g. ``"data/replay_buffers"``).
    algorithm:
        Agent name, e.g. ``"dqn"``, ``"ppo"``, ``"q_learning"``.
    task:
        Environment ID, e.g. ``"highway-v0"``.
    max_run_mb:
        Per-run size limit in megabytes.  If a single run exceeds this,
        ``save_buffer()`` will warn but not raise.
    max_total_mb:
        Aggregate limit across all runs for this (algorithm, task) pair.
        Oldest runs are deleted to stay under this limit.
    """

    def __init__(
        self,
        root: str,
        algorithm: str,
        task: str,
        max_run_mb: float = 200.0,
        max_total_mb: float = 1000.0,
    ):
        self.root = Path(root)
        self.algorithm = algorithm
        self.task = task
        self.max_run_bytes = max_run_mb * _MB
        self.max_total_bytes = max_total_mb * _MB

        self.base_dir = self.root / algorithm / task
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def new_run(self, hparams: Optional[Dict[str, Any]] = None) -> Path:
        """Create a new timestamped run directory and write metadata.

        Returns
        -------
        Path
            Absolute path to the newly created run directory.
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "run_id": run_id,
            "algorithm": self.algorithm,
            "task": self.task,
            "created_at": run_id,
            "hparams": hparams or {},
            "episodes": 0,
            "size_bytes": 0,
        }
        self._write_metadata(run_dir, metadata)
        return run_dir

    def save_buffer(self, buf, run_dir: Path, tag: str = "buffer") -> Path:
        """Save *buf* (a ReplayBuffer / PER) to *run_dir/<tag>.npz*.

        Updates the run's metadata with final episode count and file size.

        Parameters
        ----------
        buf:
            Any buffer object with a ``.save(path)`` method (no extension).
        run_dir:
            Path returned by ``new_run()``.
        tag:
            Filename stem (default ``"buffer"``).
        """
        out_path = run_dir / tag
        buf.save(str(out_path))
        npz_path = Path(str(out_path) + ".npz")
        size_bytes = npz_path.stat().st_size if npz_path.exists() else 0

        if size_bytes > self.max_run_bytes:
            print(
                f"[BufferManager] WARNING: run buffer {size_bytes / _MB:.1f} MB "
                f"> limit {self.max_run_bytes / _MB:.1f} MB"
            )

        # Update metadata
        md = self._read_metadata(run_dir)
        md["size_bytes"] = size_bytes
        md["episodes"] = getattr(buf, "_size", 0)
        self._write_metadata(run_dir, md)

        return npz_path

    # ------------------------------------------------------------------
    # Size enforcement  (dynamic replacement of oldest runs)
    # ------------------------------------------------------------------

    def enforce_size_limit(self) -> List[str]:
        """Delete oldest run(s) until total size is under ``max_total_mb``.

        Returns list of deleted run IDs.
        """
        deleted = []
        while True:
            total = self._total_size_bytes()
            if total <= self.max_total_bytes:
                break
            oldest = self._oldest_run()
            if oldest is None:
                break
            print(
                f"[BufferManager] Removing old run {oldest.name} "
                f"(total={total / _MB:.1f} MB > limit={self.max_total_bytes / _MB:.1f} MB)"
            )
            shutil.rmtree(oldest)
            deleted.append(oldest.name)

        return deleted

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def list_runs(self) -> List[Dict]:
        """Return metadata dicts for all runs, sorted newest first."""
        runs = []
        for run_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if run_dir.is_dir():
                md = self._read_metadata(run_dir)
                runs.append(md)
        return runs

    def total_size_mb(self) -> float:
        return self._total_size_bytes() / _MB

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_metadata(self, run_dir: Path, metadata: dict) -> None:
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _read_metadata(self, run_dir: Path) -> dict:
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {"run_id": run_dir.name, "size_bytes": 0}

    def _total_size_bytes(self) -> int:
        total = 0
        for run_dir in self.base_dir.iterdir():
            if run_dir.is_dir():
                for f in run_dir.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return total

    def _oldest_run(self) -> Optional[Path]:
        """Return path of the oldest run directory (by name / timestamp)."""
        runs = sorted(
            [d for d in self.base_dir.iterdir() if d.is_dir()]
        )
        return runs[0] if runs else None
