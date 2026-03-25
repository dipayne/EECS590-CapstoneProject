"""Dynamic replay-buffer replacement script.

Enforces disk-usage limits by removing the oldest replay-buffer runs for a
given (algorithm, task) pair.  Can be run manually or as a cron / post-training
hook.

Usage
-----
    # Enforce 800 MB limit for dqn/highway-v0
    python scripts/replace_replay_buffer.py \
        --algo dqn --task highway-v0 --max-total-mb 800

    # Enforce 200 MB limit for every algorithm / task combination
    python scripts/replace_replay_buffer.py --all --max-total-mb 200

    # List current runs without deleting anything
    python scripts/replace_replay_buffer.py \
        --algo dqn --task highway-v0 --list-only
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow importing from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.replay_buffer.manager import BufferManager


def parse_args():
    p = argparse.ArgumentParser(
        description="Dynamic replay-buffer size management"
    )
    p.add_argument("--root", default="data/replay_buffers",
                   help="Replay buffer root directory")
    p.add_argument("--algo", default="dqn",
                   help="Algorithm name (dqn, ppo, q_learning, sarsa, ...)")
    p.add_argument("--task", default="highway-v0",
                   help="Task / environment ID")
    p.add_argument("--max-total-mb", type=float, default=800.0,
                   help="Maximum aggregate storage in MB for this (algo, task)")
    p.add_argument("--max-run-mb",   type=float, default=200.0,
                   help="Per-run size cap (warning only)")
    p.add_argument("--all",         action="store_true",
                   help="Apply to every (algorithm, task) combination found under --root")
    p.add_argument("--list-only",   action="store_true",
                   help="Print run info without deleting anything")
    p.add_argument("--dry-run",     action="store_true",
                   help="Show what would be deleted without deleting")
    return p.parse_args()


def process_pair(root, algo, task, max_run_mb, max_total_mb, list_only, dry_run):
    mgr = BufferManager(root, algo, task,
                        max_run_mb=max_run_mb, max_total_mb=max_total_mb)
    runs = mgr.list_runs()
    total_mb = mgr.total_size_mb()

    print(f"\n[{algo}/{task}]  runs={len(runs)}  total={total_mb:.1f} MB  "
          f"limit={max_total_mb:.1f} MB")

    for r in runs:
        size_mb = r.get("size_bytes", 0) / (1024 ** 2)
        ep      = r.get("episodes", "?")
        print(f"  run={r['run_id']}  size={size_mb:.1f} MB  episodes={ep}")

    if list_only:
        return

    if total_mb <= max_total_mb:
        print("  -> Under limit.  Nothing removed.")
        return

    if dry_run:
        # Simulate without deleting
        print(f"  [DRY RUN] Would remove oldest run(s) to get under {max_total_mb:.1f} MB")
        return

    deleted = mgr.enforce_size_limit()
    if deleted:
        print(f"  -> Removed {len(deleted)} run(s): {deleted}")
    else:
        print("  -> Could not free enough space (no runs to remove?).")


def main():
    args = parse_args()

    if args.all:
        # Walk the buffer root directory for all (algo, task) pairs
        root = args.root
        if not os.path.isdir(root):
            print(f"Buffer root not found: {root}")
            return
        for algo in os.listdir(root):
            algo_dir = os.path.join(root, algo)
            if not os.path.isdir(algo_dir):
                continue
            for task in os.listdir(algo_dir):
                if not os.path.isdir(os.path.join(algo_dir, task)):
                    continue
                process_pair(root, algo, task, args.max_run_mb,
                             args.max_total_mb, args.list_only, args.dry_run)
    else:
        process_pair(args.root, args.algo, args.task,
                     args.max_run_mb, args.max_total_mb,
                     args.list_only, args.dry_run)


if __name__ == "__main__":
    main()
