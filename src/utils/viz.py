from __future__ import annotations
import os
from typing import List, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt


ARROWS = {
    0: "↑",  # UP
    1: "→",  # RIGHT
    2: "↓",  # DOWN
    3: "←",  # LEFT
    4: "•",  # WAIT
}


def policy_to_grid(policy: np.ndarray, rows: int, cols: int,
                   blocked_states: Set[int], terminal_states: Set[int]) -> List[List[str]]:
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            s = r * cols + c
            if s in blocked_states:
                row.append("█")
            elif s in terminal_states:
                row.append("G")
            else:
                a = int(policy[s])
                row.append(ARROWS.get(a, "?"))
        grid.append(row)
    return grid


def print_policy_grid(grid: List[List[str]], depot_rc: Tuple[int, int], goal_rc: Tuple[int, int]) -> None:
    dr, dc = depot_rc
    gr, gc = goal_rc
    for r, row in enumerate(grid):
        out = []
        for c, ch in enumerate(row):
            if (r, c) == (dr, dc):
                out.append("D")
            elif (r, c) == (gr, gc):
                out.append("G")
            else:
                out.append(ch)
        print(" ".join(out))


def save_policy_map_png(grid: List[List[str]], depot_rc: Tuple[int, int], goal_rc: Tuple[int, int],
                        out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True)

    for r in range(rows):
        for c in range(cols):
            ch = grid[r][c]
            if (r, c) == depot_rc:
                text = "D"
            elif (r, c) == goal_rc:
                text = "G"
            else:
                text = ch
            ax.text(c, r, text, ha="center", va="center", fontsize=14)

    ax.set_title("Policy Map (D=Depot, G=Goal, █=Blocked)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_value_heatmap_png(V: np.ndarray, rows: int, cols: int, blocked_states: Set[int],
                           out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # reshape and mask blocked cells
    grid = V.reshape(rows, cols).copy()
    for s in blocked_states:
        r, c = divmod(s, cols)
        grid[r, c] = np.nan

    fig, ax = plt.subplots()
    im = ax.imshow(grid)
    ax.set_title("Value Function Heatmap (NaN=Blocked)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
