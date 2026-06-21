"""Render the three environments as PNG grid diagrams for the report."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

import minigrid  # registers MiniGrid-* envs
import envs  # registers NineRooms-v0, DeceptiveRooms-v0
import gymnasium as gym


def grid_to_array(env):
    """Extract a 2D array from a MiniGrid env: 0=floor, 1=wall, 2=goal, 3=agent."""
    grid = env.unwrapped.grid
    w, h = grid.width, grid.height
    arr = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            cell = grid.get(x, y)
            if cell is None:
                arr[y, x] = 0
            elif cell.type == "wall":
                arr[y, x] = 1
            elif cell.type == "goal":
                arr[y, x] = 2
    ax, ay = env.unwrapped.agent_pos
    arr[ay, ax] = 3
    return arr


def render_grid(arr, title, ax):
    cmap = ListedColormap(["white", "#333333", "#4CAF50", "#2196F3"])
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=3, origin="upper")

    h, w = arr.shape
    for x in range(w + 1):
        ax.axvline(x - 0.5, color="#999999", linewidth=0.5)
    for y in range(h + 1):
        ax.axhline(y - 0.5, color="#999999", linewidth=0.5)

    # Mark goal and agent with text
    gy, gx = np.argwhere(arr == 2)[0]
    ax.text(gx, gy, "G", ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ay, ax_ = np.argwhere(arr == 3)[0]
    ax.text(ax_, ay, "A", ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


envs_to_render = [
    ("MiniGrid-FourRooms-v0", "FourRooms (19x19)", {"max_steps": 200}),
    ("NineRooms-v0", "NineRooms (25x25)", {"max_steps": 400}),
    ("DeceptiveRooms-v0", "DeceptiveRooms (19x19)", {"max_steps": 300}),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for (env_id, title, kwargs), ax in zip(envs_to_render, axes):
    env = gym.make(env_id, **kwargs)
    env.reset(seed=0)
    arr = grid_to_array(env)
    render_grid(arr, title, ax)
    env.close()

plt.tight_layout()
plt.savefig("report/fig_environments.png", dpi=200, bbox_inches="tight")
print("Saved report/fig_environments.png")
plt.close()
