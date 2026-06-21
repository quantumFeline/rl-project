"""BFS shortest-path distances from a source cell on a MiniGrid grid."""

from collections import deque
import numpy as np


def bfs_distances(unwrapped_env, source: tuple[int, int]) -> dict[tuple[int, int], int]:
    """Return {(col, row): distance} for every free cell reachable from source.

    Distance is measured in grid steps (not agent actions: turning costs zero
    grid steps here, so this is the minimum number of forward moves needed).
    """
    grid = unwrapped_env.grid
    w, h = unwrapped_env.width, unwrapped_env.height

    dist = {}
    queue = deque()
    dist[source] = 0
    queue.append(source)

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in dist:
                cell = grid.get(nx, ny)
                # None = empty cell, or it could be the goal cell
                if cell is None or cell.type == "goal":
                    dist[(nx, ny)] = dist[(cx, cy)] + 1
                    queue.append((nx, ny))

    return dist
