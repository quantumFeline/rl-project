import numpy as np
from curriculum.base import StartSelector
from curriculum.bfs import bfs_distances


class Reverse(StartSelector):
    """Reverse curriculum: start near goal, expand outward as success rate rises."""

    def __init__(
        self,
        goal: tuple[int, int],
        seed: int = 0,
        initial_band: int = 3,
        expand_threshold: float = 0.7,
        window: int = 20,
    ):
        self.goal = goal
        self.rng = np.random.default_rng(seed)

        # Distance band: agent starts within [0, max_dist] of the goal.
        # Expands by 1 when recent success rate exceeds expand_threshold.
        self.max_dist = initial_band
        self.expand_threshold = expand_threshold

        # Only frontier episodes count towards expansion
        self.window = window
        self.frontier_successes: list[bool] = []

        # Distance of the most recent episode's start cell
        self._last_start_dist: int = 0

        # Populated on first set_grid call
        self.dist_map: dict[tuple[int, int], int] = {}
        self.max_possible_dist = 0
        self.cells_by_dist: dict[int, list[tuple[int, int]]] = {}

    def set_grid(self, unwrapped_env) -> None:
        if self.dist_map:
            return

        self.dist_map = bfs_distances(unwrapped_env, self.goal)
        self.max_possible_dist = max(self.dist_map.values())

        # Group free cells (excluding goal) by distance
        for cell, d in self.dist_map.items():
            if cell == self.goal:
                continue
            self.cells_by_dist.setdefault(d, []).append(cell)

    def sample(self) -> tuple[int, int]:
        # Collect all cells within the current distance band
        candidates = []
        for d in range(1, self.max_dist + 1):
            candidates.extend(self.cells_by_dist.get(d, []))

        if not candidates:
            candidates = self.cells_by_dist.get(1, [])

        idx = self.rng.integers(0, len(candidates))
        cell = candidates[idx]
        self._last_start_dist = self.dist_map[cell]
        return cell

    def update(self, *, success: bool | None = None, **kwargs) -> None:
        if success is None:
            return

        # Only count episodes that started on the frontier ring
        if self._last_start_dist < self.max_dist:
            return

        self.frontier_successes.append(success)
        if len(self.frontier_successes) > self.window:
            self.frontier_successes.pop(0)

        if len(self.frontier_successes) >= self.window:
            rate = sum(self.frontier_successes) / len(self.frontier_successes)
            if rate >= self.expand_threshold and self.max_dist < self.max_possible_dist:
                self.max_dist += 1
                self.frontier_successes.clear()
