"""Open grid with a spiral labyrinth around the goal.

BFS distance through the spiral is short, but the narrow corridors with
multiple turns make those cells much harder to learn than open-space cells
at the same or greater BFS distance. Designed to test whether TDError
can outperform Reverse when distance is a poor proxy for difficulty.

Layout (19x19):
- Large open area covering most of the grid
- 2-ring spiral in the bottom-right quadrant surrounds the goal
- Outer ring opens towards the open area (left side)
- Inner ring opens on a different side (bottom)
- 3-cell-wide corridors between rings
"""

from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class DeceptiveRoomsEnv(MiniGridEnv):

    def __init__(self, max_steps: int = 300, **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "reach the goal")
        super().__init__(
            mission_space=mission_space,
            grid_size=19,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Goal at the centre of the spiral (needs radius 6 clearance to walls)
        gx, gy = 11, 11
        self.put_obj(Goal(), gx, gy)

        # Ring 1 (inner): radius 3, opening on the bottom
        # Ring 2 (outer): radius 6, opening on the left (towards open area)
        rings = [
            (3, 2),  # radius 3, open bottom
            (6, 3),  # radius 6, open left
        ]

        # Place all walls
        for radius, _ in rings:
            x0, y0 = gx - radius, gy - radius
            x1, y1 = gx + radius, gy + radius
            for x in range(x0, x1 + 1):
                self.grid.set(x, y0, Wall())
                self.grid.set(x, y1, Wall())
            for y in range(y0, y1 + 1):
                self.grid.set(x0, y, Wall())
                self.grid.set(x1, y, Wall())

        # Carve openings
        for radius, open_side in rings:
            x0, y0 = gx - radius, gy - radius
            x1, y1 = gx + radius, gy + radius
            if open_side == 0:
                self.grid.set(gx, y0, None)
            elif open_side == 1:
                self.grid.set(x1, gy, None)
            elif open_side == 2:
                self.grid.set(gx, y1, None)
            elif open_side == 3:
                self.grid.set(x0, gy, None)

        self.agent_pos = (1, 1)
        self.agent_dir = 0
