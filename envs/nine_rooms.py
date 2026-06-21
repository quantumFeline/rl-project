"""3x3 grid of rooms with single-cell doorways between adjacent rooms.

Layout (room_size=7, grid 25x25):

    +---------+---------+---------+
    |         |         |         |
    |  (0,0)  d  (1,0)  d  (2,0)  |
    |         |         |         |
    +----d----+----d----+----d----+
    |         |         |         |
    |  (0,1)  d  (1,1)  d  (2,1)  |
    |         |         |         |
    +----d----+----d----+----d----+
    |         |         |         |
    |  (0,2)  d  (1,2)  d  (2,2)  |
    |         |         |         |
    +---------+---------+---------+

'd' marks doorways (single empty cell in an otherwise solid wall).
"""

from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class NineRoomsEnv(MiniGridEnv):

    def __init__(self, room_size: int = 7, max_steps: int = 400, **kwargs):
        self.room_size = room_size
        size = 3 * (room_size + 1) + 1
        mission_space = MissionSpace(mission_func=lambda: "reach the goal")
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        s = self.room_size
        self.grid = Grid(width, height)

        # Outer walls
        self.grid.wall_rect(0, 0, width, height)

        # Vertical interior walls (x = s+1, x = 2*(s+1))
        for i in range(1, 3):
            x = i * (s + 1)
            for y in range(1, height - 1):
                self.grid.set(x, y, Wall())

        # Horizontal interior walls (y = s+1, y = 2*(s+1))
        for j in range(1, 3):
            y = j * (s + 1)
            for x in range(1, width - 1):
                self.grid.set(x, y, Wall())

        # Doorways in vertical walls: one per room pair
        for i in range(1, 3):
            x = i * (s + 1)
            for j in range(3):
                door_y = j * (s + 1) + (s + 1) // 2
                self.grid.set(x, door_y, None)

        # Doorways in horizontal walls: one per room pair
        for j in range(1, 3):
            y = j * (s + 1)
            for i in range(3):
                door_x = i * (s + 1) + (s + 1) // 2
                self.grid.set(door_x, y, None)

        # Goal in the bottom-right room
        self.put_obj(Goal(), width - 2, height - 2)

        # Agent in the top-left room
        self.agent_pos = (1, 1)
        self.agent_dir = 0
