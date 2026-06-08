import numpy as np
from curriculum.base import StartSelector


class Uniform(StartSelector):

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.free_cells: list[tuple[int, int]] = []

    def set_grid(self, unwrapped_env) -> None:
        if self.free_cells:
            return  # already cached (grid is static for Empty/FourRooms)

        grid = unwrapped_env.grid
        for x in range(unwrapped_env.width):
            for y in range(unwrapped_env.height):
                cell = grid.get(x, y)
                if cell is None:
                    self.free_cells.append((x, y))

    def sample(self) -> tuple[int, int]:
        idx = self.rng.integers(0, len(self.free_cells))
        return self.free_cells[idx]
