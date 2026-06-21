import numpy as np
from curriculum.base import StartSelector


class TDError(StartSelector):
    """Start where the agent's predictions are most wrong (highest TD error)."""

    def __init__(self, seed: int = 0, alpha: float = 0.1, epsilon: float = 0.01):
        self.rng = np.random.default_rng(seed)
        # Exponential moving average smoothing factor
        self.alpha = alpha
        # Minimum probability floor so every cell has some chance of being chosen
        self.epsilon = epsilon

        self.free_cells: list[tuple[int, int]] = []
        self.cell_to_idx: dict[tuple[int, int], int] = {}
        # Per-cell EMA of absolute TD error
        self.td_scores: np.ndarray | None = None

    def set_grid(self, unwrapped_env) -> None:
        if self.free_cells:
            return

        grid = unwrapped_env.grid
        for x in range(unwrapped_env.width):
            for y in range(unwrapped_env.height):
                cell = grid.get(x, y)
                if cell is None:
                    self.cell_to_idx[(x, y)] = len(self.free_cells)
                    self.free_cells.append((x, y))

        self.td_scores = np.ones(len(self.free_cells), dtype=np.float64)

    def sample(self) -> tuple[int, int]:
        # Probability proportional to TD score, with an epsilon floor
        scores = self.td_scores + self.epsilon
        probs = scores / scores.sum()

        idx = self.rng.choice(len(self.free_cells), p=probs)
        return self.free_cells[idx]

    def update(self, *, obs_batch: np.ndarray, td_errors: np.ndarray,
               grid_w: int, grid_h: int, **kwargs) -> None:
        """Update per-cell TD error estimates.

        Args:
            obs_batch: (batch_size, obs_dim) normalized observations from the batch.
            td_errors: (batch_size,) absolute TD error per sample.
            grid_w: grid width, for denormalizing obs back to cell coordinates.
            grid_h: grid height, for denormalizing obs back to cell coordinates.
        """
        for i in range(len(td_errors)):
            col = int(round(obs_batch[i, 0] * (grid_w - 1)))
            row = int(round(obs_batch[i, 1] * (grid_h - 1)))
            cell = (col, row)
            if cell in self.cell_to_idx:
                idx = self.cell_to_idx[cell]
                self.td_scores[idx] = (
                    (1 - self.alpha) * self.td_scores[idx]
                    + self.alpha * td_errors[i]
                )
