"""
Maze data container, DFS generator, and gymnasium environment.

Grid convention: 0 = free cell, 1 = wall.

The DFS generator produces a "perfect maze": every pair of cells is connected
by exactly one path, corridors are one cell wide, no loops.
Grid size must be odd. Borders are always walls and are never carved.

Goal is always the bottom-right room cell (size-2, size-2).
Start is not fixed (it is passed to MazeEnv.reset() each episode)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Maze:
    """
    Static maze layout. Holds the wall grid, the list of free cells, and the goal.
    Does not hold episode state (agent position, step count); that lives in MazeEnv.
    """

    def __init__(self, grid: np.ndarray, goal: tuple = None):
        """
        :param grid: 2D int8 array of shape (n_rows, n_cols), 0=free, 1=wall.
        :param goal: (r, c) of the goal cell. Defaults to bottom-right room cell
                     (n_rows-2, n_cols-2). Must be a free cell.
        """
        assert grid.ndim == 2, "Grid must be 2D"
        self.grid = grid
        self.n_rows, self.n_cols = grid.shape

        # Free cells in row-major order: list of (r, c) tuples.
        # Used to build one-hot observations and to sample random starts.
        rows, cols = np.where(grid == 0)
        self.free_cells = list(zip(rows.tolist(), cols.tolist()))
        self.n_free = len(self.free_cells)

        # Reverse lookup: (r, c) -> index in free_cells.
        # Needed to convert a position to a one-hot vector in O(1).
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.free_cells)}

        self.goal = tuple(goal) if goal is not None else (self.n_rows - 2, self.n_cols - 2)
        assert self.grid[self.goal] == 0, f"Goal cell {self.goal} is a wall"


class MazeGenerator:
    """
    Generates perfect mazes using iterative DFS (depth-first search).

    The algorithm:
      1. Start with a grid of all walls.
      2. Pick the goal cell (bottom-right room). Mark it free. Push to stack.
      3. From the current cell, look at room neighbors two steps away.
         If any are still walls (unvisited), pick one at random:
           - Mark it free (carve the room).
           - Mark the cell between them free (knock down the wall).
           - Push the neighbor onto the stack.
      4. If no unvisited neighbors, pop the stack (backtrack).
      5. Stop when the stack is empty: every room has been visited exactly once.
    """

    def __init__(self, width: int, height: int):
        assert width % 2 == 1 and height % 2 == 1, "Width and height must both be odd"
        self.width = width    # number of columns
        self.height = height  # number of rows

    def _dfs(self, grid: np.ndarray, rng: np.random.Generator, goal: tuple) -> None:
        """Carves passages into `grid` in-place, starting from the goal cell."""
        # Directions as 2-step jumps (room to room).
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        grid[goal] = 0
        stack = [goal]

        while stack:
            r, c = stack[-1]
            unvisited = [
                (r + dr, c + dc, r + dr // 2, c + dc // 2)
                for dr, dc in directions
                if 1 <= r + dr <= self.height - 2
                and 1 <= c + dc <= self.width - 2
                and grid[r + dr, c + dc] == 1
            ]
            if not unvisited:
                stack.pop()
            else:
                nr, nc, wr, wc = unvisited[rng.integers(len(unvisited))]
                grid[nr, nc] = 0
                grid[wr, wc] = 0
                stack.append((nr, nc))

    def generate_maze(self, seed: int = None, goal: tuple = None) -> Maze:
        """
        Returns a freshly generated Maze. Seed controls the random layout.
        goal defaults to the bottom-right room cell (height-2, width-2).
        """
        rng = np.random.default_rng(seed)
        goal = tuple(goal) if goal is not None else (self.height - 2, self.width - 2)
        grid = np.ones((self.height, self.width), dtype=np.int8)
        self._dfs(grid, rng, goal)
        return Maze(grid, goal)


class MazeEnv(gym.Env):
    """
    Gymnasium environment for maze navigation.

    Each episode the agent starts at a cell chosen by the caller via reset(start=...)
    and must reach the fixed goal. This is the hook the curriculum uses: it passes a
    different start each episode depending on its strategy.

    Observation : one-hot float32 vector of length maze.n_free.
                  Entry i is 1.0 iff the agent is at maze.free_cells[i].
    Action      : Discrete(4), values: 0=up, 1=down, 2=left, 3=right.
                  Moving into a wall leaves the agent in place (no penalty).
    Reward      : +1.0 on reaching the goal, 0.0 otherwise (sparse).
    Termination : reaching the goal (terminated=True).
    Truncation  : exceeding max_steps (truncated=True).
    """

    metadata = {'render_modes': ['ansi']}
    _DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def __init__(self, maze: Maze, max_steps: int = None, render_mode: str = None):
        super().__init__()
        self.maze = maze
        self.max_steps = max_steps if max_steps is not None else maze.n_rows * maze.n_cols
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(maze.n_free,),
            dtype=np.float32,
        )

        # Free cells excluding the goal, used for default random start sampling.
        # Precomputed to avoid filtering on every reset() call.
        self.start_cells = [c for c in maze.free_cells if c != maze.goal]

        self._agent_pos: tuple | None = None
        self._steps: int = 0

    def _make_obs(self, pos: tuple) -> np.ndarray:
        obs = np.zeros(self.maze.n_free, dtype=np.float32)
        obs[self.maze.cell_to_idx[pos]] = 1.0
        return obs

    def reset(self, *, seed: int = None, options: dict = None,
              start: tuple = None) -> tuple:
        """
        start : (r, c) of the desired start cell. The curriculum passes this
                explicitly. If None, samples uniformly from free cells.
        """
        super().reset(seed=seed)
        if start is not None:
            assert self.maze.grid[start] == 0, f"Requested start {start} is a wall"
            self._agent_pos = tuple(start)
        else:
            idx = self.np_random.integers(len(self.start_cells))
            self._agent_pos = self.start_cells[idx]
        self._steps = 0
        return self._make_obs(self._agent_pos), {}

    def step(self, action: int) -> tuple:
        dr, dc = self._DELTAS[action]
        r, c = self._agent_pos
        nr, nc = r + dr, c + dc
        if (0 <= nr < self.maze.n_rows
                and 0 <= nc < self.maze.n_cols
                and self.maze.grid[nr, nc] == 0):
            self._agent_pos = (nr, nc)
        self._steps += 1
        terminated = (self._agent_pos == self.maze.goal)
        truncated = (self._steps >= self.max_steps)
        reward = 1.0 if terminated else 0.0
        return self._make_obs(self._agent_pos), reward, terminated, truncated, {}

    def render(self) -> str | None:
        """
        Print the maze.
        :return: a string representation of the maze.
        """
        if self.render_mode != 'ansi':
            return None
        lines = []
        for r in range(self.maze.n_rows):
            row = []
            for c in range(self.maze.n_cols):
                if (r, c) == self._agent_pos:
                    row.append('A')
                elif (r, c) == self.maze.goal:
                    row.append('G')
                elif self.maze.grid[r, c] == 1:
                    row.append('#')
                else:
                    row.append('.')
            lines.append(''.join(row))
        return '\n'.join(lines)
