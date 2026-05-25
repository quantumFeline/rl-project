"""
Tests for envs/maze.py: Maze data container, MazeGenerator, and MazeEnv.
This file has been generated automatically.
"""
import numpy as np
import pytest

from envs.maze import Maze, MazeGenerator, MazeEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gen7():
    return MazeGenerator(width=7, height=7)

@pytest.fixture
def maze7(gen7):
    return gen7.generate_maze(seed=42)

@pytest.fixture
def env7(maze7):
    return MazeEnv(maze7, render_mode='ansi')


# ---------------------------------------------------------------------------
# MazeGenerator
# ---------------------------------------------------------------------------

class TestMazeGenerator:

    def test_odd_size_required(self):
        with pytest.raises(AssertionError):
            MazeGenerator(width=6, height=7)
        with pytest.raises(AssertionError):
            MazeGenerator(width=7, height=6)

    def test_returns_maze(self, gen7):
        maze = gen7.generate_maze(seed=0)
        assert isinstance(maze, Maze)

    def test_seeded_reproducibility(self, gen7):
        m1 = gen7.generate_maze(seed=99)
        m2 = gen7.generate_maze(seed=99)
        np.testing.assert_array_equal(m1.grid, m2.grid)

    def test_different_seeds_differ(self, gen7):
        m1 = gen7.generate_maze(seed=0)
        m2 = gen7.generate_maze(seed=1)
        assert not np.array_equal(m1.grid, m2.grid)

    def test_borders_are_walls(self, gen7):
        maze = gen7.generate_maze(seed=0)
        g = maze.grid
        assert g[0, :].all()    # top row
        assert g[-1, :].all()   # bottom row
        assert g[:, 0].all()    # left col
        assert g[:, -1].all()   # right col

    def test_grid_values_binary(self, gen7):
        maze = gen7.generate_maze(seed=0)
        unique = set(maze.grid.flatten().tolist())
        assert unique <= {0, 1}

    def test_all_rooms_reachable(self, gen7):
        """Every room cell must be reachable from the goal via BFS."""
        maze = gen7.generate_maze(seed=5)
        g = maze.grid
        n_rows, n_cols = g.shape

        # BFS from the goal over free cells
        visited = set()
        queue = [maze.goal]
        visited.add(maze.goal)
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (0 <= nr < n_rows and 0 <= nc < n_cols
                        and g[nr, nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        assert visited == set(maze.free_cells)

    def test_rectangular_maze(self):
        gen = MazeGenerator(width=7, height=11)
        maze = gen.generate_maze(seed=0)
        assert maze.grid.shape == (11, 7)
        assert maze.goal == (9, 5)

    def test_custom_goal(self, gen7):
        maze = gen7.generate_maze(seed=0, goal=(1, 1))
        assert maze.goal == (1, 1)
        assert maze.grid[1, 1] == 0


# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------

class TestMaze:

    def test_requires_2d_grid(self):
        with pytest.raises(AssertionError):
            Maze(np.zeros((7, 7, 7), dtype=np.int8))

    def test_free_cells_match_grid(self, maze7):
        for r, c in maze7.free_cells:
            assert maze7.grid[r, c] == 0
        # Count matches
        assert len(maze7.free_cells) == int((maze7.grid == 0).sum())

    def test_cell_to_idx_inverse_of_free_cells(self, maze7):
        for i, cell in enumerate(maze7.free_cells):
            assert maze7.cell_to_idx[cell] == i

    def test_goal_is_free(self, maze7):
        assert maze7.grid[maze7.goal] == 0

    def test_default_goal_bottom_right(self, maze7):
        assert maze7.goal == (maze7.n_rows - 2, maze7.n_cols - 2)

    def test_custom_goal(self):
        gen = MazeGenerator(width=7, height=7)
        maze = gen.generate_maze(seed=0, goal=(1, 1))
        assert maze.goal == (1, 1)

    def test_wall_goal_rejected(self):
        gen = MazeGenerator(width=7, height=7)
        maze = gen.generate_maze(seed=0)
        # Manually construct a Maze with a wall cell as goal
        with pytest.raises(AssertionError):
            Maze(maze.grid, goal=(0, 0))  # (0,0) is always a border wall

    def test_accepts_non_odd_grid(self):
        # Maze itself imposes no odd-size constraint; only MazeGenerator does
        grid = np.zeros((4, 6), dtype=np.int8)
        grid[0, :] = 1  # add some walls so goal cell is free
        m = Maze(grid, goal=(1, 1))
        assert m.goal == (1, 1)


# ---------------------------------------------------------------------------
# MazeEnv
# ---------------------------------------------------------------------------

class TestMazeEnv:

    def test_start_cells_exclude_goal(self, env7):
        assert env7.maze.goal not in env7.start_cells

    def test_start_cells_are_free(self, env7):
        for cell in env7.start_cells:
            assert env7.maze.grid[cell] == 0

    def test_reset_obs_is_one_hot(self, env7):
        obs, info = env7.reset(seed=0)
        assert obs.shape == (env7.maze.n_free,)
        assert obs.sum() == pytest.approx(1.0)
        assert set(obs.tolist()) <= {0.0, 1.0}

    def test_reset_default_never_starts_at_goal(self, env7):
        for seed in range(50):
            env7.reset(seed=seed)
            assert env7._agent_pos != env7.maze.goal

    def test_reset_explicit_start(self, env7, maze7):
        start = maze7.free_cells[0]
        env7.reset(start=start)
        assert env7._agent_pos == start

    def test_reset_wall_start_rejected(self, env7, maze7):
        wall = next((r, c) for r in range(maze7.n_rows)
                    for c in range(maze7.n_cols) if maze7.grid[r, c] == 1)
        with pytest.raises(AssertionError):
            env7.reset(start=wall)

    def test_step_obs_is_one_hot(self, env7):
        env7.reset(seed=0)
        obs, reward, terminated, truncated, info = env7.step(0)
        assert obs.sum() == pytest.approx(1.0)

    def test_step_into_wall_stays_in_place(self, env7, maze7):
        # Place agent at goal and step toward border (always a wall)
        env7.reset(start=maze7.goal)
        pos_before = env7._agent_pos
        env7.step(1)  # down -- hits border wall
        # Agent should not have moved if the cell below is a wall
        r, c = pos_before
        nr = r + 1
        if nr >= maze7.n_rows or maze7.grid[nr, c] == 1:
            assert env7._agent_pos == pos_before

    def test_reward_only_at_goal(self, env7, maze7):
        # Step around non-goal cells, expect 0 reward
        env7.reset(start=maze7.free_cells[0])
        _, reward, terminated, _, _ = env7.step(0)
        if not terminated:
            assert reward == 0.0

    def test_termination_at_goal(self, env7, maze7):
        # Start one step away from goal and walk in
        goal = maze7.goal
        # Find a neighbor of the goal that is free
        neighbor = None
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = goal[0]+dr, goal[1]+dc
            if maze7.grid[nr, nc] == 0:
                # Action that moves from (nr,nc) to goal
                action = {(-1,0):1,(1,0):0,(0,-1):3,(0,1):2}[(dr,dc)]
                neighbor = (nr, nc)
                break
        assert neighbor is not None, "Goal has no free neighbor"
        env7.reset(start=neighbor)
        obs, reward, terminated, truncated, _ = env7.step(action)
        assert terminated
        assert reward == pytest.approx(1.0)

    def test_truncation_at_max_steps(self, maze7):
        env = MazeEnv(maze7, max_steps=3)
        env.reset(start=maze7.free_cells[0])
        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(0)
            assert not truncated
        _, _, terminated, truncated, _ = env.step(0)
        if not terminated:
            assert truncated

    def test_render_contains_agent_and_goal(self, env7):
        env7.reset(seed=0)
        rendered = env7.render()
        assert 'A' in rendered
        assert 'G' in rendered
