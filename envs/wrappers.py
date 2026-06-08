import numpy as np
import gymnasium as gym
from gymnasium import spaces


# MiniGrid direction encoding:
#   0 = right  (+x)
#   1 = down   (+y)
#   2 = left   (-x)
#   3 = up     (-y)
NUM_DIRS = 4


class StartStateWrapper(gym.Wrapper):
    """Overrides agent start position (and optionally goal) each episode."""

    def __init__(self, env, selector, fixed_goal: tuple[int, int] | None = None):
        super().__init__(env)
        self.selector = selector
        self.fixed_goal = fixed_goal

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.unwrapped

        if self.fixed_goal is not None:
            self._pin_goal(base)

        # Let the selector know which cells are free (it may cache this).
        # Called after goal pinning so the goal cell is excluded from free cells.
        self.selector.set_grid(base)

        col, row = self.selector.sample()
        direction = base.np_random.integers(0, NUM_DIRS)

        base.agent_pos = np.array([col, row])
        base.agent_dir = int(direction)

        obs = base.gen_obs()
        return obs, info

    def _pin_goal(self, base) -> None:
        from minigrid.core.world_object import Goal

        grid = base.grid
        # Remove existing goal(s)
        for x in range(base.width):
            for y in range(base.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    grid.set(x, y, None)
        # Place goal at the fixed position
        gx, gy = self.fixed_goal
        grid.set(gx, gy, Goal())


class PosObsWrapper(gym.ObservationWrapper):
    """Replaces MiniGrid's dict observation with a normalized [col, row, dir] vector."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def observation(self, obs):
        col, row = self.unwrapped.agent_pos
        direction = self.unwrapped.agent_dir
        w = self.unwrapped.width
        h = self.unwrapped.height
        return np.array(
            [col / (w - 1), row / (h - 1), direction / (NUM_DIRS - 1)],
            dtype=np.float32,
        )


class ActionWrapper(gym.ActionWrapper):
    """Restricts action space to the 3 movement actions (turn left, turn right, forward)."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # Identity: MiniGrid actions 0,1,2 are already turn_left, turn_right, forward.
        return action
