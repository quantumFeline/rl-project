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
    """Overrides the agent's start position each episode using a curriculum selector."""

    def __init__(self, env, selector):
        super().__init__(env)
        self.selector = selector

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Let the selector know which cells are free (it may cache this).
        self.selector.set_grid(self.unwrapped)

        col, row = self.selector.sample()
        direction = self.unwrapped.np_random.integers(0, NUM_DIRS)

        self.unwrapped.agent_pos = np.array([col, row])
        self.unwrapped.agent_dir = int(direction)

        obs = self.unwrapped.gen_obs()
        return obs, info


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
