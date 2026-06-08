import gymnasium as gym
import minigrid  # noqa: F401 (registers MiniGrid envs with gymnasium)

from envs.wrappers import StartStateWrapper, ActionWrapper, PosObsWrapper


def make_env(env_id: str, selector, seed: int = 0) -> gym.Env:
    env = gym.make(env_id)
    env = StartStateWrapper(env, selector)
    env = ActionWrapper(env)
    env = PosObsWrapper(env)
    return env
