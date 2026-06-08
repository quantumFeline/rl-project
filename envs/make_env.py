import gymnasium as gym
import minigrid  # noqa: F401 (registers MiniGrid envs with gymnasium)

from envs.wrappers import StartStateWrapper, ActionWrapper, PosObsWrapper


def make_env(
    env_id: str,
    selector,
    seed: int = 0,
    fixed_goal: tuple[int, int] | None = None,
    max_steps: int | None = None,
) -> gym.Env:
    kwargs = {}
    if max_steps is not None:
        kwargs["max_episode_steps"] = max_steps
    env = gym.make(env_id, **kwargs)
    env = StartStateWrapper(env, selector, fixed_goal=fixed_goal)
    env = ActionWrapper(env)
    env = PosObsWrapper(env)
    return env
