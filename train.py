import torch

from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer
from curriculum.uniform import Uniform
from envs.make_env import make_env


def train(
    env_id: str = "MiniGrid-Empty-8x8-v0",
    total_steps: int = 100_000,
    buffer_capacity: int = 100_000,
    batch_size: int = 128,
    learning_starts: int = 1_000,
    train_every: int = 1,
    target_update_every: int = 1_000,
    eval_every: int = 10_000,
    seed: int = 0,
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selector = Uniform(seed=seed)
    env = make_env(env_id, selector, seed=seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, device=device)
    buffer = ReplayBuffer(capacity=buffer_capacity, obs_dim=obs_dim)

    obs, _ = env.reset(seed=seed)
    for step in range(total_steps):
        action = agent.act(obs, step)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.push(obs, action, reward, next_obs, terminated)
        obs = next_obs

        if terminated or truncated:
            # TODO: episode bookkeeping (return, length) for logging + curriculum
            obs, _ = env.reset()

        if step >= learning_starts and step % train_every == 0:
            batch = buffer.sample(batch_size)
            # TODO: agent.update returns the per-sample TD error; the curriculum
            # will later consume it to weight start states.
            agent.update(batch)

        if step % target_update_every == 0:
            agent.sync_target()

        if step % eval_every == 0:
            # TODO: evaluate greedy success rate from a fixed start set
            pass

    env.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    train(device=device)
