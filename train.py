import time
import torch
import numpy as np

from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer
from curriculum.uniform import Uniform
from envs.make_env import make_env
from plot import plot_success_rate


def train(
    env_id: str = "MiniGrid-Empty-8x8-v0",
    total_steps: int = 100_000,
    buffer_capacity: int = 100_000,
    batch_size: int = 128,
    learning_starts: int = 1_000,
    train_every: int = 1,
    target_update_every: int = 1_000,
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

    # Episode tracking: each entry is (step_at_which_episode_ended, success_bool)
    episode_log = []
    episode_return = 0.0
    log_every = max(1, total_steps // 20)
    t_start = time.perf_counter()

    obs, _ = env.reset(seed=seed)
    for step in range(total_steps):
        action = agent.act(obs, step)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.push(obs, action, reward, next_obs, terminated)
        obs = next_obs
        episode_return += reward

        if terminated or truncated:
            episode_log.append((step, episode_return > 0))
            episode_return = 0.0
            obs, _ = env.reset()

        if step >= learning_starts and step % train_every == 0:
            batch = buffer.sample(batch_size)
            agent.update(batch)

        if step % target_update_every == 0:
            agent.sync_target()

        if step > 0 and step % log_every == 0:
            elapsed = time.perf_counter() - t_start
            steps_per_sec = step / elapsed
            eta = (total_steps - step) / steps_per_sec
            recent = episode_log[-50:] if episode_log else []
            success_rate = sum(s for _, s in recent) / len(recent) if recent else 0.0
            print(
                f"step {step}/{total_steps} "
                f"({100*step/total_steps:.0f}%) | "
                f"eps {agent._epsilon(step):.3f} | "
                f"success {success_rate:.2f} (last {len(recent)} ep) | "
                f"{steps_per_sec:.0f} steps/s | "
                f"ETA {eta:.0f}s"
            )

    env.close()

    # Plot learning curve
    if episode_log:
        steps = np.array([s for s, _ in episode_log])
        successes = np.array([float(s) for _, s in episode_log])
        plot_success_rate(steps, successes)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    train(device=device)
