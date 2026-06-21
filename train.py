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
    selector=None,
    selector_name: str = "uniform",
    fixed_goal: tuple[int, int] | None = None,
    max_steps: int | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if selector is None:
        selector = Uniform(seed=seed)
    env = make_env(env_id, selector, seed=seed, fixed_goal=fixed_goal, max_steps=max_steps)

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
            success = episode_return > 0
            episode_log.append((step, success))
            selector.update(success=success)
            episode_return = 0.0
            obs, _ = env.reset()

        if step >= learning_starts and step % train_every == 0:
            batch = buffer.sample(batch_size)
            td_errors = agent.update(batch)
            selector.update(
                obs_batch=batch.obs.numpy(),
                td_errors=td_errors,
                grid_w=env.unwrapped.width,
                grid_h=env.unwrapped.height,
            )

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

    steps_arr = np.array([s for s, _ in episode_log])
    successes_arr = np.array([float(s) for _, s in episode_log])

    # Plot individual learning curve
    if episode_log:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_short = env_id.replace("MiniGrid-", "").replace("-v0", "").lower()
        save_path = f"curves/{env_short}_{selector_name}_s{seed}_{total_steps // 1000}k_{timestamp}.png"
        plot_success_rate(steps_arr, successes_arr, save_path=save_path)

    return steps_arr, successes_arr


def make_selector(name: str, seed: int, goal: tuple[int, int]):
    if name == "fixed":
        from curriculum.fixed import Fixed
        return Fixed(col=1, row=1)
    if name == "uniform":
        from curriculum.uniform import Uniform
        return Uniform(seed=seed)
    if name == "reverse":
        from curriculum.reverse import Reverse
        return Reverse(goal=goal, seed=seed)
    if name == "tderror":
        from curriculum.td_error import TDError
        return TDError(seed=seed)
    raise ValueError(f"Unknown selector: {name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DQN on MiniGrid")
    parser.add_argument(
        "--selector", type=str, default="uniform",
        choices=["fixed", "uniform", "reverse", "tderror"],
        help="Start-state selector (default: uniform)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument(
        "--env", type=str, default="MiniGrid-FourRooms-v0",
        help="MiniGrid environment ID",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    goal = (17, 1)
    selector = make_selector(args.selector, args.seed, goal)
    print(f"selector: {args.selector} | env: {args.env} | seed: {args.seed} | steps: {args.total_steps}")

    train(
        env_id=args.env,
        total_steps=args.total_steps,
        seed=args.seed,
        device=device,
        selector=selector,
        selector_name=args.selector,
        fixed_goal=goal,
        max_steps=200,
    )
