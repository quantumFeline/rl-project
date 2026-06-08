import numpy as np
import torch
from typing import NamedTuple


class Batch(NamedTuple):
    obs:        torch.Tensor   # float32, shape (batch_size, obs_dim)
    actions:    torch.Tensor   # int64,   shape (batch_size,)
    rewards:    torch.Tensor   # float32, shape (batch_size,)
    next_obs:   torch.Tensor   # float32, shape (batch_size, obs_dim)
    terminated: torch.Tensor   # float32, shape (batch_size,), 0.0 or 1.0


class ReplayBuffer:

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.pos = 0
        self.size = 0

        # Store as numpy for fast random indexing; convert to tensors at sample time
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, terminated: bool) -> None:
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminated[i] = float(terminated)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        return Batch(
            obs=torch.from_numpy(self.obs[idx]),
            actions=torch.from_numpy(self.actions[idx]),
            rewards=torch.from_numpy(self.rewards[idx]),
            next_obs=torch.from_numpy(self.next_obs[idx]),
            terminated=torch.from_numpy(self.terminated[idx]),
        )

    def __len__(self) -> int:
        return self.size
