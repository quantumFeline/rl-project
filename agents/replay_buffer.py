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

        self.obs = torch.empty((capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.empty(capacity, dtype=torch.int64)
        self.rewards = torch.empty(capacity, dtype=torch.float32)
        self.next_obs = torch.empty((capacity, obs_dim), dtype=torch.float32)
        self.terminated = torch.empty(capacity, dtype=torch.float32)

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, terminated: bool) -> None:
        i = self.pos
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32)
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = torch.as_tensor(next_obs, dtype=torch.float32)
        self.terminated[i] = float(terminated)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = torch.randint(0, self.size, (batch_size,))
        return Batch(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            terminated=self.terminated[idx],
        )

    def __len__(self) -> int:
        return self.size
