import numpy as np
import torch
from typing import NamedTuple


class Batch(NamedTuple):
    obs:        torch.Tensor   # float32, shape (batch_size, obs_dim)
    actions:    torch.Tensor   # int64,   shape (batch_size,)
    rewards:    torch.Tensor   # float32, shape (batch_size,)
    next_obs:   torch.Tensor   # float32, shape (batch_size, obs_dim)
    terminated: torch.Tensor   # float32, shape (batch_size,) -- 0.0 or 1.0


class ReplayBuffer:

    def __init__(self, capacity: int, obs_dim: int):
        pass

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, terminated: bool) -> None:
        """obs and next_obs are numpy arrays from the gymnasium env."""

    def sample(self, batch_size: int) -> Batch:
        """Returns tensors on CPU. Caller moves to device with .to(device)."""

    def __len__(self) -> int:
        pass
