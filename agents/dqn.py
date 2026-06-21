import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP that maps a state vector to Q-values for each action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device: torch.device,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 80_000,
    ):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.q_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net = QNetwork(obs_dim, n_actions).to(device)
        self.sync_target()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def _epsilon(self, step: int) -> float:
        """Linear decay from epsilon_start to epsilon_end over epsilon_decay_steps."""
        progress = min(step / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def act(self, obs: np.ndarray, step: int) -> int:
        """Epsilon-greedy: random with probability epsilon, greedy otherwise."""
        if np.random.random() < self._epsilon(step):
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            q_values = self.q_net(obs_t.unsqueeze(0))  # add batch dim
            return int(q_values.argmax(dim=1).item())

    def update(self, batch) -> np.ndarray:
        """One gradient step on a batch from the replay buffer.
        Returns per-sample absolute TD errors as a numpy array, shape (batch_size,)."""
        obs = batch.obs.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_obs = batch.next_obs.to(self.device)
        terminated = batch.terminated.to(self.device)

        # Q(s, a) for the actions that were actually taken in these transitions
        q_values = self.q_net(obs)                          # (batch, n_actions)
        q_sa = q_values.gather(1, actions.unsqueeze(1))     # (batch, 1)
        q_sa = q_sa.squeeze(1)                              # (batch,)

        # Double DQN: online net picks the action, target net evaluates it
        with torch.no_grad():
            best_actions = self.q_net(next_obs).argmax(dim=1)
            next_q = self.target_net(next_obs).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            td_target = rewards + self.gamma * next_q * (1.0 - terminated)

        td_error = td_target - q_sa
        loss = nn.functional.smooth_l1_loss(q_sa, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return td_error.abs().detach().cpu().numpy()

    def sync_target(self) -> None:
        """Hard-copy Q-network weights into the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())
