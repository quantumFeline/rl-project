from abc import ABC, abstractmethod


class StartSelector(ABC):

    @abstractmethod
    def set_grid(self, unwrapped_env) -> None:
        """Called every reset. Inspect the grid to determine valid start cells."""

    @abstractmethod
    def sample(self) -> tuple[int, int]:
        """Return (col, row) for the agent's start position this episode."""

    def update(self, **kwargs) -> None:
        """Receive training feedback (TD errors, episode returns, etc.). Optional."""
