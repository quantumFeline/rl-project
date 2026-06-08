from curriculum.base import StartSelector


class Fixed(StartSelector):
    """Always starts the agent at a fixed cell (default: top-left corner)."""

    def __init__(self, col: int = 1, row: int = 1):
        self.col = col
        self.row = row

    def set_grid(self, unwrapped_env) -> None:
        pass

    def sample(self) -> tuple[int, int]:
        return (self.col, self.row)
