from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_success_rate(
    steps: np.ndarray,
    successes: np.ndarray,
    window: int = 50,
    save_path: str = "learning_curve.png",
):
    """Plot rolling success rate over training steps.

    Args:
        steps: shape (n_episodes,), the env step at which each episode ended.
        successes: shape (n_episodes,), 1.0 if the episode reached the goal, 0.0 otherwise.
        window: number of episodes to average over for smoothing.
        save_path: where to save the figure.
    """
    window = min(window, len(successes))

    # Cumulative sum trick for a rolling mean over variable-width windows.
    # For the first (window-1) episodes, average over all episodes so far.
    # After that, average over the full window.
    cumsum = np.cumsum(successes)
    smoothed = np.empty(len(successes))
    for i in range(len(successes)):
        w = min(i + 1, window)
        if i < window:
            smoothed[i] = cumsum[i] / w
        else:
            smoothed[i] = (cumsum[i] - cumsum[i - window]) / window

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, smoothed)
    plt.xlabel("Environment steps")
    plt.ylabel(f"Success rate (rolling {window} episodes)")
    plt.ylim(-0.05, 1.05)
    plt.title("Training progress")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved plot to {save_path}")


def smooth(successes: np.ndarray, window: int = 50) -> np.ndarray:
    window = min(window, len(successes))
    cumsum = np.cumsum(successes)
    smoothed = np.empty(len(successes))
    for i in range(len(successes)):
        if i < window:
            smoothed[i] = cumsum[i] / (i + 1)
        else:
            smoothed[i] = (cumsum[i] - cumsum[i - window]) / window
    return smoothed


def plot_comparison(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    window: int = 50,
    save_path: str = "curves/comparison.png",
):
    """Plot multiple selectors on one figure.

    Args:
        results: {selector_name: (steps_array, successes_array)}
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))

    for name, (steps, successes) in results.items():
        smoothed = smooth(successes, window)
        plt.plot(steps, smoothed, label=name)

    plt.xlabel("Environment steps")
    plt.ylabel(f"Success rate (rolling {window} episodes)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.title("Selector comparison")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")
