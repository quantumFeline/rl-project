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
