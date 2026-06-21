"""Run all selector conditions and produce a combined comparison plot."""

import argparse
from datetime import datetime

import torch

from train import train, make_selector, env_config
from plot import plot_comparison


SELECTORS = ["fixed", "uniform", "reverse", "tderror"]


def run_comparison(
    selectors: list[str],
    env_id: str,
    total_steps: int,
    seed: int,
    device: torch.device,
):
    cfg = env_config(env_id)
    goal = cfg["goal"]
    results = {}

    for name in selectors:
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}\n")

        selector = make_selector(name, seed, goal)
        steps, successes = train(
            env_id=env_id,
            total_steps=total_steps,
            seed=seed,
            device=device,
            selector=selector,
            selector_name=name,
            fixed_goal=goal,
            max_steps=cfg["max_steps"],
        )
        results[name] = (steps, successes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = env_id.replace("MiniGrid-", "").replace("-v0", "").lower()
    save_path = f"curves/{env_short}_comparison_s{seed}_{total_steps // 1000}k_{timestamp}.png"
    plot_comparison(results, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare start-state selectors")
    parser.add_argument(
        "--selectors", nargs="+", default=SELECTORS,
        choices=SELECTORS,
        help=f"Which selectors to compare (default: all)",
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

    run_comparison(
        selectors=args.selectors,
        env_id=args.env,
        total_steps=args.total_steps,
        seed=args.seed,
        device=device,
    )
