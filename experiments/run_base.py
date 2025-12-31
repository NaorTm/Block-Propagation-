"""Baseline experiment runner and CDF plot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import SimulationConfig
from src.simulator import format_aggregate, run_experiments


def plot_cdf(arrival_times, label: str) -> None:
    finite = sorted(t for t in arrival_times if t < float("inf"))
    if not finite:
        return
    xs = finite
    ys = [(i + 1) / len(finite) for i in range(len(finite))]
    plt.plot(xs, ys, label=label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments and plot CDFs")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--latency-min", type=float, default=0.05)
    parser.add_argument("--latency-max", type=float, default=0.2)
    parser.add_argument("--bandwidth-mbps", type=float, default=10.0)
    parser.add_argument("--block-bytes", type=int, default=1_000_000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/baseline_cdf.png"),
        help="Path to save the CDF plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        num_nodes=args.nodes,
        degree=args.degree,
        latency_min=args.latency_min,
        latency_max=args.latency_max,
        bandwidth_mbps=args.bandwidth_mbps,
        block_size_bytes=args.block_bytes,
    )

    naive = run_experiments("naive", args.runs, config, args.seed)
    two_phase = run_experiments("two-phase", args.runs, config, args.seed)

    print(format_aggregate(naive))
    print(format_aggregate(two_phase))

    plt.figure(figsize=(7, 4))
    plot_cdf(naive.runs[0].arrival_times, "naive")
    plot_cdf(two_phase.runs[0].arrival_times, "two-phase")
    plt.xlabel("Arrival time (s)")
    plt.ylabel("CDF")
    plt.title("Block arrival CDF (single run)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
