"""Animate block propagation using matplotlib."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import SimulationConfig
from src.network import build_network
from src.protocols import simulate_naive_flooding, simulate_two_phase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate block propagation")
    parser.add_argument("--protocol", choices=["naive", "two-phase"], default="naive")
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--topology", default="random-regular")
    parser.add_argument("--interval", type=int, default=200, help="Frame interval (ms)")
    parser.add_argument("--save", type=Path, default=None, help="Optional output GIF path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        num_nodes=args.nodes,
        degree=args.degree,
        topology=args.topology,
    )

    # Build graph and run simulation for arrival times.
    adjacency, _, _, _, _, _, _, _ = build_network(config, random.Random(args.seed))
    if args.protocol == "naive":
        result = simulate_naive_flooding(config, random.Random(args.seed))
    else:
        result = simulate_two_phase(config, random.Random(args.seed))

    # Simple circular layout.
    import math

    angles = [2 * math.pi * i / config.num_nodes for i in range(config.num_nodes)]
    positions = [
        (0.5 + 0.4 * math.cos(a), 0.5 + 0.4 * math.sin(a)) for a in angles
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Draw edges.
    for u, neighbors in enumerate(adjacency):
        for v in neighbors:
            if u < v:
                x1, y1 = positions[u]
                x2, y2 = positions[v]
                ax.plot([x1, x2], [y1, y2], color="#cccccc", linewidth=0.5, zorder=1)

    scatter = ax.scatter([], [], s=30, zorder=2)

    times = result.arrival_times
    finite = [t for t in times if t < float("inf")]
    max_time = max(finite) if finite else 1.0
    frames = [i / 50 * max_time for i in range(51)]

    def update(frame_time: float):
        xs, ys, colors = [], [], []
        for idx, t in enumerate(times):
            x, y = positions[idx]
            xs.append(x)
            ys.append(y)
            colors.append("#2ca02c" if t <= frame_time else "#d62728")
        scatter.set_offsets(list(zip(xs, ys)))
        scatter.set_color(colors)
        ax.set_title(f"{args.protocol} @ t={frame_time:.2f}s")
        return scatter,

    anim = FuncAnimation(fig, update, frames=frames, interval=args.interval, blit=True, repeat=False)

    if args.save:
        anim.save(args.save, writer="pillow")
    else:
        plt.show()


if __name__ == "__main__":
    import random

    main()
