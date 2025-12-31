"""Run a series of beyond-baseline experiments and export CSV summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import SimulationConfig
from src.simulator import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario sweep experiments")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/series_summary.csv"),
        help="CSV output path",
    )
    return parser.parse_args()


def scenario_rows() -> list[dict]:
    return [
        {
            "name": "baseline",
            "protocol": "naive",
            "config": SimulationConfig(),
        },
        {
            "name": "fanout_push",
            "protocol": "push",
            "config": SimulationConfig(gossip_fanout=4),
        },
        {
            "name": "topology_scale_free",
            "protocol": "two-phase",
            "config": SimulationConfig(topology="scale-free", scale_free_m=3),
        },
        {
            "name": "topology_small_world",
            "protocol": "two-phase",
            "config": SimulationConfig(topology="small-world", degree=8, rewire_prob=0.2),
        },
        {
            "name": "noise_churn",
            "protocol": "two-phase",
            "config": SimulationConfig(drop_prob=0.1),
        },
        {
            "name": "bottleneck_nodes",
            "protocol": "two-phase",
            "config": SimulationConfig(
                bottleneck_fraction=0.1,
                bottleneck_latency_mult=3.0,
                bottleneck_bandwidth_mult=0.5,
            ),
        },
        {
            "name": "bitcoin_compact",
            "protocol": "bitcoin-compact",
            "config": SimulationConfig(
                compact_block_bytes=20_000,
                compact_success_prob=0.9,
            ),
        },
    ]


def main() -> None:
    args = parse_args()
    rows = []
    for scenario in scenario_rows():
        aggregate = run_experiments(
            scenario["protocol"], args.runs, scenario["config"], args.seed
        )
        summary = aggregate.summary
        rows.append(
            {
                "scenario": scenario["name"],
                "protocol": scenario["protocol"],
                "t50_mean": summary["t50"]["mean"],
                "t90_mean": summary["t90"]["mean"],
                "t100_mean": summary["t100"]["mean"],
                "messages_mean": summary["messages"]["mean"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
