"""Run a comprehensive test suite of simulation scenarios and export CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import SimulationConfig
from src.metrics import macro_metrics
from src.simulator import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all simulation scenarios")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/all_tests_summary.csv"),
        help="CSV output path",
    )
    return parser.parse_args()


def scenarios() -> list[dict]:
    return [
        {
            "name": "baseline_naive",
            "protocol": "naive",
            "config": SimulationConfig(),
        },
        {
            "name": "baseline_two_phase",
            "protocol": "two-phase",
            "config": SimulationConfig(),
        },
        {
            "name": "push_fanout",
            "protocol": "push",
            "config": SimulationConfig(gossip_fanout=4),
        },
        {
            "name": "pull_interval",
            "protocol": "pull",
            "config": SimulationConfig(pull_interval=0.5, pull_fanout=2, max_time=20),
        },
        {
            "name": "push_pull",
            "protocol": "push-pull",
            "config": SimulationConfig(gossip_fanout=4, pull_interval=0.5, max_time=20),
        },
        {
            "name": "scale_free",
            "protocol": "two-phase",
            "config": SimulationConfig(topology="scale-free", scale_free_m=3),
        },
        {
            "name": "small_world",
            "protocol": "two-phase",
            "config": SimulationConfig(topology="small-world", degree=8, rewire_prob=0.2),
        },
        {
            "name": "relay_overlay",
            "protocol": "two-phase",
            "config": SimulationConfig(
                relay_fraction=0.2,
                relay_overlay_degree=2,
                relay_latency_mult=0.5,
                relay_bandwidth_mult=2.0,
            ),
        },
        {
            "name": "compact_blocks",
            "protocol": "bitcoin-compact",
            "config": SimulationConfig(
                mempool_overlap_mean=0.85,
                mempool_overlap_std=0.05,
            ),
        },
        {
            "name": "bottlenecks",
            "protocol": "two-phase",
            "config": SimulationConfig(
                bottleneck_fraction=0.1,
                bottleneck_latency_mult=3.0,
                bottleneck_bandwidth_mult=0.5,
            ),
        },
        {
            "name": "churn_delay",
            "protocol": "push-pull",
            "config": SimulationConfig(
                churn_prob=0.1,
                churn_time_min=1.0,
                churn_time_max=3.0,
                delay_prob=0.2,
                delay_latency_mult=2.0,
                delay_bandwidth_mult=0.7,
                pull_interval=0.5,
                max_time=20,
            ),
        },
        {
            "name": "macro_metrics",
            "protocol": "two-phase",
            "config": SimulationConfig(),
            "macro": True,
        },
    ]


def main() -> None:
    args = parse_args()
    rows = []
    for scenario in scenarios():
        aggregate = run_experiments(
            scenario["protocol"], args.runs, scenario["config"], args.seed
        )
        summary = aggregate.summary
        macro = [
            macro_metrics(run.arrival_times, 600.0) for run in aggregate.runs
        ]
        macro_row = {
            "compete_p_t90_mean": sum(m.competing_block_prob_t90 for m in macro) / len(macro),
            "lambda_t100_mean": sum(m.expected_competing_blocks_t100 for m in macro) / len(macro),
            "p_ge1_t100_mean": sum(m.prob_competing_blocks_ge1_t100 for m in macro) / len(macro),
            "p_ge2_t100_mean": sum(m.prob_competing_blocks_ge2_t100 for m in macro) / len(macro),
            "security_margin_t50_mean": sum(m.security_margin_t50 for m in macro) / len(macro),
        }
        row = {
            "scenario": scenario["name"],
            "protocol": scenario["protocol"],
            "t50_mean": summary["t50"]["mean"],
            "t90_mean": summary["t90"]["mean"],
            "t100_mean": summary["t100"]["mean"],
            "messages_mean": summary["messages"]["mean"],
        }
        row.update(macro_row)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
