"""Plot CSV summaries produced by run_all.py or run_series.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scenario summaries")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/all_tests_summary.csv"),
        help="Input CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/summary_plot.png"),
        help="Output image path",
    )
    return parser.parse_args()


def read_rows(path: Path):
    import csv

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit("No rows found in input CSV.")

    labels = [row["scenario"] for row in rows]
    t90 = [float(row["t90_mean"]) for row in rows]
    messages = [float(row["messages_mean"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(labels, t90, color="#1f77b4", alpha=0.7, label="T90 (s)")
    ax1.set_ylabel("T90 (s)")
    ax1.tick_params(axis="x", rotation=45, labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(labels, messages, color="#ff7f0e", marker="o", label="Messages")
    ax2.set_ylabel("Messages")

    ax1.set_title("Scenario Comparison")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
