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
    t50 = [float(row["t50_mean"]) for row in rows]
    t90 = [float(row["t90_mean"]) for row in rows]
    t100 = [float(row["t100_mean"]) for row in rows]
    messages = [float(row["messages_mean"]) for row in rows]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6), sharex=True)

    x = range(len(labels))
    width = 0.25
    ax1.bar([i - width for i in x], t50, width=width, label="T50 (s)")
    ax1.bar(x, t90, width=width, label="T90 (s)")
    ax1.bar([i + width for i in x], t100, width=width, label="T100 (s)")
    ax1.set_ylabel("Propagation time (s)")
    ax1.legend()
    ax1.set_title("Scenario Comparison")

    ax2.plot(labels, messages, color="#ff7f0e", marker="o")
    ax2.set_ylabel("Messages")
    ax2.tick_params(axis="x", rotation=45, labelsize=8)

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
