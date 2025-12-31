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

    parsed = []
    for row in rows:
        parsed.append(
            {
                "label": row["scenario"],
                "t50": float(row["t50_mean"]),
                "t90": float(row["t90_mean"]),
                "t100": float(row["t100_mean"]),
                "messages": float(row["messages_mean"]),
            }
        )

    parsed.sort(key=lambda item: item["t90"])
    labels = [row["label"] for row in parsed]
    t50 = [row["t50"] for row in parsed]
    t90 = [row["t90"] for row in parsed]
    t100 = [row["t100"] for row in parsed]
    messages = [row["messages"] for row in parsed]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6), sharex=True)

    x = range(len(labels))
    ax1.plot(x, t50, marker="o", label="T50 (s)")
    ax1.plot(x, t90, marker="o", label="T90 (s)")
    ax1.plot(x, t100, marker="o", label="T100 (s)")
    ax1.set_ylabel("Propagation time (s)")
    ax1.legend()
    ax1.set_title("Scenario Comparison (sorted by T90)")

    ax2.bar(x, messages, color="#ff7f0e")
    ax2.set_ylabel("Messages")
    ax2.set_xticks(list(x), labels, rotation=45, fontsize=8)

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
