"""Parse node logs into simulation-compatible CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from datetime import datetime, timezone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse node logs")
    parser.add_argument("--input", type=Path, required=True, help="Log file or folder")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/real_world_summary.csv")
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional block-level summary CSV (t50/t90/t100)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "bitcoin-core"],
        default="csv",
        help="Input format: csv or bitcoin-core debug.log",
    )
    parser.add_argument(
        "--node-id",
        type=str,
        default=None,
        help="Optional node id override for single-file inputs",
    )
    return parser.parse_args()


def iter_input_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if path.is_dir():
        for item in sorted(path.iterdir()):
            if item.is_file():
                yield item
    else:
        yield path


def read_events_csv(files: Iterable[Path], node_id_override: str | None) -> List[Tuple[str, str, str, float]]:
    events: List[Tuple[str, str, str, float]] = []
    for file_path in files:
        with file_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                node_id = (node_id_override or row.get("node_id") or "").strip()
                event = (row.get("event") or "").strip().lower()
                block_id = (row.get("block_id") or "").strip()
                timestamp_raw = (row.get("timestamp") or "").strip()
                if not node_id or not event or not timestamp_raw:
                    continue
                if event not in {"header", "block"}:
                    continue
                try:
                    timestamp = float(timestamp_raw)
                except ValueError:
                    continue
                if not block_id:
                    block_id = "unknown"
                events.append((node_id, event, block_id, timestamp))
    return events


def _parse_iso_ts(value: str) -> float | None:
    cleaned = value.rstrip("Z")
    if value.endswith("Z"):
        cleaned = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def read_events_bitcoin_core(
    files: Iterable[Path], node_id_override: str | None
) -> List[Tuple[str, str, str, float]]:
    events: List[Tuple[str, str, str, float]] = []
    patterns = [
        re.compile(
            r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+.*received block\s+(?P<hash>[0-9a-fA-F]{64})"
        ),
        re.compile(
            r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+.*new block\s+(?P<hash>[0-9a-fA-F]{64})"
        ),
        re.compile(
            r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+UpdateTip: new best=(?P<hash>[0-9a-fA-F]{64})"
        ),
    ]

    for file_path in files:
        node_id = node_id_override or file_path.stem
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                for pattern in patterns:
                    match = pattern.search(line)
                    if not match:
                        continue
                    ts_raw = match.group("ts")
                    block_id = match.group("hash")
                    ts = _parse_iso_ts(ts_raw)
                    if ts is None:
                        continue
                    events.append((node_id, "block", block_id, ts))
                    break
    return events


def first_block_times(
    events: Iterable[Tuple[str, str, str, float]]
) -> Dict[Tuple[str, str], float]:
    first_times: Dict[Tuple[str, str], float] = {}
    for node_id, event, block_id, timestamp in events:
        if event != "block":
            continue
        key = (node_id, block_id)
        if key not in first_times or timestamp < first_times[key]:
            first_times[key] = timestamp
    return first_times


def write_per_node_summary(
    output: Path, first_times: Dict[Tuple[str, str], float]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "block_id", "first_block_time"])
        for (node_id, block_id), timestamp in sorted(first_times.items()):
            writer.writerow([node_id, block_id, f"{timestamp:.6f}"])


def write_block_summary(
    output: Path, first_times: Dict[Tuple[str, str], float]
) -> None:
    blocks: Dict[str, List[float]] = {}
    for (_, block_id), timestamp in first_times.items():
        blocks.setdefault(block_id, []).append(timestamp)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block_id", "nodes", "t50", "t90", "t100"])
        for block_id, times in sorted(blocks.items()):
            times.sort()
            nodes = len(times)
            t50 = times[int(max(0, round(0.5 * nodes) - 1))]
            t90 = times[int(max(0, round(0.9 * nodes) - 1))]
            t100 = times[-1]
            writer.writerow([block_id, nodes, f"{t50:.6f}", f"{t90:.6f}", f"{t100:.6f}"])


def main() -> None:
    args = parse_args()
    files = list(iter_input_files(args.input))
    if args.node_id and len(files) > 1:
        raise SystemExit("--node-id can only be used with a single input file.")
    if args.format == "csv":
        events = read_events_csv(files, args.node_id)
    elif args.format == "bitcoin-core":
        events = read_events_bitcoin_core(files, args.node_id)
    else:
        raise SystemExit(f"Unsupported format: {args.format}")
    if not events:
        raise SystemExit("No valid events found. Check input CSV format.")
    first_times = first_block_times(events)
    write_per_node_summary(args.output, first_times)
    if args.summary_out:
        write_block_summary(args.summary_out, first_times)


if __name__ == "__main__":
    main()
