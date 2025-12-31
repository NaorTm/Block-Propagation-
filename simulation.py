"""Discrete-event simulation of block propagation in a P2P network.

The module implements two dissemination strategies:
- Naive flooding of the full block.
- A two-phase announce/request/block protocol inspired by Bitcoin.

Both strategies model per-edge latency and a fixed transmission time derived from
block size and bandwidth. Use the CLI to run experiments and summarize results.
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class SimulationConfig:
    """Configurable parameters for a single simulation run."""

    num_nodes: int = 500
    degree: int = 8
    latency_min: float = 0.05
    latency_max: float = 0.2
    bandwidth_mbps: float = 10.0
    block_size_bytes: int = 1_000_000
    source: int = 0

    @property
    def transmission_time(self) -> float:
        """Seconds to transmit the full block over one edge."""

        bits = self.block_size_bytes * 8
        bandwidth_bps = self.bandwidth_mbps * 1_000_000
        return bits / bandwidth_bps


@dataclass
class RunResult:
    """Stores metrics for a single simulation run."""

    protocol: str
    config: SimulationConfig
    arrival_times: List[float]
    t50: float
    t90: float
    t100: float
    messages: Dict[str, int]

    @property
    def total_messages(self) -> int:
        return sum(self.messages.values())


@dataclass
class AggregateResult:
    """Aggregated metrics over multiple runs."""

    protocol: str
    runs: List[RunResult]
    summary: Dict[str, Dict[str, float]]


def generate_random_regular_graph(
    num_nodes: int, degree: int, rng: random.Random, max_attempts: int = 200
) -> List[set]:
    """Generate an approximate random regular graph with rejection sampling.

    The function retries until every node reaches the target degree or the
    attempt limit is exceeded.
    """

    if num_nodes <= 1:
        raise ValueError("Number of nodes must be at least 2")
    if degree >= num_nodes:
        raise ValueError("Degree must be less than the number of nodes")
    if (num_nodes * degree) % 2 != 0:
        raise ValueError("num_nodes * degree must be even")

    for _ in range(max_attempts):
        adjacency = [set() for _ in range(num_nodes)]
        stubs = [node for node in range(num_nodes) for _ in range(degree)]
        rng.shuffle(stubs)
        success = True

        while stubs:
            a = stubs.pop()
            b = stubs.pop()
            if a == b or b in adjacency[a]:
                success = False
                break
            adjacency[a].add(b)
            adjacency[b].add(a)

        if success and all(len(neighbors) == degree for neighbors in adjacency):
            return adjacency

    raise RuntimeError("Failed to generate a random regular graph within attempts")


def assign_latencies(
    adjacency: Sequence[Iterable[int]], latency_min: float, latency_max: float, rng: random.Random
) -> Dict[Tuple[int, int], float]:
    """Assign a symmetric latency to each edge."""

    latencies: Dict[Tuple[int, int], float] = {}
    for u, neighbors in enumerate(adjacency):
        for v in neighbors:
            if u < v:  # ensure single entry per edge
                latencies[(u, v)] = rng.uniform(latency_min, latency_max)
    return latencies


def edge_latency(latencies: Dict[Tuple[int, int], float], u: int, v: int) -> float:
    key = (u, v) if u < v else (v, u)
    return latencies[key]


def threshold_time(arrival_times: Sequence[float], fraction: float) -> float:
    """Return the time when the given fraction of nodes have the block."""

    finite_times = sorted(t for t in arrival_times if math.isfinite(t))
    if not finite_times:
        return math.inf

    target_index = math.ceil(fraction * len(arrival_times)) - 1
    if target_index < 0:
        return 0.0
    if target_index >= len(finite_times):
        return math.inf
    return finite_times[target_index]


def simulate_naive_flooding(config: SimulationConfig, rng: random.Random) -> RunResult:
    adjacency = generate_random_regular_graph(config.num_nodes, config.degree, rng)
    latencies = assign_latencies(adjacency, config.latency_min, config.latency_max, rng)

    arrival_times = [math.inf] * config.num_nodes
    arrival_times[config.source] = 0.0

    pq: List[Tuple[float, str, int, int]] = []  # time, event_type, src, dst
    for neighbor in adjacency[config.source]:
        travel_time = edge_latency(latencies, config.source, neighbor) + config.transmission_time
        heapq.heappush(pq, (travel_time, "block", config.source, neighbor))

    messages = defaultdict(int)
    messages["block"] += len(adjacency[config.source])

    while pq:
        time, event_type, src, dst = heapq.heappop(pq)
        if event_type != "block":
            continue
        if math.isfinite(arrival_times[dst]):
            continue

        arrival_times[dst] = time
        for neighbor in adjacency[dst]:
            travel_time = edge_latency(latencies, dst, neighbor) + config.transmission_time
            heapq.heappush(pq, (time + travel_time, "block", dst, neighbor))
            messages["block"] += 1

    t50 = threshold_time(arrival_times, 0.5)
    t90 = threshold_time(arrival_times, 0.9)
    t100 = threshold_time(arrival_times, 1.0)

    return RunResult(
        protocol="naive",
        config=config,
        arrival_times=arrival_times,
        t50=t50,
        t90=t90,
        t100=t100,
        messages=dict(messages),
    )


def simulate_two_phase(config: SimulationConfig, rng: random.Random) -> RunResult:
    adjacency = generate_random_regular_graph(config.num_nodes, config.degree, rng)
    latencies = assign_latencies(adjacency, config.latency_min, config.latency_max, rng)

    knows_block = [False] * config.num_nodes
    has_full_block = [False] * config.num_nodes
    arrival_times = [math.inf] * config.num_nodes

    source = config.source
    knows_block[source] = True
    has_full_block[source] = True
    arrival_times[source] = 0.0

    pq: List[Tuple[float, str, int, int]] = []
    messages = defaultdict(int)

    def enqueue(event_time: float, event_type: str, src: int, dst: int) -> None:
        heapq.heappush(pq, (event_time, event_type, src, dst))
        messages[event_type] += 1

    # Initial announces
    for neighbor in adjacency[source]:
        latency = edge_latency(latencies, source, neighbor)
        enqueue(latency, "announce", source, neighbor)

    while pq:
        time, event_type, src, dst = heapq.heappop(pq)

        if event_type == "announce":
            if not knows_block[dst]:
                knows_block[dst] = True
                back_latency = edge_latency(latencies, src, dst)
                enqueue(time + back_latency, "request", src, dst)

        elif event_type == "request":
            if has_full_block[src]:
                delivery_latency = edge_latency(latencies, src, dst)
                arrival = time + delivery_latency + config.transmission_time
                enqueue(arrival, "block", src, dst)

        elif event_type == "block":
            if has_full_block[dst]:
                continue
            has_full_block[dst] = True
            arrival_times[dst] = time

            for neighbor in adjacency[dst]:
                if not knows_block[neighbor]:
                    latency = edge_latency(latencies, dst, neighbor)
                    enqueue(time + latency, "announce", dst, neighbor)

        else:
            raise ValueError(f"Unknown event type: {event_type}")

    t50 = threshold_time(arrival_times, 0.5)
    t90 = threshold_time(arrival_times, 0.9)
    t100 = threshold_time(arrival_times, 1.0)

    return RunResult(
        protocol="two-phase",
        config=config,
        arrival_times=arrival_times,
        t50=t50,
        t90=t90,
        t100=t100,
        messages=dict(messages),
    )


def summarize_runs(protocol: str, runs: List[RunResult]) -> AggregateResult:
    def metric_values(selector) -> List[float]:
        return [selector(run) for run in runs]

    summary: Dict[str, Dict[str, float]] = {}
    for name, selector in (
        ("t50", lambda r: r.t50),
        ("t90", lambda r: r.t90),
        ("t100", lambda r: r.t100),
        ("messages", lambda r: r.total_messages),
    ):
        values = metric_values(selector)
        summary[name] = {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    return AggregateResult(protocol=protocol, runs=runs, summary=summary)


def run_experiments(
    protocol: str, runs: int, config: SimulationConfig, seed: int | None
) -> AggregateResult:
    rng = random.Random(seed)
    results: List[RunResult] = []

    for _ in range(runs):
        if protocol == "naive":
            result = simulate_naive_flooding(config, rng)
        elif protocol == "two-phase":
            result = simulate_two_phase(config, rng)
        else:
            raise ValueError("Protocol must be 'naive' or 'two-phase'")
        results.append(result)

    return summarize_runs(protocol, results)


def format_run_result(result: RunResult) -> str:
    parts = [
        f"Protocol: {result.protocol}",
        f"T50: {result.t50:.3f}s, T90: {result.t90:.3f}s, T100: {result.t100:.3f}s",
        "Messages: "
        + ", ".join(f"{k}={v}" for k, v in sorted(result.messages.items())),
    ]
    return " | ".join(parts)


def format_aggregate(aggregate: AggregateResult) -> str:
    lines = [f"Protocol: {aggregate.protocol} (runs={len(aggregate.runs)})"]
    for metric in ("t50", "t90", "t100", "messages"):
        stats = aggregate.summary[metric]
        lines.append(
            f"  {metric.upper()}: mean={stats['mean']:.3f}s min={stats['min']:.3f}s max={stats['max']:.3f}s"
            if metric != "messages"
            else f"  MESSAGES: mean={stats['mean']:.0f} min={stats['min']:.0f} max={stats['max']:.0f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block propagation simulator")
    parser.add_argument("--protocol", choices=["naive", "two-phase"], default="naive")
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--latency-min", type=float, default=0.05)
    parser.add_argument("--latency-max", type=float, default=0.2)
    parser.add_argument("--bandwidth-mbps", type=float, default=10.0)
    parser.add_argument("--block-bytes", type=int, default=1_000_000)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
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
        source=args.source,
    )

    aggregate = run_experiments(args.protocol, args.runs, config, args.seed)

    if args.runs == 1:
        print(format_run_result(aggregate.runs[0]))
    else:
        print(format_aggregate(aggregate))


if __name__ == "__main__":
    main()
