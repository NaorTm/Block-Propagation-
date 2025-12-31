"""Discrete-event simulation of block propagation in a P2P network.

The module implements two dissemination strategies:
- Naive flooding of the full block.
- A two-phase announce/request/block protocol inspired by Bitcoin.

Both strategies model per-edge latency and a fixed transmission time derived from
block size and bandwidth, with optional forwarding churn. Use the CLI to run
experiments and summarize results.
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

import networkx as nx


@dataclass(frozen=True)
class SimulationConfig:
    """Configurable parameters for a single simulation run."""

    num_nodes: int = 500
    degree: int = 8
    topology: str = "random-regular"
    rewire_prob: float = 0.1
    scale_free_m: int | None = None
    latency_dist: str = "uniform"
    latency_min: float = 0.05
    latency_max: float = 0.2
    latency_mu: float = -2.3
    latency_sigma: float = 0.4
    bandwidth_dist: str = "fixed"
    bandwidth_mbps: float = 10.0
    bandwidth_min: float = 5.0
    bandwidth_max: float = 25.0
    bandwidth_mu: float = 2.3
    bandwidth_sigma: float = 0.3
    block_size_bytes: int = 1_000_000
    drop_prob: float = 0.0
    source: int = 0

    def transmission_time(self, bandwidth_mbps: float) -> float:
        """Seconds to transmit the full block over one edge."""

        bits = self.block_size_bytes * 8
        bandwidth_bps = bandwidth_mbps * 1_000_000
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


def adjacency_from_graph(graph: nx.Graph) -> List[set]:
    return [set(graph.neighbors(node)) for node in graph.nodes]


def generate_graph(config: SimulationConfig, rng: random.Random) -> List[set]:
    if config.topology == "random-regular":
        return generate_random_regular_graph(config.num_nodes, config.degree, rng)
    if config.topology == "scale-free":
        m = config.scale_free_m or max(1, config.degree // 2)
        if m >= config.num_nodes:
            raise ValueError("scale_free_m must be less than num_nodes")
        seed = rng.randint(0, 2**32 - 1)
        graph = nx.barabasi_albert_graph(config.num_nodes, m, seed=seed)
        return adjacency_from_graph(graph)
    if config.topology == "small-world":
        if config.degree % 2 != 0:
            raise ValueError("degree must be even for small-world topology")
        seed = rng.randint(0, 2**32 - 1)
        graph = nx.watts_strogatz_graph(
            config.num_nodes, config.degree, config.rewire_prob, seed=seed
        )
        return adjacency_from_graph(graph)
    raise ValueError(f"Unknown topology: {config.topology}")


def assign_latencies(
    adjacency: Sequence[Iterable[int]], config: SimulationConfig, rng: random.Random
) -> Dict[Tuple[int, int], float]:
    """Assign a symmetric latency to each edge."""

    latencies: Dict[Tuple[int, int], float] = {}
    for u, neighbors in enumerate(adjacency):
        for v in neighbors:
            if u < v:  # ensure single entry per edge
                if config.latency_dist == "uniform":
                    latency = rng.uniform(config.latency_min, config.latency_max)
                elif config.latency_dist == "lognormal":
                    latency = rng.lognormvariate(config.latency_mu, config.latency_sigma)
                else:
                    raise ValueError(f"Unknown latency distribution: {config.latency_dist}")
                latencies[(u, v)] = latency
    return latencies


def assign_bandwidths(
    adjacency: Sequence[Iterable[int]], config: SimulationConfig, rng: random.Random
) -> Dict[Tuple[int, int], float]:
    """Assign a symmetric bandwidth (Mbps) to each edge."""

    bandwidths: Dict[Tuple[int, int], float] = {}
    for u, neighbors in enumerate(adjacency):
        for v in neighbors:
            if u < v:
                if config.bandwidth_dist == "fixed":
                    bandwidth = config.bandwidth_mbps
                elif config.bandwidth_dist == "uniform":
                    bandwidth = rng.uniform(config.bandwidth_min, config.bandwidth_max)
                elif config.bandwidth_dist == "lognormal":
                    bandwidth = rng.lognormvariate(config.bandwidth_mu, config.bandwidth_sigma)
                else:
                    raise ValueError(f"Unknown bandwidth distribution: {config.bandwidth_dist}")
                bandwidths[(u, v)] = bandwidth
    return bandwidths


def edge_latency(latencies: Dict[Tuple[int, int], float], u: int, v: int) -> float:
    key = (u, v) if u < v else (v, u)
    return latencies[key]


def edge_bandwidth(bandwidths: Dict[Tuple[int, int], float], u: int, v: int) -> float:
    key = (u, v) if u < v else (v, u)
    return bandwidths[key]


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
    adjacency = generate_graph(config, rng)
    latencies = assign_latencies(adjacency, config, rng)
    bandwidths = assign_bandwidths(adjacency, config, rng)

    arrival_times = [math.inf] * config.num_nodes
    arrival_times[config.source] = 0.0

    will_forward = [
        True if node == config.source else rng.random() >= config.drop_prob
        for node in range(config.num_nodes)
    ]

    pq: List[Tuple[float, str, int, int]] = []  # time, event_type, src, dst
    if will_forward[config.source]:
        for neighbor in adjacency[config.source]:
            travel_time = edge_latency(latencies, config.source, neighbor) + config.transmission_time(
                edge_bandwidth(bandwidths, config.source, neighbor)
            )
            heapq.heappush(pq, (travel_time, "block", config.source, neighbor))

    messages = defaultdict(int)
    if will_forward[config.source]:
        messages["block"] += len(adjacency[config.source])

    while pq:
        time, event_type, src, dst = heapq.heappop(pq)
        if event_type != "block":
            continue
        if math.isfinite(arrival_times[dst]):
            continue

        arrival_times[dst] = time
        if not will_forward[dst]:
            continue
        for neighbor in adjacency[dst]:
            travel_time = edge_latency(latencies, dst, neighbor) + config.transmission_time(
                edge_bandwidth(bandwidths, dst, neighbor)
            )
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
    adjacency = generate_graph(config, rng)
    latencies = assign_latencies(adjacency, config, rng)
    bandwidths = assign_bandwidths(adjacency, config, rng)

    knows_block = [False] * config.num_nodes
    has_full_block = [False] * config.num_nodes
    arrival_times = [math.inf] * config.num_nodes

    source = config.source
    will_forward = [
        True if node == source else rng.random() >= config.drop_prob
        for node in range(config.num_nodes)
    ]
    knows_block[source] = True
    has_full_block[source] = True
    arrival_times[source] = 0.0

    pq: List[Tuple[float, str, int, int]] = []
    messages = defaultdict(int)

    def enqueue(event_time: float, event_type: str, src: int, dst: int) -> None:
        heapq.heappush(pq, (event_time, event_type, src, dst))
        messages[event_type] += 1

    # Initial announces
    if will_forward[source]:
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
            if has_full_block[src] and will_forward[src]:
                delivery_latency = edge_latency(latencies, src, dst)
                arrival = time + delivery_latency + config.transmission_time(
                    edge_bandwidth(bandwidths, src, dst)
                )
                enqueue(arrival, "block", src, dst)

        elif event_type == "block":
            if has_full_block[dst]:
                continue
            has_full_block[dst] = True
            arrival_times[dst] = time

            if will_forward[dst]:
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


def format_distribution(arrival_times: Sequence[float], bins: int) -> str:
    finite_times = sorted(t for t in arrival_times if math.isfinite(t))
    if not finite_times:
        return "No arrivals to summarize."
    if bins <= 0:
        raise ValueError("bins must be positive")

    t_min = finite_times[0]
    t_max = finite_times[-1]
    if t_max == t_min:
        return f"All arrivals at {t_min:.3f}s."

    width = (t_max - t_min) / bins
    counts = [0] * bins
    for t in finite_times:
        idx = min(int((t - t_min) / width), bins - 1)
        counts[idx] += 1

    total = len(finite_times)
    cumulative = 0
    lines = [f"Arrival time histogram (bins={bins})"]
    for i, count in enumerate(counts):
        start = t_min + i * width
        end = start + width
        cumulative += count
        cdf = cumulative / total
        lines.append(f"  [{start:.3f}, {end:.3f}): count={count} cdf={cdf:.3f}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block propagation simulator")
    parser.add_argument("--protocol", choices=["naive", "two-phase"], default="naive")
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument(
        "--topology",
        choices=["random-regular", "scale-free", "small-world"],
        default="random-regular",
    )
    parser.add_argument(
        "--rewire-prob",
        type=float,
        default=0.1,
        help="Small-world rewiring probability",
    )
    parser.add_argument(
        "--scale-free-m",
        type=int,
        default=None,
        help="Scale-free attachment parameter (avg degree ~ 2*m)",
    )
    parser.add_argument("--latency-min", type=float, default=0.05)
    parser.add_argument("--latency-max", type=float, default=0.2)
    parser.add_argument(
        "--latency-dist",
        choices=["uniform", "lognormal"],
        default="uniform",
    )
    parser.add_argument(
        "--latency-mu",
        type=float,
        default=-2.3,
        help="Lognormal mu for latency when --latency-dist=lognormal",
    )
    parser.add_argument(
        "--latency-sigma",
        type=float,
        default=0.4,
        help="Lognormal sigma for latency when --latency-dist=lognormal",
    )
    parser.add_argument("--bandwidth-mbps", type=float, default=10.0)
    parser.add_argument(
        "--bandwidth-dist",
        choices=["fixed", "uniform", "lognormal"],
        default="fixed",
    )
    parser.add_argument(
        "--bandwidth-min",
        type=float,
        default=5.0,
        help="Uniform min bandwidth when --bandwidth-dist=uniform",
    )
    parser.add_argument(
        "--bandwidth-max",
        type=float,
        default=25.0,
        help="Uniform max bandwidth when --bandwidth-dist=uniform",
    )
    parser.add_argument(
        "--bandwidth-mu",
        type=float,
        default=2.3,
        help="Lognormal mu for bandwidth when --bandwidth-dist=lognormal",
    )
    parser.add_argument(
        "--bandwidth-sigma",
        type=float,
        default=0.3,
        help="Lognormal sigma for bandwidth when --bandwidth-dist=lognormal",
    )
    parser.add_argument("--block-bytes", type=int, default=1_000_000)
    parser.add_argument(
        "--drop-prob",
        type=float,
        default=0.0,
        help="Probability a node will not forward after receiving the block",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=0,
        help="If > 0, print a histogram/CDF over arrival times",
    )
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        num_nodes=args.nodes,
        degree=args.degree,
        topology=args.topology,
        rewire_prob=args.rewire_prob,
        scale_free_m=args.scale_free_m,
        latency_dist=args.latency_dist,
        latency_min=args.latency_min,
        latency_max=args.latency_max,
        latency_mu=args.latency_mu,
        latency_sigma=args.latency_sigma,
        bandwidth_dist=args.bandwidth_dist,
        bandwidth_mbps=args.bandwidth_mbps,
        bandwidth_min=args.bandwidth_min,
        bandwidth_max=args.bandwidth_max,
        bandwidth_mu=args.bandwidth_mu,
        bandwidth_sigma=args.bandwidth_sigma,
        block_size_bytes=args.block_bytes,
        drop_prob=args.drop_prob,
        source=args.source,
    )

    aggregate = run_experiments(args.protocol, args.runs, config, args.seed)

    if args.runs == 1:
        print(format_run_result(aggregate.runs[0]))
        if args.hist_bins > 0:
            print(format_distribution(aggregate.runs[0].arrival_times, args.hist_bins))
    else:
        print(format_aggregate(aggregate))


if __name__ == "__main__":
    main()
