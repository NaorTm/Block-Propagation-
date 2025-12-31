from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import SimulationConfig


@dataclass(frozen=True)
class PathStats:
    mean_stretch: float
    max_stretch: float
    mean_slack: float
    max_slack: float


@dataclass(frozen=True)
class MacroMetrics:
    competing_block_prob_t90: float
    expected_competing_blocks_t100: float
    prob_competing_blocks_ge1_t100: float
    prob_competing_blocks_ge2_t100: float
    security_margin_t50: float


@dataclass(frozen=True)
class MacroSimResult:
    orphan_rate: float
    mean_competing_blocks: float
    prob_competing_blocks_ge2: float


@dataclass(frozen=True)
class BottleneckValidation:
    precision: float
    recall: float
    f1: float


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


def arrival_histogram(arrival_times: Sequence[float], bins: int) -> List[Tuple[float, float, int, float]]:
    finite_times = sorted(t for t in arrival_times if math.isfinite(t))
    if not finite_times:
        return []
    if bins <= 0:
        raise ValueError("bins must be positive")

    t_min = finite_times[0]
    t_max = finite_times[-1]
    if t_max == t_min:
        return [(t_min, t_max, len(finite_times), 1.0)]

    width = (t_max - t_min) / bins
    counts = [0] * bins
    for t in finite_times:
        idx = min(int((t - t_min) / width), bins - 1)
        counts[idx] += 1

    total = len(finite_times)
    cumulative = 0
    results = []
    for i, count in enumerate(counts):
        start = t_min + i * width
        end = start + width
        cumulative += count
        cdf = cumulative / total
        results.append((start, end, count, cdf))
    return results


def format_histogram(arrival_times: Sequence[float], bins: int) -> str:
    rows = arrival_histogram(arrival_times, bins)
    if not rows:
        return "No arrivals to summarize."
    if len(rows) == 1 and rows[0][0] == rows[0][1]:
        return f"All arrivals at {rows[0][0]:.3f}s."

    lines = [f"Arrival time histogram (bins={bins})"]
    for start, end, count, cdf in rows:
        lines.append(f"  [{start:.3f}, {end:.3f}): count={count} cdf={cdf:.3f}")
    return "\n".join(lines)


def macro_metrics(
    arrival_times: Sequence[float], block_interval: float
) -> MacroMetrics:
    if block_interval <= 0:
        raise ValueError("block_interval must be positive")
    t50 = threshold_time(arrival_times, 0.5)
    t90 = threshold_time(arrival_times, 0.9)
    t100 = threshold_time(arrival_times, 1.0)

    def competing_prob(time_value: float) -> float:
        if not math.isfinite(time_value):
            return 1.0
        return 1.0 - math.exp(-time_value / block_interval)

    lambda_100 = math.inf if not math.isfinite(t100) else t100 / block_interval
    prob_ge1 = 1.0 if lambda_100 == math.inf else 1.0 - math.exp(-lambda_100)
    prob_ge2 = (
        1.0
        if lambda_100 == math.inf
        else 1.0 - math.exp(-lambda_100) * (1.0 + lambda_100)
    )

    return MacroMetrics(
        competing_block_prob_t90=competing_prob(t90),
        expected_competing_blocks_t100=lambda_100,
        prob_competing_blocks_ge1_t100=prob_ge1,
        prob_competing_blocks_ge2_t100=prob_ge2,
        security_margin_t50=1.0 - competing_prob(t50),
    )


def simulate_orphan_rate(
    arrival_times: Sequence[float],
    block_interval: float,
    trials: int,
    rng: random.Random,
) -> MacroSimResult:
    if block_interval <= 0:
        raise ValueError("block_interval must be positive")
    if trials <= 0:
        raise ValueError("trials must be positive")
    node_count = len(arrival_times)
    if node_count == 0:
        return MacroSimResult(0.0, 0.0, 0.0)

    per_node_mean = block_interval * node_count
    orphan_events = 0
    total_competing = 0
    ge2 = 0

    for _ in range(trials):
        competing = 0
        for arrival in arrival_times:
            if not math.isfinite(arrival):
                arrival = block_interval * 10
            mining_time = rng.expovariate(1.0 / per_node_mean)
            if mining_time < arrival:
                competing += 1
        if competing > 0:
            orphan_events += 1
        if competing >= 2:
            ge2 += 1
        total_competing += competing

    return MacroSimResult(
        orphan_rate=orphan_events / trials,
        mean_competing_blocks=total_competing / trials,
        prob_competing_blocks_ge2=ge2 / trials,
    )


def top_k_nodes_by_messages(per_node_messages: Sequence[int], k: int) -> List[Tuple[int, int]]:
    ranked = sorted(enumerate(per_node_messages), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def top_k_edges_by_messages(
    per_edge_messages: Dict[Tuple[int, int], int], k: int
) -> List[Tuple[Tuple[int, int], int]]:
    ranked = sorted(per_edge_messages.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def detect_bottlenecks(
    arrival_times: Sequence[float],
    per_node_messages: Sequence[int],
    top_fraction: float,
) -> List[Tuple[int, float]]:
    if top_fraction <= 0:
        return []
    finite = [(idx, t) for idx, t in enumerate(arrival_times) if math.isfinite(t)]
    if not finite:
        return []
    finite.sort(key=lambda x: x[1], reverse=True)
    cutoff = max(1, int(round(len(finite) * top_fraction)))
    slow_nodes = finite[:cutoff]
    max_messages = max(per_node_messages) if per_node_messages else 1

    scored = []
    for node, time_value in slow_nodes:
        load = per_node_messages[node] / max_messages if max_messages else 0.0
        score = 0.7 * (time_value / slow_nodes[0][1]) + 0.3 * load
        scored.append((node, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def detect_bottleneck_edges(
    per_edge_messages: Dict[Tuple[int, int], int],
    top_fraction: float,
) -> List[Tuple[Tuple[int, int], int]]:
    if top_fraction <= 0:
        return []
    ranked = sorted(per_edge_messages.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return []
    cutoff = max(1, int(round(len(ranked) * top_fraction)))
    return ranked[:cutoff]


def validate_bottlenecks(
    detected: Sequence[int], injected: Sequence[int]
) -> BottleneckValidation:
    detected_set = set(detected)
    injected_set = set(injected)
    if not detected_set and not injected_set:
        return BottleneckValidation(precision=1.0, recall=1.0, f1=1.0)
    if not detected_set:
        return BottleneckValidation(precision=0.0, recall=0.0, f1=0.0)
    if not injected_set:
        return BottleneckValidation(precision=0.0, recall=0.0, f1=0.0)

    true_pos = len(detected_set & injected_set)
    precision = true_pos / len(detected_set)
    recall = true_pos / len(injected_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return BottleneckValidation(precision=precision, recall=recall, f1=f1)


def edge_latency(latencies: Dict[Tuple[int, int], float], u: int, v: int) -> float:
    key = (u, v) if u < v else (v, u)
    return latencies[key]


def edge_bandwidth(bandwidths: Dict[Tuple[int, int], float], u: int, v: int) -> float:
    key = (u, v) if u < v else (v, u)
    return bandwidths[key]


def shortest_path_times(
    adjacency: Sequence[Iterable[int]],
    latencies: Dict[Tuple[int, int], float],
    bandwidths: Dict[Tuple[int, int], float],
    config: SimulationConfig,
    source: int,
    overlay_adjacency: Sequence[Iterable[int]] | None = None,
    overlay_latencies: Dict[Tuple[int, int], float] | None = None,
    overlay_bandwidths: Dict[Tuple[int, int], float] | None = None,
) -> List[float]:
    distances = [math.inf] * config.num_nodes
    distances[source] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, source)]

    while pq:
        time, node = heapq.heappop(pq)
        if time > distances[node]:
            continue
        base_neighbors = set(adjacency[node])
        overlay_neighbors = set(overlay_adjacency[node]) if overlay_adjacency else set()
        for neighbor in base_neighbors | overlay_neighbors:
            latency = edge_latency(latencies, node, neighbor) if neighbor in base_neighbors else None
            bandwidth = (
                edge_bandwidth(bandwidths, node, neighbor) if neighbor in base_neighbors else None
            )
            if overlay_adjacency and neighbor in overlay_neighbors:
                o_latency = edge_latency(overlay_latencies, node, neighbor)  # type: ignore[arg-type]
                o_bandwidth = edge_bandwidth(overlay_bandwidths, node, neighbor)  # type: ignore[arg-type]
                o_weight = o_latency + config.transmission_time(o_bandwidth)
                if latency is None:
                    weight = o_weight
                else:
                    weight = min(latency + config.transmission_time(bandwidth), o_weight)
            else:
                weight = latency + config.transmission_time(bandwidth)
            candidate = time + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(pq, (candidate, neighbor))
    return distances


def path_stats(
    arrival_times: Sequence[float],
    adjacency: Sequence[Iterable[int]],
    latencies: Dict[Tuple[int, int], float],
    bandwidths: Dict[Tuple[int, int], float],
    config: SimulationConfig,
    source: int,
    overlay_adjacency: Sequence[Iterable[int]] | None = None,
    overlay_latencies: Dict[Tuple[int, int], float] | None = None,
    overlay_bandwidths: Dict[Tuple[int, int], float] | None = None,
) -> PathStats:
    shortest = shortest_path_times(
        adjacency,
        latencies,
        bandwidths,
        config,
        source,
        overlay_adjacency=overlay_adjacency,
        overlay_latencies=overlay_latencies,
        overlay_bandwidths=overlay_bandwidths,
    )
    stretches = []
    slacks = []
    for arrival, baseline in zip(arrival_times, shortest):
        if not math.isfinite(arrival) or not math.isfinite(baseline):
            continue
        if baseline == 0:
            continue
        stretches.append(arrival / baseline)
        slacks.append(arrival - baseline)

    if not stretches:
        return PathStats(mean_stretch=math.inf, max_stretch=math.inf, mean_slack=math.inf, max_slack=math.inf)

    mean_stretch = sum(stretches) / len(stretches)
    max_stretch = max(stretches)
    mean_slack = sum(slacks) / len(slacks)
    max_slack = max(slacks)
    return PathStats(
        mean_stretch=mean_stretch,
        max_stretch=max_stretch,
        mean_slack=mean_slack,
        max_slack=max_slack,
    )
