from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from .config import SimulationConfig
from .metrics import path_stats, threshold_time
from .network import build_network
from .results import RunResult


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _edge_latency(
    latencies: Dict[Tuple[int, int], float], u: int, v: int
) -> float:
    return latencies[_edge_key(u, v)]


def _edge_bandwidth(
    bandwidths: Dict[Tuple[int, int], float], u: int, v: int
) -> float:
    return bandwidths[_edge_key(u, v)]


def _select_neighbors(adjacency: List[set], node: int, rng: random.Random, fanout: int) -> List[int]:
    neighbors = list(adjacency[node])
    if fanout <= 0 or fanout >= len(neighbors):
        return neighbors
    return rng.sample(neighbors, fanout)


def _neighbor_map(
    adjacency: List[set], overlay: List[set], node: int, relay_nodes: set[int]
) -> Dict[int, bool]:
    neighbors: Dict[int, bool] = {n: False for n in adjacency[node]}
    if relay_nodes and node in relay_nodes:
        for n in overlay[node]:
            neighbors[n] = True
    return neighbors


def _build_node_status(
    config: SimulationConfig, rng: random.Random
) -> tuple[List[float], List[bool]]:
    failure_times = [math.inf] * config.num_nodes
    delayed = [False] * config.num_nodes

    for node in range(config.num_nodes):
        if rng.random() <= config.churn_prob:
            if config.churn_time_max <= config.churn_time_min:
                failure_times[node] = config.churn_time_min
            else:
                failure_times[node] = rng.uniform(config.churn_time_min, config.churn_time_max)
        if rng.random() <= config.delay_prob:
            delayed[node] = True
    return failure_times, delayed


def _can_send(time_value: float, failure_times: List[float], node: int) -> bool:
    return time_value < failure_times[node]


def _adjust_latency(
    latency: float, delayed: List[bool], node: int, config: SimulationConfig
) -> float:
    if not delayed[node]:
        return latency
    return latency * config.delay_latency_mult


def _adjust_bandwidth(
    bandwidth: float, delayed: List[bool], node: int, config: SimulationConfig
) -> float:
    if not delayed[node]:
        return bandwidth
    return bandwidth * config.delay_bandwidth_mult


def simulate_naive_flooding(
    config: SimulationConfig,
    rng: random.Random,
    include_path_stats: bool = False,
    network: tuple | None = None,
) -> RunResult:
    if network is None:
        (
            adjacency,
            latencies,
            bandwidths,
            bottleneck_nodes,
            relay_nodes,
            overlay,
            overlay_latencies,
            overlay_bandwidths,
        ) = build_network(config, rng)
    else:
        (
            adjacency,
            latencies,
            bandwidths,
            bottleneck_nodes,
            relay_nodes,
            overlay,
            overlay_latencies,
            overlay_bandwidths,
        ) = network

    arrival_times = [math.inf] * config.num_nodes
    arrival_times[config.source] = 0.0
    failure_times, delayed = _build_node_status(config, rng)

    will_forward = [
        True if node == config.source else rng.random() >= config.drop_prob
        for node in range(config.num_nodes)
    ]

    pq: List[Tuple[float, str, int, int]] = []
    messages = defaultdict(int)
    per_node_messages = [0] * config.num_nodes
    per_edge_messages: Dict[Tuple[int, int], int] = defaultdict(int)

    if will_forward[config.source] and _can_send(0.0, failure_times, config.source):
        for neighbor, is_overlay in _neighbor_map(
            adjacency, overlay, config.source, relay_nodes
        ).items():
            latency_map = overlay_latencies if is_overlay else latencies
            bandwidth_map = overlay_bandwidths if is_overlay else bandwidths
            latency = _adjust_latency(
                _edge_latency(latency_map, config.source, neighbor),
                delayed,
                config.source,
                config,
            )
            bandwidth = _adjust_bandwidth(
                _edge_bandwidth(bandwidth_map, config.source, neighbor),
                delayed,
                config.source,
                config,
            )
            travel_time = latency + config.transmission_time(bandwidth)
            heapq.heappush(pq, (travel_time, "block", config.source, neighbor))
            messages["block"] += 1
            per_node_messages[config.source] += 1
            per_edge_messages[_edge_key(config.source, neighbor)] += 1

    while pq:
        time, event_type, src, dst = heapq.heappop(pq)
        if event_type != "block":
            continue
        if math.isfinite(arrival_times[dst]):
            continue

        arrival_times[dst] = time
        if not will_forward[dst] or not _can_send(time, failure_times, dst):
            continue
        for neighbor, is_overlay in _neighbor_map(
            adjacency, overlay, dst, relay_nodes
        ).items():
            latency_map = overlay_latencies if is_overlay else latencies
            bandwidth_map = overlay_bandwidths if is_overlay else bandwidths
            latency = _adjust_latency(
                _edge_latency(latency_map, dst, neighbor), delayed, dst, config
            )
            bandwidth = _adjust_bandwidth(
                _edge_bandwidth(bandwidth_map, dst, neighbor), delayed, dst, config
            )
            travel_time = latency + config.transmission_time(bandwidth)
            heapq.heappush(pq, (time + travel_time, "block", dst, neighbor))
            messages["block"] += 1
            per_node_messages[dst] += 1
            per_edge_messages[_edge_key(dst, neighbor)] += 1

    t50 = threshold_time(arrival_times, 0.5)
    t90 = threshold_time(arrival_times, 0.9)
    t100 = threshold_time(arrival_times, 1.0)
    stats = (
        path_stats(
            arrival_times,
            adjacency,
            latencies,
            bandwidths,
            config,
            config.source,
            overlay_adjacency=overlay,
            overlay_latencies=overlay_latencies,
            overlay_bandwidths=overlay_bandwidths,
        )
        if include_path_stats
        else None
    )

    return RunResult(
        protocol="naive",
        config=config,
        arrival_times=arrival_times,
        t50=t50,
        t90=t90,
        t100=t100,
        messages=dict(messages),
        per_node_messages=per_node_messages,
        per_edge_messages=dict(per_edge_messages),
        bottleneck_nodes=bottleneck_nodes,
        relay_nodes=relay_nodes,
        path_stats=stats,
    )


def simulate_two_phase(
    config: SimulationConfig,
    rng: random.Random,
    include_path_stats: bool = False,
    compact_blocks: bool = False,
    network: tuple | None = None,
) -> RunResult:
    if network is None:
        (
            adjacency,
            latencies,
            bandwidths,
            bottleneck_nodes,
            relay_nodes,
            overlay,
            overlay_latencies,
            overlay_bandwidths,
        ) = build_network(config, rng)
    else:
        (
            adjacency,
            latencies,
            bandwidths,
            bottleneck_nodes,
            relay_nodes,
            overlay,
            overlay_latencies,
            overlay_bandwidths,
        ) = network

    knows_block = [False] * config.num_nodes
    has_full_block = [False] * config.num_nodes
    arrival_times = [math.inf] * config.num_nodes
    failure_times, delayed = _build_node_status(config, rng)

    source = config.source
    overlap_mean = (
        config.mempool_sync_prob
        if config.mempool_overlap_mean is None
        else config.mempool_overlap_mean
    )
    overlap_ratio = [
        max(0.0, min(1.0, rng.gauss(overlap_mean, config.mempool_overlap_std)))
        for _ in range(config.num_nodes)
    ]
    overlap_ratio[source] = 1.0
    will_forward = [
        True if node == source else rng.random() >= config.drop_prob
        for node in range(config.num_nodes)
    ]
    knows_block[source] = True
    has_full_block[source] = True
    arrival_times[source] = 0.0

    pq: List[Tuple[float, str, int, int]] = []
    messages = defaultdict(int)
    per_node_messages = [0] * config.num_nodes
    per_edge_messages: Dict[Tuple[int, int], int] = defaultdict(int)

    def enqueue(event_time: float, event_type: str, src: int, dst: int) -> None:
        heapq.heappush(pq, (event_time, event_type, src, dst))
        messages[event_type] += 1
        per_node_messages[src] += 1
        per_edge_messages[_edge_key(src, dst)] += 1

    if will_forward[source] and _can_send(0.0, failure_times, source):
        for neighbor, is_overlay in _neighbor_map(
            adjacency, overlay, source, relay_nodes
        ).items():
            latency_map = overlay_latencies if is_overlay else latencies
            latency = _adjust_latency(
                _edge_latency(latency_map, source, neighbor), delayed, source, config
            )
            enqueue(latency, "announce", source, neighbor)

    while pq:
        time, event_type, src, dst = heapq.heappop(pq)

        if event_type == "announce":
            if not knows_block[dst]:
                knows_block[dst] = True
                latency_map = overlay_latencies if src in relay_nodes and dst in relay_nodes and dst in overlay[src] else latencies
                back_latency = _adjust_latency(
                    _edge_latency(latency_map, src, dst), delayed, dst, config
                )
                request_time = time + back_latency
                if _can_send(request_time, failure_times, dst):
                    enqueue(request_time, "request", dst, src)

        elif event_type == "request":
            if (
                has_full_block[dst]
                and will_forward[dst]
                and _can_send(time, failure_times, dst)
            ):
                latency_map = (
                    overlay_latencies
                    if src in relay_nodes and dst in relay_nodes and dst in overlay[src]
                    else latencies
                )
                bandwidth_map = (
                    overlay_bandwidths
                    if src in relay_nodes and dst in relay_nodes and dst in overlay[src]
                    else bandwidths
                )
                delivery_latency = _adjust_latency(
                    _edge_latency(latency_map, src, dst), delayed, dst, config
                )
                bandwidth = _adjust_bandwidth(
                    _edge_bandwidth(bandwidth_map, src, dst), delayed, dst, config
                )
                if compact_blocks:
                    compact_time = config.transmission_time(
                        bandwidth, config.compact_block_bytes
                    )
                    arrival = time + delivery_latency + compact_time
                    success_prob = overlap_ratio[dst] * config.compact_success_prob
                    if rng.random() < success_prob:
                        enqueue(arrival, "block_compact_ok", dst, src)
                    else:
                        enqueue(arrival, "block_compact_fail", dst, src)
                else:
                    block_time = config.transmission_time(bandwidth)
                    arrival = time + delivery_latency + block_time
                    enqueue(arrival, "block", dst, src)

        elif event_type in ("block", "block_full", "block_compact_ok", "missing_tx"):
            if has_full_block[dst]:
                continue
            has_full_block[dst] = True
            arrival_times[dst] = time

            if will_forward[dst] and _can_send(time, failure_times, dst):
                for neighbor, is_overlay in _neighbor_map(
                    adjacency, overlay, dst, relay_nodes
                ).items():
                    if not knows_block[neighbor]:
                        latency_map = overlay_latencies if is_overlay else latencies
                        latency = _adjust_latency(
                            _edge_latency(latency_map, dst, neighbor), delayed, dst, config
                        )
                        enqueue(time + latency, "announce", dst, neighbor)
        elif event_type == "block_compact_fail":
            latency_map = (
                overlay_latencies
                if src in relay_nodes and dst in relay_nodes and dst in overlay[src]
                else latencies
            )
            bandwidth_map = (
                overlay_bandwidths
                if src in relay_nodes and dst in relay_nodes and dst in overlay[src]
                else bandwidths
            )
            latency = _adjust_latency(
                _edge_latency(latency_map, src, dst), delayed, dst, config
            )
            bandwidth = _adjust_bandwidth(
                _edge_bandwidth(bandwidth_map, src, dst), delayed, src, config
            )
            effective_overlap = overlap_ratio[dst]
            missing_bytes = max(
                int((1.0 - effective_overlap) * config.block_size_bytes),
                config.missing_tx_bytes_min,
            )
            missing_send_time = time + 2 * latency
            missing_time = missing_send_time + config.transmission_time(
                bandwidth, missing_bytes
            )
            if _can_send(missing_send_time, failure_times, src):
                enqueue(missing_time, "missing_tx", src, dst)
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    t50 = threshold_time(arrival_times, 0.5)
    t90 = threshold_time(arrival_times, 0.9)
    t100 = threshold_time(arrival_times, 1.0)
    stats = (
        path_stats(
            arrival_times,
            adjacency,
            latencies,
            bandwidths,
            config,
            config.source,
            overlay_adjacency=overlay,
            overlay_latencies=overlay_latencies,
            overlay_bandwidths=overlay_bandwidths,
        )
        if include_path_stats
        else None
    )

    return RunResult(
        protocol="two-phase",
        config=config,
        arrival_times=arrival_times,
        t50=t50,
        t90=t90,
        t100=t100,
        messages=dict(messages),
        per_node_messages=per_node_messages,
        per_edge_messages=dict(per_edge_messages),
        bottleneck_nodes=bottleneck_nodes,
        relay_nodes=relay_nodes,
        path_stats=stats,
    )


def simulate_push(
    config: SimulationConfig,
    rng: random.Random,
    include_path_stats: bool = False,
    network: tuple | None = None,
) -> RunResult:
    if network is None:
        (
            adjacency,
            latencies,
            bandwidths,
            bottleneck_nodes,
            relay_nodes,
            overlay,
            overlay_latencies,
            overlay_bandwidths,
        ) = build_network(config, rng)
    else:
        (
            adjacency,
            latencies,
            bandwidths,
            bottleneck_nodes,
            relay_nodes,
            overlay,
            overlay_latencies,
            overlay_bandwidths,
        ) = network

    arrival_times = [math.inf] * config.num_nodes
    arrival_times[config.source] = 0.0
    failure_times, delayed = _build_node_status(config, rng)

    will_forward = [
        True if node == config.source else rng.random() >= config.drop_prob
        for node in range(config.num_nodes)
    ]

    pq: List[Tuple[float, str, int, int]] = []
    messages = defaultdict(int)
    per_node_messages = [0] * config.num_nodes
    per_edge_messages: Dict[Tuple[int, int], int] = defaultdict(int)

    if will_forward[config.source] and _can_send(0.0, failure_times, config.source):
        neighbors = _select_neighbors(adjacency, config.source, rng, config.gossip_fanout)
        overlay_neighbors = list(
            _neighbor_map(adjacency, overlay, config.source, relay_nodes).keys()
        )
        for neighbor in set(neighbors + overlay_neighbors):
            is_overlay = config.source in relay_nodes and neighbor in overlay[config.source]
            latency_map = overlay_latencies if is_overlay else latencies
            bandwidth_map = overlay_bandwidths if is_overlay else bandwidths
            latency = _adjust_latency(
                _edge_latency(latency_map, config.source, neighbor),
                delayed,
                config.source,
                config,
            )
            bandwidth = _adjust_bandwidth(
                _edge_bandwidth(bandwidth_map, config.source, neighbor),
                delayed,
                config.source,
                config,
            )
            travel_time = latency + config.transmission_time(bandwidth)
            heapq.heappush(pq, (travel_time, "block", config.source, neighbor))
            messages["block"] += 1
            per_node_messages[config.source] += 1
            per_edge_messages[_edge_key(config.source, neighbor)] += 1

    while pq:
        time, event_type, src, dst = heapq.heappop(pq)
        if event_type != "block":
            continue
        if math.isfinite(arrival_times[dst]):
            continue
        arrival_times[dst] = time
        if not will_forward[dst] or not _can_send(time, failure_times, dst):
            continue
        neighbors = _select_neighbors(adjacency, dst, rng, config.gossip_fanout)
        overlay_neighbors = list(_neighbor_map(adjacency, overlay, dst, relay_nodes).keys())
        for neighbor in set(neighbors + overlay_neighbors):
            is_overlay = dst in relay_nodes and neighbor in overlay[dst]
            latency_map = overlay_latencies if is_overlay else latencies
            bandwidth_map = overlay_bandwidths if is_overlay else bandwidths
            latency = _adjust_latency(
                _edge_latency(latency_map, dst, neighbor), delayed, dst, config
            )
            bandwidth = _adjust_bandwidth(
                _edge_bandwidth(bandwidth_map, dst, neighbor), delayed, dst, config
            )
            travel_time = latency + config.transmission_time(bandwidth)
            heapq.heappush(pq, (time + travel_time, "block", dst, neighbor))
            messages["block"] += 1
            per_node_messages[dst] += 1
            per_edge_messages[_edge_key(dst, neighbor)] += 1

    t50 = threshold_time(arrival_times, 0.5)
    t90 = threshold_time(arrival_times, 0.9)
    t100 = threshold_time(arrival_times, 1.0)
    stats = (
        path_stats(
            arrival_times,
            adjacency,
            latencies,
            bandwidths,
            config,
            config.source,
            overlay_adjacency=overlay,
            overlay_latencies=overlay_latencies,
            overlay_bandwidths=overlay_bandwidths,
        )
        if include_path_stats
        else None
    )

    return RunResult(
        protocol="push",
        config=config,
        arrival_times=arrival_times,
        t50=t50,
        t90=t90,
        t100=t100,
        messages=dict(messages),
        per_node_messages=per_node_messages,
        per_edge_messages=dict(per_edge_messages),
        bottleneck_nodes=bottleneck_nodes,
        relay_nodes=relay_nodes,
        path_stats=stats,
    )


def simulate_pull(
    config: SimulationConfig,
    rng: random.Random,
    include_path_stats: bool = False,
    network: tuple | None = None,
) -> RunResult:
    if network is None:
        (
            adjacency,
            latencies,
            bandwidths,
            bottleneck_nodes,
            relay_nodes,
            overlay,
            overlay_latencies,
            overlay_bandwidths,
        ) = build_network(config, rng)
    else:
        (
            adjacency,
            latencies,
            bandwidths,
            bottleneck_nodes,
            relay_nodes,
            overlay,
            overlay_latencies,
            overlay_bandwidths,
        ) = network

    arrival_times = [math.inf] * config.num_nodes
    arrival_times[config.source] = 0.0
    has_full_block = [False] * config.num_nodes
    has_full_block[config.source] = True
    failure_times, delayed = _build_node_status(config, rng)

    will_forward = [
        True if node == config.source else rng.random() >= config.drop_prob
        for node in range(config.num_nodes)
    ]

    pq: List[Tuple[float, str, int, int]] = []
    messages = defaultdict(int)
    per_node_messages = [0] * config.num_nodes
    per_edge_messages: Dict[Tuple[int, int], int] = defaultdict(int)

    for node in range(config.num_nodes):
        if node == config.source:
            continue
        initial_time = rng.uniform(0.0, config.pull_interval)
        heapq.heappush(pq, (initial_time, "pull", node, node))

    while pq:
        time, event_type, src, dst = heapq.heappop(pq)
        if time > config.max_time:
            continue
        if event_type == "pull":
            if has_full_block[src]:
                continue
            for neighbor in _select_neighbors(adjacency, src, rng, config.pull_fanout):
                latency = _adjust_latency(
                    _edge_latency(latencies, src, neighbor), delayed, src, config
                )
                if _can_send(time, failure_times, src):
                    heapq.heappush(pq, (time + latency, "pull_req", src, neighbor))
                    messages["pull_req"] += 1
                    per_node_messages[src] += 1
                    per_edge_messages[_edge_key(src, neighbor)] += 1
            next_time = time + config.pull_interval
            if next_time <= config.max_time:
                heapq.heappush(pq, (next_time, "pull", src, src))
        elif event_type == "pull_req":
            if (
                has_full_block[dst]
                and will_forward[dst]
                and _can_send(time, failure_times, dst)
            ):
                latency_map = (
                    overlay_latencies
                    if src in relay_nodes and dst in relay_nodes and dst in overlay[src]
                    else latencies
                )
                bandwidth_map = (
                    overlay_bandwidths
                    if src in relay_nodes and dst in relay_nodes and dst in overlay[src]
                    else bandwidths
                )
                latency = _adjust_latency(
                    _edge_latency(latency_map, src, dst), delayed, dst, config
                )
                bandwidth = _adjust_bandwidth(
                    _edge_bandwidth(bandwidth_map, src, dst), delayed, dst, config
                )
                travel_time = latency + config.transmission_time(bandwidth)
                heapq.heappush(pq, (time + travel_time, "block", dst, src))
                messages["block"] += 1
                per_node_messages[dst] += 1
                per_edge_messages[_edge_key(dst, src)] += 1
        elif event_type == "block":
            if has_full_block[dst]:
                continue
            has_full_block[dst] = True
            arrival_times[dst] = time
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    t50 = threshold_time(arrival_times, 0.5)
    t90 = threshold_time(arrival_times, 0.9)
    t100 = threshold_time(arrival_times, 1.0)
    stats = (
        path_stats(
            arrival_times,
            adjacency,
            latencies,
            bandwidths,
            config,
            config.source,
            overlay_adjacency=overlay,
            overlay_latencies=overlay_latencies,
            overlay_bandwidths=overlay_bandwidths,
        )
        if include_path_stats
        else None
    )

    return RunResult(
        protocol="pull",
        config=config,
        arrival_times=arrival_times,
        t50=t50,
        t90=t90,
        t100=t100,
        messages=dict(messages),
        per_node_messages=per_node_messages,
        per_edge_messages=dict(per_edge_messages),
        bottleneck_nodes=bottleneck_nodes,
        relay_nodes=relay_nodes,
        path_stats=stats,
    )


def simulate_push_pull(
    config: SimulationConfig,
    rng: random.Random,
    include_path_stats: bool = False,
    network: tuple | None = None,
) -> RunResult:
    if network is None:
        (
            adjacency,
            latencies,
            bandwidths,
            bottleneck_nodes,
            relay_nodes,
            overlay,
            overlay_latencies,
            overlay_bandwidths,
        ) = build_network(config, rng)
    else:
        (
            adjacency,
            latencies,
            bandwidths,
            bottleneck_nodes,
            relay_nodes,
            overlay,
            overlay_latencies,
            overlay_bandwidths,
        ) = network

    arrival_times = [math.inf] * config.num_nodes
    arrival_times[config.source] = 0.0
    has_full_block = [False] * config.num_nodes
    has_full_block[config.source] = True
    failure_times, delayed = _build_node_status(config, rng)

    will_forward = [
        True if node == config.source else rng.random() >= config.drop_prob
        for node in range(config.num_nodes)
    ]

    pq: List[Tuple[float, str, int, int]] = []
    messages = defaultdict(int)
    per_node_messages = [0] * config.num_nodes
    per_edge_messages: Dict[Tuple[int, int], int] = defaultdict(int)

    def send_push(src: int, time: float) -> None:
        if not _can_send(time, failure_times, src):
            return
        neighbors = _select_neighbors(adjacency, src, rng, config.gossip_fanout)
        overlay_neighbors = list(_neighbor_map(adjacency, overlay, src, relay_nodes).keys())
        for neighbor in set(neighbors + overlay_neighbors):
            is_overlay = src in relay_nodes and neighbor in overlay[src]
            latency_map = overlay_latencies if is_overlay else latencies
            bandwidth_map = overlay_bandwidths if is_overlay else bandwidths
            latency = _adjust_latency(
                _edge_latency(latency_map, src, neighbor), delayed, src, config
            )
            bandwidth = _adjust_bandwidth(
                _edge_bandwidth(bandwidth_map, src, neighbor), delayed, src, config
            )
            travel_time = latency + config.transmission_time(bandwidth)
            heapq.heappush(pq, (time + travel_time, "block", src, neighbor))
            messages["block"] += 1
            per_node_messages[src] += 1
            per_edge_messages[_edge_key(src, neighbor)] += 1

    if will_forward[config.source]:
        send_push(config.source, 0.0)

    for node in range(config.num_nodes):
        if node == config.source:
            continue
        initial_time = rng.uniform(0.0, config.pull_interval)
        heapq.heappush(pq, (initial_time, "pull", node, node))

    while pq:
        time, event_type, src, dst = heapq.heappop(pq)
        if time > config.max_time:
            continue
        if event_type == "pull":
            if has_full_block[src]:
                continue
            for neighbor in _select_neighbors(adjacency, src, rng, config.pull_fanout):
                latency = _adjust_latency(
                    _edge_latency(latencies, src, neighbor), delayed, src, config
                )
                if _can_send(time, failure_times, src):
                    heapq.heappush(pq, (time + latency, "pull_req", src, neighbor))
                    messages["pull_req"] += 1
                    per_node_messages[src] += 1
                    per_edge_messages[_edge_key(src, neighbor)] += 1
            next_time = time + config.pull_interval
            if next_time <= config.max_time:
                heapq.heappush(pq, (next_time, "pull", src, src))
        elif event_type == "pull_req":
            if (
                has_full_block[dst]
                and will_forward[dst]
                and _can_send(time, failure_times, dst)
            ):
                latency_map = (
                    overlay_latencies
                    if src in relay_nodes and dst in relay_nodes and dst in overlay[src]
                    else latencies
                )
                bandwidth_map = (
                    overlay_bandwidths
                    if src in relay_nodes and dst in relay_nodes and dst in overlay[src]
                    else bandwidths
                )
                latency = _adjust_latency(
                    _edge_latency(latency_map, src, dst), delayed, dst, config
                )
                bandwidth = _adjust_bandwidth(
                    _edge_bandwidth(bandwidth_map, src, dst), delayed, dst, config
                )
                travel_time = latency + config.transmission_time(bandwidth)
                heapq.heappush(pq, (time + travel_time, "block", dst, src))
                messages["block"] += 1
                per_node_messages[dst] += 1
                per_edge_messages[_edge_key(dst, src)] += 1
        elif event_type == "block":
            if has_full_block[dst]:
                continue
            has_full_block[dst] = True
            arrival_times[dst] = time
            if will_forward[dst] and _can_send(time, failure_times, dst):
                send_push(dst, time)
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    t50 = threshold_time(arrival_times, 0.5)
    t90 = threshold_time(arrival_times, 0.9)
    t100 = threshold_time(arrival_times, 1.0)
    stats = (
        path_stats(
            arrival_times,
            adjacency,
            latencies,
            bandwidths,
            config,
            config.source,
            overlay_adjacency=overlay,
            overlay_latencies=overlay_latencies,
            overlay_bandwidths=overlay_bandwidths,
        )
        if include_path_stats
        else None
    )

    return RunResult(
        protocol="push-pull",
        config=config,
        arrival_times=arrival_times,
        t50=t50,
        t90=t90,
        t100=t100,
        messages=dict(messages),
        per_node_messages=per_node_messages,
        per_edge_messages=dict(per_edge_messages),
        bottleneck_nodes=bottleneck_nodes,
        relay_nodes=relay_nodes,
        path_stats=stats,
    )
