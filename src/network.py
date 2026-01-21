from __future__ import annotations

import random
from typing import Dict, Iterable, List, NamedTuple, Sequence, Tuple

import networkx as nx

from .config import SimulationConfig


class NetworkData(NamedTuple):
    """Network structure and properties for simulation.
    
    Attributes:
        adjacency: List of neighbor sets for each node.
        latencies: Edge latency map in seconds.
        bandwidths: Edge bandwidth map in Mbps.
        bottleneck_nodes: Set of nodes with degraded performance.
        relay_nodes: Set of nodes in the relay network.
        overlay: List of overlay neighbor sets for relay nodes.
        overlay_latencies: Edge latency map for overlay network.
        overlay_bandwidths: Edge bandwidth map for overlay network.
    """
    adjacency: List[set]
    latencies: Dict[Tuple[int, int], float]
    bandwidths: Dict[Tuple[int, int], float]
    bottleneck_nodes: set[int]
    relay_nodes: set[int]
    overlay: List[set]
    overlay_latencies: Dict[Tuple[int, int], float]
    overlay_bandwidths: Dict[Tuple[int, int], float]


def generate_random_regular_graph(
    num_nodes: int, degree: int, rng: random.Random, max_attempts: int = 200
) -> List[set]:
    """Generate a random regular graph."""

    if num_nodes <= 1:
        raise ValueError("Number of nodes must be at least 2")
    if degree >= num_nodes:
        raise ValueError("Degree must be less than the number of nodes")
    if (num_nodes * degree) % 2 != 0:
        raise ValueError("num_nodes * degree must be even")

    try:
        seed = rng.randint(0, 2**32 - 1)
        graph = nx.random_regular_graph(degree, num_nodes, seed=seed)
        return adjacency_from_graph(graph)
    except nx.NetworkXError as exc:
        raise RuntimeError(
            f"Failed to generate a random regular graph: {exc}"
        ) from exc


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
    if config.topology == "star":
        graph = nx.star_graph(config.num_nodes - 1)
        return adjacency_from_graph(graph)
    if config.topology == "line":
        graph = nx.path_graph(config.num_nodes)
        return adjacency_from_graph(graph)
    raise ValueError(f"Unknown topology: {config.topology}")


def select_bottleneck_nodes(
    num_nodes: int, fraction: float, rng: random.Random
) -> set[int]:
    if fraction <= 0:
        return set()
    count = int(round(num_nodes * fraction))
    if count <= 0:
        return set()
    count = min(count, num_nodes)
    return set(rng.sample(range(num_nodes), count))


def select_relay_nodes(num_nodes: int, fraction: float, rng: random.Random) -> set[int]:
    if fraction <= 0:
        return set()
    count = int(round(num_nodes * fraction))
    if count <= 0:
        return set()
    count = min(count, num_nodes)
    return set(rng.sample(range(num_nodes), count))


def build_relay_overlay(
    num_nodes: int,
    relay_nodes: set[int],
    rng: random.Random,
    degree: int,
    prob: float,
) -> List[set]:
    overlay = [set() for _ in range(num_nodes)]
    relay_list = list(relay_nodes)
    if not relay_list:
        return overlay

    if degree > 0:
        for node in relay_list:
            candidates = [n for n in relay_list if n != node and n not in overlay[node]]
            if not candidates:
                continue
            count = min(degree, len(candidates))
            chosen = rng.sample(candidates, count)
            for other in chosen:
                overlay[node].add(other)
                overlay[other].add(node)
        return overlay

    if prob > 0:
        for i, node in enumerate(relay_list):
            for other in relay_list[i + 1 :]:
                if rng.random() <= prob:
                    overlay[node].add(other)
                    overlay[other].add(node)
    return overlay


def assign_latencies(
    adjacency: Sequence[Iterable[int]],
    config: SimulationConfig,
    rng: random.Random,
    bottleneck_nodes: set[int],
    relay_nodes: set[int],
) -> Dict[Tuple[int, int], float]:
    """Assign a symmetric latency to each edge."""

    latencies: Dict[Tuple[int, int], float] = {}
    for u, neighbors in enumerate(adjacency):
        for v in neighbors:
            if u < v:
                if config.latency_dist == "uniform":
                    latency = rng.uniform(config.latency_min, config.latency_max)
                elif config.latency_dist == "lognormal":
                    latency = rng.lognormvariate(config.latency_mu, config.latency_sigma)
                else:
                    raise ValueError(f"Unknown latency distribution: {config.latency_dist}")
                if u in bottleneck_nodes or v in bottleneck_nodes:
                    latency *= config.bottleneck_latency_mult
                if u in relay_nodes and v in relay_nodes:
                    latency *= config.relay_latency_mult
                latencies[(u, v)] = latency
    return latencies


def assign_bandwidths(
    adjacency: Sequence[Iterable[int]],
    config: SimulationConfig,
    rng: random.Random,
    bottleneck_nodes: set[int],
    relay_nodes: set[int],
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
                if u in bottleneck_nodes or v in bottleneck_nodes:
                    bandwidth *= config.bottleneck_bandwidth_mult
                if u in relay_nodes and v in relay_nodes:
                    bandwidth *= config.relay_bandwidth_mult
                bandwidths[(u, v)] = bandwidth
    return bandwidths


def build_network(
    config: SimulationConfig, rng: random.Random
) -> NetworkData:
    """Build a complete network with all properties for simulation.
    
    Args:
        config: Simulation configuration parameters.
        rng: Random number generator for reproducibility.
    
    Returns:
        NetworkData containing all network structures and properties.
    
    Example:
        >>> config = SimulationConfig(num_nodes=100, degree=8)
        >>> network = build_network(config, random.Random(42))
        >>> print(len(network.adjacency))
        100
    """
    adjacency = generate_graph(config, rng)
    bottleneck_nodes = select_bottleneck_nodes(
        config.num_nodes, config.bottleneck_fraction, rng
    )
    relay_nodes = select_relay_nodes(config.num_nodes, config.relay_fraction, rng)
    overlay = build_relay_overlay(
        config.num_nodes,
        relay_nodes,
        rng,
        config.relay_overlay_degree,
        config.relay_overlay_prob,
    )
    latencies = assign_latencies(adjacency, config, rng, bottleneck_nodes, relay_nodes)
    bandwidths = assign_bandwidths(adjacency, config, rng, bottleneck_nodes, relay_nodes)
    overlay_latencies = assign_latencies(overlay, config, rng, bottleneck_nodes, relay_nodes)
    overlay_bandwidths = assign_bandwidths(overlay, config, rng, bottleneck_nodes, relay_nodes)
    return NetworkData(
        adjacency=adjacency,
        latencies=latencies,
        bandwidths=bandwidths,
        bottleneck_nodes=bottleneck_nodes,
        relay_nodes=relay_nodes,
        overlay=overlay,
        overlay_latencies=overlay_latencies,
        overlay_bandwidths=overlay_bandwidths,
    )
