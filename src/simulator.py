from __future__ import annotations

import random
import statistics
from typing import Dict, List

from .config import SimulationConfig
from .metrics import (
    detect_bottleneck_edges,
    detect_bottlenecks,
    format_histogram,
    macro_metrics,
    simulate_orphan_rate,
    top_k_edges_by_messages,
    top_k_nodes_by_messages,
    validate_bottlenecks,
)
from .protocols import (
    simulate_naive_flooding,
    simulate_pull,
    simulate_push,
    simulate_push_pull,
    simulate_two_phase,
)
from .results import AggregateResult, RunResult


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
    protocol: str,
    runs: int,
    config: SimulationConfig,
    seed: int | None,
    include_path_stats: bool = False,
) -> AggregateResult:
    rng = random.Random(seed)
    results: List[RunResult] = []

    for _ in range(runs):
        if protocol == "naive":
            result = simulate_naive_flooding(config, rng, include_path_stats=include_path_stats)
        elif protocol == "two-phase":
            result = simulate_two_phase(config, rng, include_path_stats=include_path_stats)
        elif protocol == "push":
            result = simulate_push(config, rng, include_path_stats=include_path_stats)
        elif protocol == "pull":
            result = simulate_pull(config, rng, include_path_stats=include_path_stats)
        elif protocol == "push-pull":
            result = simulate_push_pull(config, rng, include_path_stats=include_path_stats)
        elif protocol == "bitcoin-compact":
            result = simulate_two_phase(
                config, rng, include_path_stats=include_path_stats, compact_blocks=True
            )
        else:
            raise ValueError(
                "Protocol must be 'naive', 'two-phase', 'push', 'pull', 'push-pull', or 'bitcoin-compact'"
            )
        results.append(result)

    return summarize_runs(protocol, results)


def format_run_result(
    result: RunResult,
    show_overhead: bool = False,
    top_k: int = 5,
    detect_bottlenecks_flag: bool = False,
    bottleneck_fraction: float = 0.1,
    show_macro: bool = False,
    block_interval: float = 600.0,
    macro_sim_trials: int = 0,
    validate_bottlenecks_flag: bool = False,
    validate_bottleneck_edges_flag: bool = False,
) -> str:
    parts = [
        f"Protocol: {result.protocol}",
        f"T50: {result.t50:.3f}s, T90: {result.t90:.3f}s, T100: {result.t100:.3f}s",
        "Messages: " + ", ".join(f"{k}={v}" for k, v in sorted(result.messages.items())),
    ]
    if result.path_stats:
        parts.append(
            "PathStats: "
            f"mean_stretch={result.path_stats.mean_stretch:.3f}, "
            f"max_stretch={result.path_stats.max_stretch:.3f}, "
            f"mean_slack={result.path_stats.mean_slack:.3f}s, "
            f"max_slack={result.path_stats.max_slack:.3f}s"
        )
    if result.bottleneck_nodes:
        parts.append(f"BottleneckNodes: {len(result.bottleneck_nodes)}")
    if result.relay_nodes:
        parts.append(f"RelayNodes: {len(result.relay_nodes)}")
    if show_macro:
        metrics = macro_metrics(result.arrival_times, block_interval)
        parts.append(
            "MacroMetrics: "
            f"compete_p_t90={metrics.competing_block_prob_t90:.3f}, "
            f"lambda_t100={metrics.expected_competing_blocks_t100:.3f}, "
            f"p_ge1_t100={metrics.prob_competing_blocks_ge1_t100:.3f}, "
            f"p_ge2_t100={metrics.prob_competing_blocks_ge2_t100:.3f}, "
            f"security_margin_t50={metrics.security_margin_t50:.3f}"
        )
        if macro_sim_trials > 0:
            sim = simulate_orphan_rate(
                result.arrival_times,
                block_interval,
                macro_sim_trials,
                random.Random(0),
            )
            parts.append(
                "MacroSim: "
                f"orphan_rate={sim.orphan_rate:.3f}, "
                f"mean_competing={sim.mean_competing_blocks:.3f}, "
                f"p_ge2={sim.prob_competing_blocks_ge2:.3f}"
            )
    if show_overhead:
        top_nodes = top_k_nodes_by_messages(result.per_node_messages, top_k)
        top_edges = top_k_edges_by_messages(result.per_edge_messages, top_k)
        parts.append(
            "TopNodeMsgs: "
            + ", ".join(f"{node}:{count}" for node, count in top_nodes)
        )
        parts.append(
            "TopEdgeMsgs: "
            + ", ".join(f"{edge[0]}-{edge[1]}:{count}" for edge, count in top_edges)
        )
    if detect_bottlenecks_flag:
        detected = detect_bottlenecks(
            result.arrival_times, result.per_node_messages, bottleneck_fraction
        )
        if detected:
            parts.append(
                "DetectedBottlenecks: "
                + ", ".join(f"{node}:{score:.2f}" for node, score in detected[:top_k])
            )
            if validate_bottlenecks_flag and result.bottleneck_nodes:
                validation = validate_bottlenecks(
                    [node for node, _ in detected], list(result.bottleneck_nodes)
                )
                parts.append(
                    "BottleneckValidation: "
                    f"precision={validation.precision:.2f}, "
                    f"recall={validation.recall:.2f}, "
                    f"f1={validation.f1:.2f}"
                )
            if validate_bottleneck_edges_flag and result.bottleneck_nodes:
                injected_edges = {
                    edge
                    for edge in result.per_edge_messages.keys()
                    if edge[0] in result.bottleneck_nodes or edge[1] in result.bottleneck_nodes
                }
                detected_edges = detect_bottleneck_edges(
                    result.per_edge_messages, bottleneck_fraction
                )
                validation = validate_bottlenecks(
                    [edge for edge, _ in detected_edges], list(injected_edges)
                )
                parts.append(
                    "EdgeBottleneckValidation: "
                    f"precision={validation.precision:.2f}, "
                    f"recall={validation.recall:.2f}, "
                    f"f1={validation.f1:.2f}"
                )
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


def format_histogram_if_requested(result: RunResult, bins: int) -> str | None:
    if bins <= 0:
        return None
    return format_histogram(result.arrival_times, bins)
