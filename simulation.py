"""Command-line entry point for the block propagation simulator."""

from __future__ import annotations

import argparse

from src.config import SimulationConfig
from src.simulator import (
    format_aggregate,
    format_histogram_if_requested,
    format_run_result,
    run_experiments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block propagation simulator")
    parser.add_argument(
        "--protocol",
        choices=["naive", "two-phase", "push", "pull", "push-pull", "bitcoin-compact"],
        default="naive",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument(
        "--topology",
        choices=["random-regular", "scale-free", "small-world", "star", "line"],
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
    parser.add_argument(
        "--latency-dist",
        choices=["uniform", "lognormal"],
        default="uniform",
    )
    parser.add_argument(
        "--latency-min",
        type=float,
        default=0.05,
        help="Uniform latency min",
    )
    parser.add_argument(
        "--latency-max",
        type=float,
        default=0.2,
        help="Uniform latency max",
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
    parser.add_argument(
        "--bandwidth-dist",
        choices=["fixed", "uniform", "lognormal"],
        default="fixed",
    )
    parser.add_argument(
        "--bandwidth-mbps",
        type=float,
        default=10.0,
        help="Fixed bandwidth in Mbps",
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
        "--compact-block-bytes",
        type=int,
        default=20_000,
        help="Compact block size for bitcoin-compact protocol",
    )
    parser.add_argument(
        "--compact-success-prob",
        type=float,
        default=0.9,
        help="Probability compact block succeeds without fallback",
    )
    parser.add_argument(
        "--mempool-sync-prob",
        type=float,
        default=0.9,
        help="Legacy mean overlap if --mempool-overlap-mean is not set",
    )
    parser.add_argument(
        "--mempool-overlap-mean",
        type=float,
        default=None,
        help="Mean transaction overlap ratio for compact reconstruction",
    )
    parser.add_argument(
        "--mempool-overlap-std",
        type=float,
        default=0.05,
        help="Std dev for transaction overlap ratio",
    )
    parser.add_argument(
        "--missing-tx-bytes-min",
        type=int,
        default=0,
        help="Minimum missing transaction payload size when reconstruction fails",
    )
    parser.add_argument(
        "--missing-tx-bytes",
        dest="missing_tx_bytes_min",
        type=int,
        default=argparse.SUPPRESS,
        help="Deprecated alias for --missing-tx-bytes-min",
    )
    parser.add_argument(
        "--drop-prob",
        type=float,
        default=0.0,
        help="Probability a node will not forward after receiving the block",
    )
    parser.add_argument(
        "--gossip-fanout",
        type=int,
        default=0,
        help="Push/push-pull fanout (0 = all neighbors)",
    )
    parser.add_argument(
        "--pull-interval",
        type=float,
        default=1.0,
        help="Pull/push-pull interval in seconds",
    )
    parser.add_argument(
        "--pull-fanout",
        type=int,
        default=1,
        help="Number of neighbors queried per pull",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=60.0,
        help="Maximum simulated time for pull-based protocols",
    )
    parser.add_argument(
        "--bottleneck-fraction",
        type=float,
        default=0.0,
        help="Fraction of nodes marked as bottlenecks",
    )
    parser.add_argument(
        "--bottleneck-latency-mult",
        type=float,
        default=1.0,
        help="Latency multiplier for edges incident to bottleneck nodes",
    )
    parser.add_argument(
        "--bottleneck-bandwidth-mult",
        type=float,
        default=1.0,
        help="Bandwidth multiplier for edges incident to bottleneck nodes",
    )
    parser.add_argument(
        "--relay-fraction",
        type=float,
        default=0.0,
        help="Fraction of nodes marked as relay nodes",
    )
    parser.add_argument(
        "--relay-overlay-degree",
        type=int,
        default=0,
        help="Extra relay-only edges per relay node (overlay graph)",
    )
    parser.add_argument(
        "--relay-overlay-prob",
        type=float,
        default=0.0,
        help="Probability of adding relay-only edges (overlay graph)",
    )
    parser.add_argument(
        "--relay-latency-mult",
        type=float,
        default=1.0,
        help="Latency multiplier for edges between relay nodes",
    )
    parser.add_argument(
        "--relay-bandwidth-mult",
        type=float,
        default=1.0,
        help="Bandwidth multiplier for edges between relay nodes",
    )
    parser.add_argument(
        "--churn-prob",
        type=float,
        default=0.0,
        help="Probability a node fails at a random time and stops sending",
    )
    parser.add_argument(
        "--churn-time-min",
        type=float,
        default=0.0,
        help="Minimum churn failure time in seconds",
    )
    parser.add_argument(
        "--churn-time-max",
        type=float,
        default=0.0,
        help="Maximum churn failure time in seconds",
    )
    parser.add_argument(
        "--delay-prob",
        type=float,
        default=0.0,
        help="Probability a node has delayed responses",
    )
    parser.add_argument(
        "--delay-latency-mult",
        type=float,
        default=1.0,
        help="Latency multiplier for delayed nodes",
    )
    parser.add_argument(
        "--delay-bandwidth-mult",
        type=float,
        default=1.0,
        help="Bandwidth multiplier for delayed nodes",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=0,
        help="If > 0, print a histogram/CDF over arrival times",
    )
    parser.add_argument(
        "--show-overhead",
        action="store_true",
        help="Print top per-node and per-edge message counts",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top nodes/edges to show in summaries",
    )
    parser.add_argument(
        "--detect-bottlenecks",
        action="store_true",
        help="Run heuristic bottleneck detection",
    )
    parser.add_argument(
        "--validate-bottlenecks",
        action="store_true",
        help="Compare detected bottlenecks to injected bottleneck nodes",
    )
    parser.add_argument(
        "--validate-bottleneck-edges",
        action="store_true",
        help="Compare detected bottleneck edges to injected bottleneck edges",
    )
    parser.add_argument(
        "--detect-bottleneck-fraction",
        type=float,
        default=0.1,
        help="Fraction of slowest nodes considered for bottleneck detection",
    )
    parser.add_argument(
        "--show-macro",
        action="store_true",
        help="Print macro metric proxies (orphan/fork/security) based on block interval",
    )
    parser.add_argument(
        "--macro-sim-trials",
        type=int,
        default=0,
        help="Monte Carlo trials for orphan/fork simulation (0 disables)",
    )
    parser.add_argument(
        "--block-interval",
        type=float,
        default=600.0,
        help="Block interval in seconds for macro metric proxies",
    )
    parser.add_argument(
        "--path-stats",
        action="store_true",
        help="Compute shortest-path stretch/slack statistics",
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
        compact_block_bytes=args.compact_block_bytes,
        compact_success_prob=args.compact_success_prob,
        mempool_sync_prob=args.mempool_sync_prob,
        mempool_overlap_mean=args.mempool_overlap_mean,
        mempool_overlap_std=args.mempool_overlap_std,
        missing_tx_bytes_min=args.missing_tx_bytes_min,
        drop_prob=args.drop_prob,
        gossip_fanout=args.gossip_fanout,
        pull_interval=args.pull_interval,
        pull_fanout=args.pull_fanout,
        max_time=args.max_time,
        bottleneck_fraction=args.bottleneck_fraction,
        bottleneck_latency_mult=args.bottleneck_latency_mult,
        bottleneck_bandwidth_mult=args.bottleneck_bandwidth_mult,
        relay_fraction=args.relay_fraction,
        relay_overlay_degree=args.relay_overlay_degree,
        relay_overlay_prob=args.relay_overlay_prob,
        relay_latency_mult=args.relay_latency_mult,
        relay_bandwidth_mult=args.relay_bandwidth_mult,
        churn_prob=args.churn_prob,
        churn_time_min=args.churn_time_min,
        churn_time_max=args.churn_time_max,
        delay_prob=args.delay_prob,
        delay_latency_mult=args.delay_latency_mult,
        delay_bandwidth_mult=args.delay_bandwidth_mult,
        source=args.source,
    )

    aggregate = run_experiments(
        args.protocol, args.runs, config, args.seed, include_path_stats=args.path_stats
    )

    if args.runs == 1:
        result = aggregate.runs[0]
        print(
            format_run_result(
                result,
                show_overhead=args.show_overhead,
                top_k=args.top_k,
                detect_bottlenecks_flag=args.detect_bottlenecks,
                bottleneck_fraction=args.detect_bottleneck_fraction,
                show_macro=args.show_macro,
                block_interval=args.block_interval,
                macro_sim_trials=args.macro_sim_trials,
                validate_bottlenecks_flag=args.validate_bottlenecks,
                validate_bottleneck_edges_flag=args.validate_bottleneck_edges,
            )
        )
        histogram = format_histogram_if_requested(result, args.hist_bins)
        if histogram:
            print(histogram)
    else:
        print(format_aggregate(aggregate))


if __name__ == "__main__":
    main()
