from __future__ import annotations

from dataclasses import dataclass


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
    compact_block_bytes: int = 20_000
    compact_success_prob: float = 0.9
    mempool_sync_prob: float = 0.9
    mempool_overlap_mean: float | None = None
    mempool_overlap_std: float = 0.05
    missing_tx_bytes_min: int = 0
    drop_prob: float = 0.0
    bottleneck_fraction: float = 0.0
    bottleneck_latency_mult: float = 1.0
    bottleneck_bandwidth_mult: float = 1.0
    relay_fraction: float = 0.0
    relay_latency_mult: float = 1.0
    relay_bandwidth_mult: float = 1.0
    relay_overlay_degree: int = 0
    relay_overlay_prob: float = 0.0
    gossip_fanout: int = 0
    pull_interval: float = 1.0
    pull_fanout: int = 1
    max_time: float = 60.0
    churn_prob: float = 0.0
    churn_time_min: float = 0.0
    churn_time_max: float = 0.0
    delay_prob: float = 0.0
    delay_latency_mult: float = 1.0
    delay_bandwidth_mult: float = 1.0
    source: int = 0

    def transmission_time(self, bandwidth_mbps: float, block_size_bytes: int | None = None) -> float:
        """Seconds to transmit the full block over one edge."""

        size_bytes = self.block_size_bytes if block_size_bytes is None else block_size_bytes
        bits = size_bytes * 8
        bandwidth_bps = bandwidth_mbps * 1_000_000
        return bits / bandwidth_bps
