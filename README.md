# Block Propagation Simulation

This project models how a new block spreads through a decentralized P2P network. It connects a formal network model (graph topology, per-link latency and bandwidth) to simulated protocol behavior and measurable propagation metrics.

Current scope:
- Network topologies: random regular, scale-free (Barabasi-Albert), small-world (Watts-Strogatz).
- Protocols: naive flooding and a two-phase announce/request/block flow inspired by Bitcoin.
- Link characteristics: per-edge latency and bandwidth distributions with a transmission-time term.
- Forwarding churn: optional per-node drop probability (nodes receive but do not forward).
- Metrics: T50/T90/T100 propagation times and message counts per protocol.

## Quick start

Run a single simulation with default parameters (500 nodes, degree 8, 10 Mbps bandwidth, 1,000,000-byte block):

```bash
python simulation.py --protocol naive
python simulation.py --protocol two-phase
```

Compare multiple runs and summarize the distributions:

```bash
python simulation.py --protocol naive --runs 10 --seed 42
python simulation.py --protocol two-phase --runs 10 --seed 42
```

Key options:

- `--nodes` / `--degree` - node count and target degree (small-world requires even degree).
- `--topology` - `random-regular`, `scale-free`, or `small-world`.
- `--scale-free-m` - Barabasi-Albert attachment parameter (avg degree ~ 2*m).
- `--rewire-prob` - Watts-Strogatz rewiring probability.
- `--latency-dist` - `uniform` or `lognormal`.
- `--latency-min` / `--latency-max` - latency bounds in seconds (uniform).
- `--latency-mu` / `--latency-sigma` - lognormal parameters for latency.
- `--bandwidth-dist` - `fixed`, `uniform`, or `lognormal`.
- `--bandwidth-mbps` - fixed bandwidth in megabits per second.
- `--bandwidth-min` / `--bandwidth-max` - uniform bandwidth bounds.
- `--bandwidth-mu` / `--bandwidth-sigma` - lognormal parameters for bandwidth.
- `--block-bytes` - block size in bytes (default 1,000,000).
- `--drop-prob` - probability a node will not forward after receiving the block.
- `--hist-bins` - if > 0, print histogram/CDF of arrival times.
- `--runs` - number of independent experiments to average over.
- `--seed` - seed for reproducibility.

Each run prints T50/T90/T100 (times for 50%, 90%, and 100% of nodes to receive the full block) and message counts. When `--runs` > 1, aggregate min/mean/max values are shown.

## Project structure

- `simulation.py` - core simulation logic for both protocols, graph generation, CLI entry point.

## Requirements

Tested with Python 3.11. Install dependencies with:

```bash
pip install -r requirements.txt
```
