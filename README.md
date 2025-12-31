# Block Propagation Analysis

This project studies how a single 1 MB block propagates through a decentralized P2P network. It connects a theoretical network model (graph topology, per-link latency, bandwidth) to protocol behavior and measurable propagation metrics.

Scope implemented in code:
- Network topologies: random regular, scale-free (Barabasi-Albert), small-world (Watts-Strogatz), star, line.
- Protocols: naive flooding and a two-stage announce/request/block flow inspired by Bitcoin.
- Gossip variants: push, pull, and push-pull.
- Bitcoin optimizations (modeled): compact blocks and relay networks.
- Link characteristics: per-edge latency and bandwidth distributions, plus a transmission-time term for the 1 MB block.
- Forwarding churn: optional per-node drop probability.
- Bottlenecks: optional high-latency/low-bandwidth node subsets.
- Metrics: T50/T90/T100, message counts, optional path stretch/slack, optional arrival histogram/CDF.

Full project description and deliverables live in `docs/PROJECT.md`.

## Quick start

```bash
pip install -r requirements.txt
python simulation.py --protocol naive
python simulation.py --protocol two-phase
```

Compare multiple runs and summarize the distributions:

```bash
python simulation.py --protocol naive --runs 10 --seed 42
python simulation.py --protocol two-phase --runs 10 --seed 42
```

Run the baseline experiment and produce a CDF plot:

```bash
python experiments/run_base.py --runs 10 --seed 42
```

Run a scenario sweep and export CSV summaries:

```bash
python experiments/run_series.py --runs 5 --seed 42
```

Run the full test matrix and plot results:

```bash
python experiments/run_all.py --runs 3 --seed 42
python experiments/plot_results.py --input outputs/all_tests_summary.csv --output outputs/summary_plot.png
```

Animate propagation (interactive window or save to GIF):

```bash
python experiments/animate_propagation.py --protocol naive --nodes 100 --degree 6
python experiments/animate_propagation.py --protocol two-phase --nodes 100 --degree 6 --save outputs/prop.gif
```

## Key options

- `--nodes` / `--degree` - node count and target degree (small-world requires even degree).
- `--topology` - `random-regular`, `scale-free`, `small-world`, `star`, `line`.
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
- `--compact-block-bytes` / `--compact-success-prob` - compact block size and success rate for `bitcoin-compact`.
- `--mempool-overlap-mean` / `--mempool-overlap-std` - transaction overlap ratio for compact reconstruction.
- `--mempool-sync-prob` - legacy overlap mean if `--mempool-overlap-mean` is unset.
- `--missing-tx-bytes-min` - minimum missing transaction payload size when reconstruction fails.
- `--gossip-fanout` - push/push-pull fanout (0 = all neighbors).
- `--pull-interval` / `--pull-fanout` - pull period and number of neighbors queried.
- `--max-time` - maximum simulated time for pull-based protocols.
- `--drop-prob` - probability a node will not forward after receiving the block.
- `--bottleneck-fraction` - fraction of nodes treated as bottlenecks.
- `--bottleneck-latency-mult` - latency multiplier for bottleneck edges.
- `--bottleneck-bandwidth-mult` - bandwidth multiplier for bottleneck edges.
- `--relay-fraction` - fraction of nodes treated as relay nodes.
- `--relay-overlay-degree` / `--relay-overlay-prob` - add a relay-only overlay graph.
- `--relay-latency-mult` - latency multiplier for edges between relay nodes.
- `--relay-bandwidth-mult` - bandwidth multiplier for edges between relay nodes.
- `--churn-prob` / `--churn-time-min` / `--churn-time-max` - mid-propagation failure model.
- `--delay-prob` / `--delay-latency-mult` / `--delay-bandwidth-mult` - delayed responder model.
- `--hist-bins` - if > 0, print histogram/CDF of arrival times.
- `--path-stats` - compute shortest-path stretch/slack statistics.
- `--show-overhead` - print top per-node and per-edge message counts.
- `--top-k` - number of top nodes/edges for summaries.
- `--detect-bottlenecks` - run heuristic bottleneck detection.
- `--validate-bottlenecks` - compare detected bottlenecks to injected nodes.
- `--validate-bottleneck-edges` - compare detected bottleneck edges to injected edges.
- `--detect-bottleneck-fraction` - fraction of slowest nodes considered.
- `--show-macro` - print macro metric proxies (orphan/fork/security).
- `--macro-sim-trials` - Monte Carlo trials for orphan/fork simulation.
- `--block-interval` - block interval in seconds for macro metrics.

## Project structure

- `simulation.py` - CLI entrypoint.
- `src/config.py` - simulation configuration dataclass.
- `src/network.py` - topology generation and per-edge attributes.
- `src/protocols.py` - dissemination protocols.
- `src/metrics.py` - thresholds, histogram/CDF, path stats.
- `src/simulator.py` - experiment runner and formatting.
- `experiments/run_base.py` - baseline run + CDF plot.
- `real_world/` - placeholders for real-world log parsing.
- `real_world/parse_logs.py` - parse CSV logs into arrival time summaries.
- `real_world/parse_logs.py --format bitcoin-core` - parse Bitcoin Core debug.log lines.

## Requirements

Tested with Python 3.11. Install dependencies with:

```bash
pip install -r requirements.txt
```
