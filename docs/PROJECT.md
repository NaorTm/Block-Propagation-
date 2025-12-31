# Block Propagation Analysis in a P2P Network

This document captures the combined project description, theoretical model, protocol formalization, simulation plan, and deliverables.

## 1. General description and objectives

### 1.1 Overall goal

Systematically examine how a new block propagates in a decentralized P2P network under different gossip dissemination protocols and network characteristics, while connecting:
- The theoretical network model,
- The protocol used or compared against,
- A controlled simulation of propagation,
- Real measurements or integration with an existing system (optional).

### 1.2 Specific focus: single 1 MB block study

We focus on a single 1 MB block under two dissemination protocols:
- Naive flooding: each node forwards the full block on first receipt.
- Two-stage announce/request/block: announce is sent first; nodes request if needed; then receive the full block.

The simulation explicitly models latency and transmission time (block size / bandwidth). Metrics include T50/T90/T100 and arrival time distributions.

### 1.3 Additional possible direction

Propose or analyze a real improvement to dissemination (e.g., latency-aware or adaptive push/pull) and quantify the impact in simulation.

## 2. Theoretical model and network construction

### 2.1 Network model

The network is a graph G = (V, E), where:
- V is the set of nodes (full nodes),
- E is the set of bidirectional P2P connections.

Each edge has:
- Latency,
- Effective bandwidth.

## 3. Topology families

- Random regular graph: each node has degree approximately d.
- Scale-free: degree distribution approximates a power law (few supernodes, many low-degree nodes).
- Small-world: small diameter, high clustering.

## 4. Communication model

### 4.1 Latency

Latency is modeled as random per-edge (e.g., uniform in [0.05, 0.2] seconds).

### 4.2 Bandwidth

Block transfer time depends on size and bandwidth:

T_total = latency + (message_size / bandwidth)

## 5. Block size and traversal time

We set block size to 1 MB = 1,000,000 bytes. Transmission time for 10 Mbps is ~0.8 s for a 1 MB block.

## 6. Protocols (broad set)

- Simple flooding.
- Push, pull, push-pull.
- Bitcoin-style inv/getdata/block.
- Compact blocks (BIP152), mempool sync, block relay networks.

## 7. Formalization of the two core protocols

### 7.1 Flooding

Nodes record first_receive_time for the block. On first receipt, a node forwards to all neighbors.

### 7.2 Two-stage announce/request/block

Message types:
- ANNOUNCE: small, latency-only delivery.
- REQUEST: small, latency-only delivery.
- BLOCK: full 1 MB payload, latency + transmission time.

Nodes track:
- knows_block,
- has_full_block,
- first_receive_time.

## 8. Performance dependence

We study:
- Block dissemination rate (T50/T90/T100),
- Tail latency and arrival distributions,
- Effects of latency variability,
- Degree distribution effects,
- Bandwidth and block size effects,
- Macro metrics: orphan rate, fork depth, security margin (proxy + Monte Carlo).

## 9. Simulation design

### 9.1 Implementation choices

Python with networkx, numpy, matplotlib or plotly.

### 9.2 Node behavior model

Nodes maintain:
- Neighbor list,
- Incoming message queue with timestamps,
- State (knows_block / has_full_block),
- Protocol logic.

### 9.3 Parameters

- Number of nodes,
- Topology type,
- Degree distribution,
- Latency distribution,
- Dropout or churn with mid-propagation failure times and delayed responders,
- Block size and bandwidth.

### 9.4 Discrete event simulation

Maintain a priority queue of events (time, type, src, dst). Pop events, update state, and schedule future events.

## 10. Metrics

- T50/T90/T100 thresholds,
- Arrival time distributions (CDF/histogram),
- Path-based analysis (shortest-time path vs observed),
- Load/overhead (messages per node/edge),
- Bottleneck scenarios (high latency/low bandwidth subsets),
- Bottleneck detection validation (precision/recall against injected bottlenecks),
- Edge-level bottleneck validation (precision/recall against injected edges),
- Macro metrics (proxy models for orphan rate, fork depth, security margin).

## 11. Experiment plan

1. Baseline: N=500, degree=8, latency 0.05-0.2 s, bandwidth 10 Mbps.
2. Flooding run and metrics.
3. Multi-run aggregation (mean/min/max).
4. Two-stage protocol with same settings.
5. Compare protocols and plot CDFs.
6. Extended experiments: fanout, topology, churn/dropout, improved protocol.

## 12. Optional real system integration

Setup a Bitcoin/Ethereum node (testnet/regtest), collect propagation timestamps, and compare with simulation after parameter calibration.

## 13. Project architecture

Suggested modular components:
- Network generator,
- Protocols,
- Simulator (event queue),
- Metrics,
- Visualization,
- Real-world adapter.
  - Placeholder scripts live under `real_world/` for future log ingestion.

## 14. Deliverables

- Final report (theory + simulation + results),
- Clean code repository with reproducible experiments,
- Summary presentation,
- Demo video or live visualization.
### 7.3 Push, pull, push-pull variants (implemented)

Push: on first receipt, a node pushes the full block to a fanout of neighbors.

Pull: nodes periodically request the block from neighbors; a neighbor with the block responds with the full payload.

Push-pull: combines periodic pull with a push on first receipt.

### 7.4 Bitcoin optimizations (modeled)

Compact blocks: full block payload is replaced with a smaller compact payload. Reconstruction uses an explicit transaction-overlap ratio; missing transactions are requested based on the overlap shortfall.

Relay networks: a subset of nodes are designated as relay nodes; edges between relay nodes have lower latency and/or higher bandwidth, with an optional relay-only overlay layer.
