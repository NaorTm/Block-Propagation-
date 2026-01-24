from __future__ import annotations

import json
import sys
import math
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import SimulationConfig
from src.network import build_network
from src.protocols import (
    simulate_naive_flooding,
    simulate_pull,
    simulate_push,
    simulate_push_pull,
    simulate_two_phase,
)
from src.simulator import run_experiments
import networkx as nx


def _scenario_overrides(name: str) -> Dict[str, Any]:
    if name == "baseline_naive":
        return {}
    if name == "baseline_two_phase":
        return {}
    if name == "push_fanout":
        return {"gossip_fanout": 4}
    if name == "pull_interval":
        return {"pull_interval": 0.5, "pull_fanout": 2, "max_time": 20}
    if name == "push_pull":
        return {"gossip_fanout": 4, "pull_interval": 0.5, "max_time": 20}
    if name == "scale_free":
        return {"topology": "scale-free", "scale_free_m": 3}
    if name == "small_world":
        return {"topology": "small-world", "degree": 8, "rewire_prob": 0.2}
    if name == "relay_overlay":
        return {
            "relay_fraction": 0.2,
            "relay_overlay_degree": 2,
            "relay_latency_mult": 0.5,
            "relay_bandwidth_mult": 2.0,
        }
    if name == "compact_blocks":
        return {"mempool_overlap_mean": 0.85, "mempool_overlap_std": 0.05}
    if name == "bottlenecks":
        return {
            "bottleneck_fraction": 0.1,
            "bottleneck_latency_mult": 3.0,
            "bottleneck_bandwidth_mult": 0.5,
        }
    if name == "churn_delay":
        return {
            "churn_prob": 0.1,
            "churn_time_min": 1.0,
            "churn_time_max": 3.0,
            "delay_prob": 0.2,
            "delay_latency_mult": 2.0,
            "delay_bandwidth_mult": 0.7,
            "pull_interval": 0.5,
            "max_time": 20,
        }
    if name == "macro_metrics":
        return {}
    return {}


def _apply_overrides(overrides: Dict[str, Any]) -> SimulationConfig:
    config = SimulationConfig()
    data = config.__dict__.copy()
    data.update(overrides)
    return SimulationConfig(**data)


def _protocol_runner(protocol: str):
    if protocol == "naive":
        return simulate_naive_flooding
    if protocol == "two-phase":
        return simulate_two_phase
    if protocol == "push":
        return simulate_push
    if protocol == "pull":
        return simulate_pull
    if protocol == "push-pull":
        return simulate_push_pull
    if protocol == "bitcoin-compact":
        return lambda config, rng, include_path_stats, trace_events, network: simulate_two_phase(
            config,
            rng,
            include_path_stats=include_path_stats,
            compact_blocks=True,
            trace_events=trace_events,
            network=network,
        )
    raise ValueError(
        "Protocol must be 'naive', 'two-phase', 'push', 'pull', 'push-pull', or 'bitcoin-compact'"
    )


def _layout_positions(adjacency: list[set], seed: int | None) -> Dict[int, tuple[float, float]]:
    graph = nx.Graph()
    for node, neighbors in enumerate(adjacency):
        for neighbor in neighbors:
            if node < neighbor:
                graph.add_edge(node, neighbor)
    positions = nx.spring_layout(graph, seed=seed)
    return {node: (float(pos[0]), float(pos[1])) for node, pos in positions.items()}


def _normalize_positions(positions: Dict[int, tuple[float, float]]) -> Dict[int, tuple[float, float]]:
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    if not xs or not ys:
        return positions
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0
    return {
        node: ((x - min_x) / span_x, (y - min_y) / span_y)
        for node, (x, y) in positions.items()
    }


def _simulate_trace(protocol: str, config: SimulationConfig, seed: int | None) -> Dict[str, Any]:
    rng = random.Random(seed)
    network = build_network(config, rng)
    trace_events: list[Dict[str, Any]] = []

    runner = _protocol_runner(protocol)
    result = runner(
        config,
        rng,
        include_path_stats=False,
        trace_events=trace_events,
        network=network,
    )

    positions = _normalize_positions(_layout_positions(network.adjacency, seed))
    nodes = [
        {
            "id": node,
            "x": positions.get(node, (0.5, 0.5))[0],
            "y": positions.get(node, (0.5, 0.5))[1],
            "relay": node in network.relay_nodes,
            "bottleneck": node in network.bottleneck_nodes,
        }
        for node in range(config.num_nodes)
    ]
    edges = []
    for node, neighbors in enumerate(network.adjacency):
        for neighbor in neighbors:
            if node < neighbor:
                edges.append({"source": node, "target": neighbor, "type": "base"})
    for node, neighbors in enumerate(network.overlay):
        for neighbor in neighbors:
            if node < neighbor:
                edges.append({"source": node, "target": neighbor, "type": "relay"})

    trace_events.sort(key=lambda event: event["time"])
    max_time = trace_events[-1]["time"] if trace_events else 0.0

    def _clean(value: float) -> float | None:
        return value if math.isfinite(value) else None

    summary = {
        "t50": _clean(result.t50),
        "t90": _clean(result.t90),
        "t100": _clean(result.t100),
        "messages": _clean(result.total_messages),
    }
    return {
        "summary": summary,
        "graph": {"nodes": nodes, "edges": edges},
        "events": trace_events,
        "meta": {
            "protocol": protocol,
            "topology": config.topology,
            "max_time": max_time,
            "seed": seed,
        },
    }


class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json({"error": "not_found"}, status=404)
            return
        self._send_json({"status": "ok"})

    def do_POST(self) -> None:
        if self.path not in ("/simulate", "/simulate_trace"):
            self._send_json({"error": "not_found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid_json"}, status=400)
            return

        protocol = data.get("protocol", "two-phase")
        scenario = data.get("scenario", "")
        runs = int(data.get("runs", 3))
        seed = data.get("seed", 42)

        overrides = _scenario_overrides(scenario)
        for key in (
            "num_nodes",
            "degree",
            "topology",
            "scale_free_m",
            "block_size_bytes",
            "bandwidth_mbps",
            "latency_min",
            "latency_max",
            "mempool_overlap_mean",
            "mempool_overlap_std",
            "churn_prob",
            "delay_prob",
            "delay_latency_mult",
            "delay_bandwidth_mult",
        ):
            if key in data:
                overrides[key] = data[key]
        if "nodes" in data:
            overrides["num_nodes"] = data["nodes"]

        try:
            config = _apply_overrides(overrides)
            if self.path == "/simulate_trace":
                payload = _simulate_trace(protocol, config, seed)
                self._send_json(payload)
            else:
                result = run_experiments(protocol, runs, config, seed)
                def _clean(value: float) -> float | None:
                    return value if math.isfinite(value) else None

                summary = {
                    "t50": _clean(result.summary["t50"]["mean"]),
                    "t90": _clean(result.summary["t90"]["mean"]),
                    "t100": _clean(result.summary["t100"]["mean"]),
                    "messages": _clean(result.summary["messages"]["mean"]),
                }
                self._send_json({"summary": summary})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def main() -> None:
    server = HTTPServer(("0.0.0.0", 8000), _Handler)
    print("Simulation server running on http://localhost:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
