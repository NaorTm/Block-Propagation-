from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import SimulationConfig
from src.simulator import run_experiments


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
        if self.path != "/simulate":
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

        try:
            config = _apply_overrides(overrides)
            result = run_experiments(protocol, runs, config, seed)
            summary = {
                "t50": result.summary["t50"]["mean"],
                "t90": result.summary["t90"]["mean"],
                "t100": result.summary["t100"]["mean"],
                "messages": result.summary["messages"]["mean"],
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
