from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import SimulationConfig
from .metrics import PathStats


@dataclass
class RunResult:
    """Stores metrics for a single simulation run."""

    protocol: str
    config: SimulationConfig
    arrival_times: List[float]
    t50: float
    t90: float
    t100: float
    messages: Dict[str, int]
    per_node_messages: List[int]
    per_edge_messages: Dict[Tuple[int, int], int]
    bottleneck_nodes: set[int]
    relay_nodes: set[int]
    path_stats: PathStats | None = None

    @property
    def total_messages(self) -> int:
        return sum(self.messages.values())


@dataclass
class AggregateResult:
    """Aggregated metrics over multiple runs."""

    protocol: str
    runs: List[RunResult]
    summary: Dict[str, Dict[str, float]]
