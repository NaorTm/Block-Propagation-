"""Interactive Plotly Dash dashboard for block propagation results."""

from __future__ import annotations

import csv
import functools
import math
import random
from pathlib import Path
import sys
from typing import Dict, List

import dash
from dash import Dash, Input, Output, State, dcc, html, dash_table
import networkx as nx
import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import SimulationConfig
from src.metrics import detect_bottleneck_edges, detect_bottlenecks
from src.network import build_network
from src.protocols import (
    simulate_naive_flooding,
    simulate_pull,
    simulate_push,
    simulate_push_pull,
    simulate_two_phase,
)

SUMMARY_CSV = Path("outputs/all_tests_summary.csv")


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _edge_latency(latencies: Dict[tuple, float], u: int, v: int) -> float:
    return latencies[_edge_key(u, v)]


def _edge_bandwidth(bandwidths: Dict[tuple, float], u: int, v: int) -> float:
    return bandwidths[_edge_key(u, v)]


def scenarios() -> List[dict]:
    return [
        {"name": "baseline_naive", "protocol": "naive", "config": SimulationConfig()},
        {"name": "baseline_two_phase", "protocol": "two-phase", "config": SimulationConfig()},
        {"name": "push_fanout", "protocol": "push", "config": SimulationConfig(gossip_fanout=4)},
        {
            "name": "pull_interval",
            "protocol": "pull",
            "config": SimulationConfig(pull_interval=0.5, pull_fanout=2, max_time=20),
        },
        {
            "name": "push_pull",
            "protocol": "push-pull",
            "config": SimulationConfig(gossip_fanout=4, pull_interval=0.5, max_time=20),
        },
        {
            "name": "scale_free",
            "protocol": "two-phase",
            "config": SimulationConfig(topology="scale-free", scale_free_m=3),
        },
        {
            "name": "small_world",
            "protocol": "two-phase",
            "config": SimulationConfig(topology="small-world", degree=8, rewire_prob=0.2),
        },
        {
            "name": "relay_overlay",
            "protocol": "two-phase",
            "config": SimulationConfig(
                relay_fraction=0.2,
                relay_overlay_degree=2,
                relay_latency_mult=0.5,
                relay_bandwidth_mult=2.0,
            ),
        },
        {
            "name": "compact_blocks",
            "protocol": "bitcoin-compact",
            "config": SimulationConfig(mempool_overlap_mean=0.85, mempool_overlap_std=0.05),
        },
        {
            "name": "bottlenecks",
            "protocol": "two-phase",
            "config": SimulationConfig(
                bottleneck_fraction=0.1,
                bottleneck_latency_mult=3.0,
                bottleneck_bandwidth_mult=0.5,
            ),
        },
        {
            "name": "churn_delay",
            "protocol": "push-pull",
            "config": SimulationConfig(
                churn_prob=0.1,
                churn_time_min=1.0,
                churn_time_max=3.0,
                delay_prob=0.2,
                delay_latency_mult=2.0,
                delay_bandwidth_mult=0.7,
                pull_interval=0.5,
                max_time=20,
            ),
        },
        {"name": "macro_metrics", "protocol": "two-phase", "config": SimulationConfig()},
    ]


SCENARIO_MAP = {entry["name"]: entry for entry in scenarios()}


def read_summary_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


@functools.lru_cache(maxsize=32)
def run_scenario(scenario_name: str, seed: int) -> dict:
    entry = SCENARIO_MAP[scenario_name]
    config: SimulationConfig = entry["config"]
    protocol = entry["protocol"]

    network_rng = random.Random(seed)
    sim_rng = random.Random(seed + 1)
    network = build_network(config, network_rng)

    if protocol == "naive":
        result = simulate_naive_flooding(config, sim_rng, network=network)
    elif protocol == "two-phase":
        result = simulate_two_phase(config, sim_rng, network=network)
    elif protocol == "push":
        result = simulate_push(config, sim_rng, network=network)
    elif protocol == "pull":
        result = simulate_pull(config, sim_rng, network=network)
    elif protocol == "push-pull":
        result = simulate_push_pull(config, sim_rng, network=network)
    elif protocol == "bitcoin-compact":
        result = simulate_two_phase(config, sim_rng, compact_blocks=True, network=network)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    adjacency = network[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(config.num_nodes))
    for node, neighbors in enumerate(adjacency):
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)

    try:
        positions = nx.spring_layout(graph, seed=seed)
    except ModuleNotFoundError:
        positions = nx.random_layout(graph, seed=seed)

    return {"result": result, "graph": graph, "positions": positions, "network": network}


def compute_cdf(arrival_times: List[float]) -> tuple[List[float], List[float]]:
    finite_times = [t for t in arrival_times if math.isfinite(t)]
    if not finite_times:
        return [], []
    x = sorted(finite_times)
    y = [(i + 1) / len(x) for i in range(len(x))]
    return x, y


def scenario_summary_table(config: SimulationConfig, result) -> List[dict]:
    return [
        {"metric": "Protocol", "value": result.protocol},
        {"metric": "Topology", "value": config.topology},
        {"metric": "Nodes", "value": config.num_nodes},
        {"metric": "Degree", "value": config.degree},
        {"metric": "Latency (min,max)", "value": f"{config.latency_min:.3f}s, {config.latency_max:.3f}s"},
        {"metric": "Bandwidth (min,max)", "value": f"{config.bandwidth_min:.1f}, {config.bandwidth_max:.1f} Mbps"},
        {"metric": "Block size", "value": f"{config.block_size_bytes} bytes"},
        {"metric": "T50", "value": f"{result.t50:.3f}s"},
        {"metric": "T90", "value": f"{result.t90:.3f}s"},
        {"metric": "T100", "value": f"{result.t100:.3f}s"},
        {"metric": "Total messages", "value": f"{result.total_messages:.0f}"},
    ]


def ranking_figure(rows: List[Dict[str, str]]) -> go.Figure:
    if not rows:
        return go.Figure()
    parsed = [
        {
            "scenario": row["scenario"],
            "t50": float(row["t50_mean"]),
            "t90": float(row["t90_mean"]),
            "t100": float(row["t100_mean"]),
        }
        for row in rows
    ]
    parsed.sort(key=lambda item: item["t90"])
    labels = [row["scenario"] for row in parsed]
    fig = go.Figure()
    fig.add_bar(x=labels, y=[row["t50"] for row in parsed], name="T50")
    fig.add_bar(x=labels, y=[row["t90"] for row in parsed], name="T90")
    fig.add_bar(x=labels, y=[row["t100"] for row in parsed], name="T100")
    fig.update_layout(barmode="group", title="T50 / T90 / T100 (sorted by T90)")
    return fig


def cdf_figure(arrival_times: List[float]) -> go.Figure:
    x, y = compute_cdf(arrival_times)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="CDF"))
    fig.update_layout(title="Arrival Time CDF", xaxis_title="Time (s)", yaxis_title="CDF")
    return fig


def message_overhead_figure(messages: Dict[str, int]) -> go.Figure:
    if not messages:
        return go.Figure()
    fig = go.Figure()
    for label, value in messages.items():
        fig.add_bar(x=["messages"], y=[value], name=label)
    fig.update_layout(
        title="Message Overhead by Type",
        xaxis_title="Scenario",
        yaxis_title="Count",
        barmode="stack",
    )
    return fig


def network_animation_figure(
    graph: nx.Graph,
    positions: dict,
    arrival_times: List[float],
    edge_times: Dict[tuple, float],
) -> go.Figure:
    nodes = list(graph.nodes())
    max_time = max((t for t in arrival_times if math.isfinite(t)), default=1.0)
    times = np.linspace(0.0, max_time, num=25)

    edges = list(graph.edges())

    base_nodes = go.Scatter(
        x=[positions[n][0] for n in nodes],
        y=[positions[n][1] for n in nodes],
        mode="markers",
        marker=dict(size=8, color="lightgray"),
        text=[str(n) for n in nodes],
        hoverinfo="text",
    )

    frames = []
    for t in times:
        active_x = []
        active_y = []
        inactive_x = []
        inactive_y = []
        for u, v in edges:
            activation_time = edge_times.get((u, v), math.inf)
            target_x = active_x if activation_time <= t else inactive_x
            target_y = active_y if activation_time <= t else inactive_y
            target_x += [positions[u][0], positions[v][0], None]
            target_y += [positions[u][1], positions[v][1], None]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=inactive_x,
                        y=inactive_y,
                        mode="lines",
                        line=dict(width=1, color="#bbb"),
                        hoverinfo="none",
                    ),
                    go.Scatter(
                        x=active_x,
                        y=active_y,
                        mode="lines",
                        line=dict(width=2, color="#f39c12"),
                        hoverinfo="none",
                    ),
                    go.Scatter(
                        x=[positions[n][0] for n in nodes],
                        y=[positions[n][1] for n in nodes],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=[
                                arrival_times[n] if arrival_times[n] <= t else max_time * 1.1
                                for n in nodes
                            ],
                            cmin=0,
                            cmax=max_time,
                            colorscale="Viridis",
                            colorbar=dict(title="Arrival time (s)"),
                        ),
                        text=[f"{n}: {arrival_times[n]:.2f}s" for n in nodes],
                        hoverinfo="text",
                    )
                ],
                name=f"{t:.2f}",
            )
        )

    fig = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="lines", line=dict(width=1, color="#bbb")),
            go.Scatter(x=[], y=[], mode="lines", line=dict(width=2, color="#f39c12")),
            base_nodes,
        ],
        layout=go.Layout(
            title="Propagation Animation",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "method": "animate",
                            "args": [[frame.name], {"mode": "immediate", "frame": {"duration": 0}}],
                            "label": frame.name,
                        }
                        for frame in frames
                    ],
                    "currentvalue": {"prefix": "t="},
                }
            ],
        ),
        frames=frames,
    )
    return fig


def edge_activation_times(
    graph: nx.Graph,
    arrival_times: List[float],
    network: tuple,
    config: SimulationConfig,
) -> Dict[tuple, float]:
    (
        _adjacency,
        latencies,
        bandwidths,
        _bottleneck_nodes,
        relay_nodes,
        overlay,
        overlay_latencies,
        overlay_bandwidths,
    ) = network
    times: Dict[tuple, float] = {}
    for u, v in graph.edges():
        is_overlay = u in relay_nodes and v in relay_nodes and v in overlay[u]
        latency_map = overlay_latencies if is_overlay else latencies
        bandwidth_map = overlay_bandwidths if is_overlay else bandwidths
        latency = _edge_latency(latency_map, u, v)
        bandwidth = _edge_bandwidth(bandwidth_map, u, v)
        travel_time = latency + config.transmission_time(bandwidth)
        best = math.inf
        if math.isfinite(arrival_times[u]):
            best = min(best, arrival_times[u] + travel_time)
        if math.isfinite(arrival_times[v]):
            best = min(best, arrival_times[v] + travel_time)
        times[(u, v)] = best
    return times


def bottleneck_figure(graph: nx.Graph, positions: dict, result) -> go.Figure:
    nodes = list(graph.nodes())
    scores = detect_bottlenecks(result.arrival_times, result.per_node_messages, 0.1)
    score_map = {node: score for node, score in scores}
    max_time = max((t for t in result.arrival_times if math.isfinite(t)), default=1.0)
    times = np.linspace(0.0, max_time, num=25)

    edge_scores = detect_bottleneck_edges(result.per_edge_messages, 0.1)
    edge_score_map = {edge: score for edge, score in edge_scores[:50]}

    edge_x = []
    edge_y = []
    for u, v in graph.edges():
        edge_x += [positions[u][0], positions[v][0], None]
        edge_y += [positions[u][1], positions[v][1], None]

    edge_times = {}
    for (u, v) in edge_score_map.keys():
        u_time = result.arrival_times[u]
        v_time = result.arrival_times[v]
        if not (math.isfinite(u_time) and math.isfinite(v_time)):
            edge_times[(u, v)] = math.inf
        else:
            edge_times[(u, v)] = max(u_time, v_time)

    frames = []
    for t in times:
        time_scores = []
        for node in nodes:
            arrival = result.arrival_times[node]
            if not math.isfinite(arrival):
                time_scores.append(1.0)
            else:
                time_scores.append(1.0 if arrival > t else 0.0)
        active_edge_x = []
        active_edge_y = []
        for (u, v), edge_time in edge_times.items():
            if edge_time <= t:
                active_edge_x += [positions[u][0], positions[v][0], None]
                active_edge_y += [positions[u][1], positions[v][1], None]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode="lines",
                        line=dict(width=1, color="#bbb"),
                        hoverinfo="none",
                    ),
                    go.Scatter(
                        x=active_edge_x,
                        y=active_edge_y,
                        mode="lines",
                        line=dict(width=2, color="rgb(200,30,30)"),
                        hoverinfo="none",
                    ),
                    go.Scatter(
                        x=[positions[n][0] for n in nodes],
                        y=[positions[n][1] for n in nodes],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=time_scores,
                            colorscale="Reds",
                            cmin=0,
                            cmax=1,
                            colorbar=dict(title="Late (time)"),
                        ),
                        text=[f"{n}: {arrival:.2f}s" for n, arrival in enumerate(result.arrival_times)],
                        hoverinfo="text",
                    )
                ],
                name=f"{t:.2f}",
            )
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="#bbb"),
            hoverinfo="none",
            name="Edges",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(width=2, color="rgb(200,30,30)"),
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[positions[n][0] for n in nodes],
            y=[positions[n][1] for n in nodes],
            mode="markers",
            marker=dict(
                size=8,
                color=[0.0 for _ in nodes],
                colorscale="Reds",
                cmin=0,
                cmax=1,
                colorbar=dict(title="Late (time)"),
            ),
            text=[f"{n}: {score_map.get(n, 0.0):.2f}" for n in nodes],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Bottleneck Heatmap (nodes scored, edges shown)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "args": [[frame.name], {"mode": "immediate", "frame": {"duration": 0}}],
                        "label": frame.name,
                    }
                    for frame in frames
                ],
                "currentvalue": {"prefix": "t="},
            }
        ],
    )
    fig.frames = frames
    return fig


app: Dash = dash.Dash(__name__)

summary_rows = read_summary_rows(SUMMARY_CSV)

app.layout = html.Div(
    [
        html.H2("Block Propagation Dashboard"),
        html.Div(
            [
                html.Label("Seed"),
                dcc.Input(id="seed", type="number", value=42, step=1),
            ],
            style={"maxWidth": "200px"},
        ),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Comparison",
                    children=[
                        html.Div(
                            [
                                html.Label(
                                    [
                                        "Compare scenarios (2-4) ",
                                        html.Span(
                                            "(?)",
                                            title=(
                                                "Scenario = a predefined experiment setup "
                                                "(protocol + topology + parameters)."
                                            ),
                                            style={"cursor": "help", "textDecoration": "underline dotted"},
                                        ),
                                    ],
                                    title=(
                                        "Scenario = a predefined experiment setup "
                                        "(protocol + topology + parameters)."
                                    ),
                                ),
                                dcc.Dropdown(
                                    id="compare-scenarios",
                                    options=[
                                        {"label": name, "value": name}
                                        for name in SCENARIO_MAP
                                    ],
                                    value=["baseline_naive", "baseline_two_phase"],
                                    multi=True,
                                ),
                            ],
                            style={"maxWidth": "600px"},
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="compare-cdf"),
                                dcc.Graph(id="compare-messages"),
                            ]
                        ),
                        dcc.Graph(id="ranking-graph", figure=ranking_figure(summary_rows)),
                    ],
                ),
                dcc.Tab(
                    label="Scenario Viewer",
                    children=[
                        html.Div(
                            [
                                html.Label("Run mode"),
                                dcc.RadioItems(
                                    id="run-mode",
                                    options=[
                                        {"label": "Preset scenario", "value": "preset"},
                                        {"label": "Custom config", "value": "custom"},
                                    ],
                                    value="preset",
                                    inline=True,
                                ),
                                html.Label(
                                    [
                                        "Scenario ",
                                        html.Span(
                                            "(?)",
                                            title=(
                                                "Scenario = a predefined experiment setup "
                                                "(protocol + topology + parameters)."
                                            ),
                                            style={"cursor": "help", "textDecoration": "underline dotted"},
                                        ),
                                    ],
                                    title=(
                                        "Scenario = a predefined experiment setup "
                                        "(protocol + topology + parameters)."
                                    ),
                                ),
                                dcc.Dropdown(
                                    id="scenario",
                                    options=[
                                        {"label": name, "value": name}
                                        for name in SCENARIO_MAP
                                    ],
                                    value="baseline_two_phase",
                                    clearable=False,
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            [
                                                "Protocol ",
                                                html.Span(
                                                    "(?)",
                                                    title=(
                                                        "Protocol = the dissemination algorithm "
                                                        "(e.g., naive, two-phase, push, pull)."
                                                    ),
                                                    style={"cursor": "help", "textDecoration": "underline dotted"},
                                                ),
                                            ],
                                            title=(
                                                "Protocol = the dissemination algorithm "
                                                "(e.g., naive, two-phase, push, pull)."
                                            ),
                                        ),
                                        dcc.Dropdown(
                                            id="custom-protocol",
                                            options=[
                                                {"label": "naive", "value": "naive"},
                                                {"label": "two-phase", "value": "two-phase"},
                                                {"label": "push", "value": "push"},
                                                {"label": "pull", "value": "pull"},
                                                {"label": "push-pull", "value": "push-pull"},
                                                {
                                                    "label": "bitcoin-compact",
                                                    "value": "bitcoin-compact",
                                                },
                                            ],
                                            value="two-phase",
                                            clearable=False,
                                        ),
                                        html.Label(
                                            [
                                                "Topology ",
                                                html.Span(
                                                    "(?)",
                                                    title=(
                                                        "Topology = the network graph shape "
                                                        "(random-regular, scale-free, small-world, etc.)."
                                                    ),
                                                    style={"cursor": "help", "textDecoration": "underline dotted"},
                                                ),
                                            ],
                                            title=(
                                                "Topology = the network graph shape "
                                                "(random-regular, scale-free, small-world, etc.)."
                                            ),
                                        ),
                                        dcc.Dropdown(
                                            id="custom-topology",
                                            options=[
                                                {
                                                    "label": "random-regular",
                                                    "value": "random-regular",
                                                },
                                                {"label": "scale-free", "value": "scale-free"},
                                                {"label": "small-world", "value": "small-world"},
                                                {"label": "star", "value": "star"},
                                                {"label": "line", "value": "line"},
                                            ],
                                            value="random-regular",
                                            clearable=False,
                                        ),
                                        html.Label("Nodes"),
                                        dcc.Input(
                                            id="custom-nodes", type="number", value=500, step=1
                                        ),
                                        html.Label("Degree"),
                                        dcc.Input(
                                            id="custom-degree", type="number", value=8, step=1
                                        ),
                                        html.Label("Scale-free m"),
                                        dcc.Input(
                                            id="custom-scale-free-m",
                                            type="number",
                                            value=3,
                                            step=1,
                                        ),
                                        html.Label("Small-world rewire prob"),
                                        dcc.Input(
                                            id="custom-rewire-prob",
                                            type="number",
                                            value=0.2,
                                            step=0.05,
                                        ),
                                        html.Label("Latency min (s)"),
                                        dcc.Input(
                                            id="custom-latency-min",
                                            type="number",
                                            value=0.05,
                                            step=0.01,
                                        ),
                                        html.Label("Latency max (s)"),
                                        dcc.Input(
                                            id="custom-latency-max",
                                            type="number",
                                            value=0.2,
                                            step=0.01,
                                        ),
                                        html.Label("Bandwidth min (Mbps)"),
                                        dcc.Input(
                                            id="custom-bandwidth-min",
                                            type="number",
                                            value=5.0,
                                            step=0.5,
                                        ),
                                        html.Label("Bandwidth max (Mbps)"),
                                        dcc.Input(
                                            id="custom-bandwidth-max",
                                            type="number",
                                            value=15.0,
                                            step=0.5,
                                        ),
                                        html.Label("Block size (bytes)"),
                                        dcc.Input(
                                            id="custom-block-bytes",
                                            type="number",
                                            value=1_000_000,
                                            step=100_000,
                                        ),
                                        html.Button("Run", id="run-button", n_clicks=0),
                                    ],
                                    style={"marginTop": "12px", "maxWidth": "400px"},
                                ),
                            ],
                            style={"maxWidth": "400px"},
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="cdf-graph"),
                                dcc.Graph(id="messages-graph"),
                            ]
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="network-anim"),
                                dcc.Graph(id="bottleneck-graph"),
                            ]
                        ),
                        html.H3("Scenario Summary"),
                        dash_table.DataTable(
                            id="summary-table",
                            columns=[
                                {"name": "Metric", "id": "metric"},
                                {"name": "Value", "id": "value"},
                            ],
                            data=[],
                            style_cell={"textAlign": "left", "padding": "4px"},
                            style_header={"fontWeight": "bold"},
                        ),
                    ],
                ),
            ]
        ),
    ],
    style={"padding": "20px"},
)


@app.callback(
    Output("compare-cdf", "figure"),
    Output("compare-messages", "figure"),
    Input("compare-scenarios", "value"),
    Input("seed", "value"),
)
def update_comparison(compare_scenarios: List[str], seed_value: int):
    seed = int(seed_value or 0)
    scenarios = list(compare_scenarios or [])[:4]
    cdf_fig = go.Figure()
    messages_fig = go.Figure()

    for scenario_name in scenarios:
        run_data = run_scenario(scenario_name, seed)
        result = run_data["result"]
        x, y = compute_cdf(result.arrival_times)
        cdf_fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", name=scenario_name)
        )
        for msg_type, count in result.messages.items():
            messages_fig.add_bar(
                x=[scenario_name],
                y=[count],
                name=msg_type,
                showlegend=True,
            )

    cdf_fig.update_layout(
        title="CDF Comparison",
        xaxis_title="Time (s)",
        yaxis_title="CDF",
    )
    messages_fig.update_layout(
        title="Message Overhead Comparison",
        xaxis_title="Scenario",
        yaxis_title="Count",
        barmode="stack",
    )
    return cdf_fig, messages_fig


@app.callback(
    Output("cdf-graph", "figure"),
    Output("messages-graph", "figure"),
    Output("network-anim", "figure"),
    Output("bottleneck-graph", "figure"),
    Output("summary-table", "data"),
    Input("run-button", "n_clicks"),
    Input("scenario", "value"),
    Input("seed", "value"),
    Input("run-mode", "value"),
    State("custom-protocol", "value"),
    State("custom-topology", "value"),
    State("custom-nodes", "value"),
    State("custom-degree", "value"),
    State("custom-scale-free-m", "value"),
    State("custom-rewire-prob", "value"),
    State("custom-latency-min", "value"),
    State("custom-latency-max", "value"),
    State("custom-bandwidth-min", "value"),
    State("custom-bandwidth-max", "value"),
    State("custom-block-bytes", "value"),
)
def update_scenario(
    _clicks: int,
    scenario_name: str,
    seed_value: int,
    run_mode: str,
    custom_protocol: str,
    custom_topology: str,
    custom_nodes: int,
    custom_degree: int,
    custom_scale_free_m: int,
    custom_rewire_prob: float,
    custom_latency_min: float,
    custom_latency_max: float,
    custom_bandwidth_min: float,
    custom_bandwidth_max: float,
    custom_block_bytes: int,
):
    seed = int(seed_value or 0)
    if run_mode == "custom":
        num_nodes = max(int(custom_nodes or 0), 2)
        degree = max(int(custom_degree or 0), 1)
        if custom_topology == "random-regular" and degree % 2 == 1:
            degree += 1
        degree = min(degree, num_nodes - 1)
        latency_min = float(custom_latency_min or 0.0)
        latency_max = float(custom_latency_max or latency_min)
        if latency_max < latency_min:
            latency_min, latency_max = latency_max, latency_min
        bandwidth_min = float(custom_bandwidth_min or 0.0)
        bandwidth_max = float(custom_bandwidth_max or bandwidth_min)
        if bandwidth_max < bandwidth_min:
            bandwidth_min, bandwidth_max = bandwidth_max, bandwidth_min
        config = SimulationConfig(
            num_nodes=num_nodes,
            degree=degree,
            topology=custom_topology,
            scale_free_m=int(custom_scale_free_m or 1),
            rewire_prob=float(custom_rewire_prob or 0.0),
            latency_min=latency_min,
            latency_max=latency_max,
            bandwidth_min=bandwidth_min,
            bandwidth_max=bandwidth_max,
            block_size_bytes=int(custom_block_bytes or 1_000_000),
        )
        network_rng = random.Random(seed)
        sim_rng = random.Random(seed + 1)
        network = build_network(config, network_rng)
        if custom_protocol == "naive":
            result = simulate_naive_flooding(config, sim_rng, network=network)
        elif custom_protocol == "two-phase":
            result = simulate_two_phase(config, sim_rng, network=network)
        elif custom_protocol == "push":
            result = simulate_push(config, sim_rng, network=network)
        elif custom_protocol == "pull":
            result = simulate_pull(config, sim_rng, network=network)
        elif custom_protocol == "push-pull":
            result = simulate_push_pull(config, sim_rng, network=network)
        elif custom_protocol == "bitcoin-compact":
            result = simulate_two_phase(config, sim_rng, compact_blocks=True, network=network)
        else:
            raise ValueError(f"Unknown protocol: {custom_protocol}")
        adjacency = network[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(config.num_nodes))
        for node, neighbors in enumerate(adjacency):
            for neighbor in neighbors:
                graph.add_edge(node, neighbor)
        try:
            positions = nx.spring_layout(graph, seed=seed)
        except ModuleNotFoundError:
            positions = nx.random_layout(graph, seed=seed)
        edge_times = edge_activation_times(graph, result.arrival_times, network, config)
        entry = {"config": config}
    else:
        entry = SCENARIO_MAP[scenario_name]
        run_data = run_scenario(scenario_name, seed)
        result = run_data["result"]
        graph = run_data["graph"]
        positions = run_data["positions"]
        edge_times = edge_activation_times(
            graph, result.arrival_times, run_data["network"], entry["config"]
        )

    return (
        cdf_figure(result.arrival_times),
        message_overhead_figure(result.messages),
        network_animation_figure(graph, positions, result.arrival_times, edge_times),
        bottleneck_figure(graph, positions, result),
        scenario_summary_table(entry["config"], result),
    )


if __name__ == "__main__":
    app.run(debug=True)
