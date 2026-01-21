import pytest
import random
from src.config import SimulationConfig
from src.network import build_network, generate_graph, NetworkData


class TestNetworkGeneration:
    def test_build_network_returns_network_data(self):
        config = SimulationConfig(num_nodes=20, degree=4)
        rng = random.Random(42)
        result = build_network(config, rng)
        assert isinstance(result, NetworkData)
    
    def test_adjacency_structure(self):
        config = SimulationConfig(num_nodes=20, degree=4)
        rng = random.Random(42)
        network = build_network(config, rng)
        assert len(network.adjacency) == 20
        for neighbors in network.adjacency:
            assert isinstance(neighbors, set)
    
    def test_scale_free_topology(self):
        config = SimulationConfig(num_nodes=50, topology="scale-free", scale_free_m=2)
        rng = random.Random(42)
        adjacency = generate_graph(config, rng)
        assert len(adjacency) == 50
    
    def test_bottleneck_node_selection(self):
        config = SimulationConfig(num_nodes=100, bottleneck_fraction=0.1)
        rng = random.Random(42)
        network = build_network(config, rng)
        assert len(network.bottleneck_nodes) == 10
    
    def test_relay_node_selection(self):
        config = SimulationConfig(num_nodes=100, relay_fraction=0.2)
        rng = random.Random(42)
        network = build_network(config, rng)
        assert len(network.relay_nodes) == 20
    
    def test_latency_map_structure(self):
        config = SimulationConfig(num_nodes=20, degree=4)
        rng = random.Random(42)
        network = build_network(config, rng)
        # Latencies should be positive
        for (u, v), latency in network.latencies.items():
            assert latency > 0
            assert u < v  # Canonical form
    
    def test_bandwidth_map_structure(self):
        config = SimulationConfig(num_nodes=20, degree=4)
        rng = random.Random(42)
        network = build_network(config, rng)
        # Bandwidths should be positive
        for (u, v), bandwidth in network.bandwidths.items():
            assert bandwidth > 0
            assert u < v  # Canonical form
    
    def test_small_world_topology(self):
        config = SimulationConfig(num_nodes=50, topology="small-world", degree=4, rewire_prob=0.1)
        rng = random.Random(42)
        adjacency = generate_graph(config, rng)
        assert len(adjacency) == 50
    
    def test_star_topology(self):
        config = SimulationConfig(num_nodes=20, topology="star")
        rng = random.Random(42)
        adjacency = generate_graph(config, rng)
        assert len(adjacency) == 20
        # Node 0 should be connected to all others
        assert len(adjacency[0]) == 19
    
    def test_line_topology(self):
        config = SimulationConfig(num_nodes=20, topology="line")
        rng = random.Random(42)
        adjacency = generate_graph(config, rng)
        assert len(adjacency) == 20
        # First and last nodes have degree 1
        assert len(adjacency[0]) == 1
        assert len(adjacency[19]) == 1
