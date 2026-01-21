import pytest
import random
import math
from src.config import SimulationConfig
from src.protocols import (
    simulate_naive_flooding,
    simulate_two_phase,
    simulate_push,
    simulate_pull,
    simulate_push_pull,
)


class TestNaiveFlooding:
    def test_all_nodes_receive_block(self):
        config = SimulationConfig(num_nodes=20, degree=4)
        rng = random.Random(42)
        result = simulate_naive_flooding(config, rng)
        assert all(math.isfinite(t) for t in result.arrival_times)
    
    def test_source_has_zero_arrival_time(self):
        config = SimulationConfig(num_nodes=20, degree=4, source=0)
        rng = random.Random(42)
        result = simulate_naive_flooding(config, rng)
        assert result.arrival_times[0] == 0.0
    
    def test_t50_less_than_t90_less_than_t100(self):
        config = SimulationConfig(num_nodes=20, degree=4)
        rng = random.Random(42)
        result = simulate_naive_flooding(config, rng)
        assert result.t50 <= result.t90 <= result.t100
    
    def test_messages_are_counted(self):
        config = SimulationConfig(num_nodes=20, degree=4)
        rng = random.Random(42)
        result = simulate_naive_flooding(config, rng)
        assert result.total_messages > 0
        assert "block" in result.messages
    
    def test_protocol_name(self):
        config = SimulationConfig(num_nodes=20, degree=4)
        rng = random.Random(42)
        result = simulate_naive_flooding(config, rng)
        assert result.protocol == "naive"


class TestTwoPhase:
    def test_all_nodes_receive_block(self):
        config = SimulationConfig(num_nodes=20, degree=4)
        rng = random.Random(42)
        result = simulate_two_phase(config, rng)
        assert all(math.isfinite(t) for t in result.arrival_times)
    
    def test_has_announce_messages(self):
        config = SimulationConfig(num_nodes=20, degree=4)
        rng = random.Random(42)
        result = simulate_two_phase(config, rng)
        assert "announce" in result.messages
    
    def test_protocol_name(self):
        config = SimulationConfig(num_nodes=20, degree=4)
        rng = random.Random(42)
        result = simulate_two_phase(config, rng)
        assert result.protocol == "two-phase"


class TestPush:
    def test_with_fanout(self):
        config = SimulationConfig(num_nodes=20, degree=4, gossip_fanout=2)
        rng = random.Random(42)
        result = simulate_push(config, rng)
        assert all(math.isfinite(t) for t in result.arrival_times)
    
    def test_protocol_name(self):
        config = SimulationConfig(num_nodes=20, degree=4, gossip_fanout=2)
        rng = random.Random(42)
        result = simulate_push(config, rng)
        assert result.protocol == "push"


class TestPull:
    def test_pull_requests_generated(self):
        config = SimulationConfig(num_nodes=20, degree=4, pull_interval=0.5, max_time=10)
        rng = random.Random(42)
        result = simulate_pull(config, rng)
        assert "pull_req" in result.messages
    
    def test_protocol_name(self):
        config = SimulationConfig(num_nodes=20, degree=4, pull_interval=0.5, max_time=10)
        rng = random.Random(42)
        result = simulate_pull(config, rng)
        assert result.protocol == "pull"


class TestPushPull:
    def test_hybrid_protocol(self):
        config = SimulationConfig(num_nodes=20, degree=4, gossip_fanout=2, pull_interval=0.5, max_time=10)
        rng = random.Random(42)
        result = simulate_push_pull(config, rng)
        assert "block" in result.messages
    
    def test_protocol_name(self):
        config = SimulationConfig(num_nodes=20, degree=4, gossip_fanout=2, pull_interval=0.5, max_time=10)
        rng = random.Random(42)
        result = simulate_push_pull(config, rng)
        assert result.protocol == "push-pull"
