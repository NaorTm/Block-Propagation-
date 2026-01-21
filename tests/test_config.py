import pytest
from src.config import SimulationConfig


class TestSimulationConfig:
    def test_default_config_valid(self):
        config = SimulationConfig()
        assert config.num_nodes == 500
        assert config.degree == 8
    
    def test_invalid_num_nodes(self):
        with pytest.raises(ValueError, match="num_nodes must be at least 2"):
            SimulationConfig(num_nodes=1)
    
    def test_invalid_degree(self):
        with pytest.raises(ValueError, match="degree must be less than num_nodes"):
            SimulationConfig(num_nodes=10, degree=15)
    
    def test_invalid_drop_prob(self):
        with pytest.raises(ValueError, match="drop_prob must be between 0 and 1"):
            SimulationConfig(drop_prob=1.5)
    
    def test_invalid_source(self):
        with pytest.raises(ValueError, match="source must be a valid node index"):
            SimulationConfig(num_nodes=10, source=15)
    
    def test_transmission_time(self):
        config = SimulationConfig(block_size_bytes=1_000_000, bandwidth_mbps=10.0)
        time = config.transmission_time(10.0)
        assert abs(time - 0.8) < 0.001  # 1MB at 10Mbps = 0.8s
    
    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size_bytes must be positive"):
            SimulationConfig(block_size_bytes=0)
    
    def test_invalid_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth_mbps must be positive"):
            SimulationConfig(bandwidth_mbps=-1.0)
    
    def test_invalid_compact_success_prob(self):
        with pytest.raises(ValueError, match="compact_success_prob must be between 0 and 1"):
            SimulationConfig(compact_success_prob=1.5)
    
    def test_invalid_mempool_sync_prob(self):
        with pytest.raises(ValueError, match="mempool_sync_prob must be between 0 and 1"):
            SimulationConfig(mempool_sync_prob=-0.1)
    
    def test_invalid_churn_prob(self):
        with pytest.raises(ValueError, match="churn_prob must be between 0 and 1"):
            SimulationConfig(churn_prob=2.0)
    
    def test_invalid_delay_prob(self):
        with pytest.raises(ValueError, match="delay_prob must be between 0 and 1"):
            SimulationConfig(delay_prob=-0.5)
    
    def test_invalid_bottleneck_fraction(self):
        with pytest.raises(ValueError, match="bottleneck_fraction must be between 0 and 1"):
            SimulationConfig(bottleneck_fraction=1.1)
    
    def test_invalid_relay_fraction(self):
        with pytest.raises(ValueError, match="relay_fraction must be between 0 and 1"):
            SimulationConfig(relay_fraction=1.5)
