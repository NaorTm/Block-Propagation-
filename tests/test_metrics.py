import pytest
import math
from src.metrics import threshold_time, arrival_histogram, macro_metrics, PathStats


class TestThresholdTime:
    def test_t50_calculation(self):
        arrivals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        t50 = threshold_time(arrivals, 0.5)
        assert abs(t50 - 0.4) < 0.01
    
    def test_with_infinite_values(self):
        arrivals = [0.0, 0.1, 0.2, math.inf, math.inf]
        t50 = threshold_time(arrivals, 0.5)
        assert math.isfinite(t50)
    
    def test_all_infinite(self):
        arrivals = [math.inf, math.inf, math.inf]
        t100 = threshold_time(arrivals, 1.0)
        assert t100 == math.inf
    
    def test_t100_calculation(self):
        arrivals = [0.0, 0.1, 0.2, 0.3, 0.4]
        t100 = threshold_time(arrivals, 1.0)
        assert t100 == 0.4
    
    def test_t90_calculation(self):
        arrivals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        t90 = threshold_time(arrivals, 0.9)
        assert abs(t90 - 0.8) < 0.01


class TestArrivalHistogram:
    def test_histogram_bins(self):
        arrivals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        hist = arrival_histogram(arrivals, bins=3)
        assert len(hist) == 3
    
    def test_invalid_bins(self):
        with pytest.raises(ValueError, match="bins must be positive"):
            arrival_histogram([0.1, 0.2], bins=0)
    
    def test_histogram_structure(self):
        arrivals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        hist = arrival_histogram(arrivals, bins=2)
        for start, end, count, cdf in hist:
            assert start <= end
            assert count >= 0
            assert 0 <= cdf <= 1.0
    
    def test_cdf_reaches_one(self):
        arrivals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        hist = arrival_histogram(arrivals, bins=3)
        assert abs(hist[-1][3] - 1.0) < 0.001


class TestMacroMetrics:
    def test_valid_metrics(self):
        arrivals = [0.0, 0.1, 0.2, 0.3, 0.4]
        metrics = macro_metrics(arrivals, block_interval=600.0)
        assert 0 <= metrics.competing_block_prob_t90 <= 1
        assert 0 <= metrics.security_margin_t50 <= 1
    
    def test_invalid_block_interval(self):
        with pytest.raises(ValueError, match="block_interval must be positive"):
            macro_metrics([0.1], block_interval=0)
    
    def test_negative_block_interval(self):
        with pytest.raises(ValueError, match="block_interval must be positive"):
            macro_metrics([0.1], block_interval=-1.0)
    
    def test_metrics_structure(self):
        arrivals = [0.0, 0.1, 0.2, 0.3, 0.4]
        metrics = macro_metrics(arrivals, block_interval=600.0)
        assert hasattr(metrics, 'competing_block_prob_t90')
        assert hasattr(metrics, 'expected_competing_blocks_t100')
        assert hasattr(metrics, 'prob_competing_blocks_ge1_t100')
        assert hasattr(metrics, 'prob_competing_blocks_ge2_t100')
        assert hasattr(metrics, 'security_margin_t50')
