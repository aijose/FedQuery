"""Unit tests for confidence evaluation thresholds.

Thresholds calibrated for cosine similarity on FOMC corpus:
high ≥ 0.55, medium ≥ 0.40, low ≥ 0.25, insufficient < 0.25.
"""

import pytest

from src.agent.nodes import evaluate_confidence_level


class TestConfidenceEvaluation:
    """Test threshold logic: high ≥0.55, medium ≥0.40, low ≥0.25, insufficient <0.25."""

    def test_high_confidence(self):
        assert evaluate_confidence_level(0.70) == "high"
        assert evaluate_confidence_level(0.55) == "high"
        assert evaluate_confidence_level(1.0) == "high"

    def test_medium_confidence(self):
        assert evaluate_confidence_level(0.50) == "medium"
        assert evaluate_confidence_level(0.40) == "medium"
        assert evaluate_confidence_level(0.54) == "medium"

    def test_low_confidence(self):
        assert evaluate_confidence_level(0.30) == "low"
        assert evaluate_confidence_level(0.25) == "low"
        assert evaluate_confidence_level(0.39) == "low"

    def test_insufficient_confidence(self):
        assert evaluate_confidence_level(0.24) == "insufficient"
        assert evaluate_confidence_level(0.0) == "insufficient"
        assert evaluate_confidence_level(0.1) == "insufficient"

    def test_boundary_values(self):
        """Test exact boundary values."""
        assert evaluate_confidence_level(0.55) == "high"
        assert evaluate_confidence_level(0.40) == "medium"
        assert evaluate_confidence_level(0.25) == "low"
        assert evaluate_confidence_level(0.24) == "insufficient"
