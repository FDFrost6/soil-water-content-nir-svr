"""
Tests for src/guardrails.py – spectral safety guardrails.
"""

import sys
import os
import numpy as np
import pytest

# Make sure the src package is importable when tests are run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from guardrails import check_spectral_outliers, check_snr, check_prediction_range


# ---------------------------------------------------------------------------
# check_spectral_outliers
# ---------------------------------------------------------------------------

class TestCheckSpectralOutliers:
    def test_no_outliers_uniform(self):
        spectrum = np.ones(100)
        result = check_spectral_outliers(spectrum)
        assert result["has_outliers"] is False
        assert result["outlier_indices"] == []
        assert "OK" in result["message"]

    def test_detects_single_spike(self):
        spectrum = np.ones(50)
        spectrum[25] = 1000.0  # obvious spike
        result = check_spectral_outliers(spectrum)
        assert result["has_outliers"] is True
        assert 25 in result["outlier_indices"]
        assert "WARNING" in result["message"]

    def test_custom_threshold(self):
        spectrum = np.ones(50)
        spectrum[10] = 5.0  # mild spike
        # With a very high threshold it should NOT be flagged
        result_high = check_spectral_outliers(spectrum, threshold=100.0)
        assert result_high["has_outliers"] is False
        # With a very low threshold it SHOULD be flagged
        result_low = check_spectral_outliers(spectrum, threshold=0.1)
        assert result_low["has_outliers"] is True

    def test_returns_z_scores_array(self):
        spectrum = np.random.default_rng(0).uniform(0.1, 0.9, 64)
        result = check_spectral_outliers(spectrum)
        assert len(result["z_scores"]) == 64

    def test_empty_spectrum_raises(self):
        with pytest.raises(ValueError):
            check_spectral_outliers([])

    def test_list_input_accepted(self):
        spectrum = [0.5] * 20
        result = check_spectral_outliers(spectrum)
        assert isinstance(result["has_outliers"], bool)


# ---------------------------------------------------------------------------
# check_snr
# ---------------------------------------------------------------------------

class TestCheckSnr:
    def test_high_snr_is_acceptable(self):
        # Large constant signal, negligible noise
        spectrum = np.full(100, 0.8)
        spectrum[:5] += np.random.default_rng(1).normal(0, 0.001, 5)
        result = check_snr(spectrum)
        assert result["is_acceptable"] is True
        assert result["snr"] >= 10.0
        assert "OK" in result["message"]

    def test_low_snr_flagged(self):
        rng = np.random.default_rng(42)
        # Noisy spectrum: signal ≈ 0.05, noise std ≈ 0.05 → SNR ≈ 1
        spectrum = rng.normal(loc=0.05, scale=0.05, size=100)
        result = check_snr(spectrum)
        assert result["is_acceptable"] is False
        assert "WARNING" in result["message"]

    def test_explicit_indices(self):
        spectrum = np.linspace(0.1, 0.9, 100)
        result = check_snr(spectrum, noise_indices=list(range(5)), signal_indices=list(range(50, 90)))
        assert "snr" in result
        assert result["signal_mean"] > 0

    def test_zero_noise_returns_inf_snr(self):
        spectrum = np.full(50, 0.5)  # perfectly flat → std == 0
        result = check_snr(spectrum, noise_indices=[0, 1, 2])
        assert result["snr"] == float("inf")
        assert result["is_acceptable"] is True

    def test_empty_spectrum_raises(self):
        with pytest.raises(ValueError):
            check_snr([])


# ---------------------------------------------------------------------------
# check_prediction_range
# ---------------------------------------------------------------------------

class TestCheckPredictionRange:
    def test_valid_water_content(self):
        predictions = [0.0, 10.0, 25.5, 50.0, 100.0]
        result = check_prediction_range(predictions)
        assert result["is_valid"] is True
        assert result["out_of_range_indices"] == []
        assert "OK" in result["message"]

    def test_negative_value_flagged(self):
        """Soil water content cannot be negative."""
        predictions = [10.0, -1.5, 30.0]
        result = check_prediction_range(predictions)
        assert result["is_valid"] is False
        assert 1 in result["out_of_range_indices"]
        assert "WARNING" in result["message"]

    def test_above_100_flagged(self):
        """Soil water content cannot exceed 100 %."""
        predictions = [50.0, 105.0]
        result = check_prediction_range(predictions)
        assert result["is_valid"] is False
        assert 1 in result["out_of_range_indices"]

    def test_clipped_values_correct(self):
        predictions = [-5.0, 50.0, 120.0]
        result = check_prediction_range(predictions)
        np.testing.assert_array_equal(result["clipped"], [0.0, 50.0, 100.0])

    def test_single_scalar_valid(self):
        result = check_prediction_range(42.0)
        assert result["is_valid"] is True

    def test_single_scalar_invalid(self):
        result = check_prediction_range(-0.001)
        assert result["is_valid"] is False

    def test_custom_range(self):
        """Custom physically-valid range (e.g. 0–50 % for a specific soil type)."""
        predictions = [10.0, 60.0]
        result = check_prediction_range(predictions, min_val=0.0, max_val=50.0)
        assert result["is_valid"] is False
        assert 1 in result["out_of_range_indices"]

    def test_boundary_values_are_valid(self):
        result = check_prediction_range([0.0, 100.0])
        assert result["is_valid"] is True
