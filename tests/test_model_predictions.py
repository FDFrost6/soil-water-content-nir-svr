"""
Tests verifying that model predictions stay within physically possible ranges
for soil water content (0 – 100 %).

These tests load the trained SVR model from models/ and run predictions on
representative spectral inputs to confirm the guardrails catch invalid outputs.
"""

import sys
import os
import pickle
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from guardrails import check_prediction_range

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "svm_water_content_model.pkl"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model():
    """Load the trained SVR model and scaler. Skip if not present."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Trained model not found – run src/train_svm_individual.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Physical range tests (no model needed)
# ---------------------------------------------------------------------------

class TestPhysicallyPossibleRange:
    """Verify that check_prediction_range correctly enforces physical bounds."""

    def test_water_content_cannot_be_negative(self):
        """A negative water content is physically impossible."""
        result = check_prediction_range(-5.0)
        assert result["is_valid"] is False, "Negative water content must be flagged."

    def test_water_content_cannot_exceed_100(self):
        """Water content above 100 % is physically impossible."""
        result = check_prediction_range(105.0)
        assert result["is_valid"] is False, "Water content > 100 % must be flagged."

    def test_typical_range_is_valid(self):
        typical = np.arange(0, 101, 5, dtype=float)
        result = check_prediction_range(typical)
        assert result["is_valid"] is True

    def test_mixed_batch_flags_only_invalid(self):
        predictions = [10.0, -2.0, 50.0, 110.0, 0.0, 100.0]
        result = check_prediction_range(predictions)
        assert not result["is_valid"]
        assert 1 in result["out_of_range_indices"]  # -2.0
        assert 3 in result["out_of_range_indices"]  # 110.0
        # Valid values must NOT be flagged
        assert 0 not in result["out_of_range_indices"]
        assert 2 not in result["out_of_range_indices"]
        assert 4 not in result["out_of_range_indices"]
        assert 5 not in result["out_of_range_indices"]

    def test_clipped_values_respect_bounds(self):
        predictions = [-10.0, 50.0, 150.0]
        result = check_prediction_range(predictions)
        assert result["clipped"][0] == 0.0
        assert result["clipped"][1] == 50.0
        assert result["clipped"][2] == 100.0


# ---------------------------------------------------------------------------
# Model integration tests (requires trained model)
# ---------------------------------------------------------------------------

class TestModelPredictionsInRange:
    """Run real model predictions and verify outputs are in physical range."""

    def test_predictions_on_typical_spectra(self):
        """Predictions on 'well-behaved' spectra should be within [0, 100]."""
        payload = load_model()
        model = payload["model"]
        scaler = payload["scaler"]

        # Build synthetic spectra that resemble typical soil reflectance (0.1–0.9)
        rng = np.random.default_rng(0)
        n_features = scaler.mean_.shape[0]
        spectra = rng.uniform(0.1, 0.9, size=(20, n_features))

        X_scaled = scaler.transform(spectra)
        predictions = model.predict(X_scaled)

        result = check_prediction_range(predictions)
        assert result["is_valid"], (
            f"Model produced out-of-range predictions: "
            f"{predictions[result['out_of_range_indices']]}"
        )

    def test_predictions_are_finite(self):
        """Model must not output NaN or infinite values."""
        payload = load_model()
        model = payload["model"]
        scaler = payload["scaler"]

        rng = np.random.default_rng(1)
        n_features = scaler.mean_.shape[0]
        spectra = rng.uniform(0.0, 1.0, size=(10, n_features))

        X_scaled = scaler.transform(spectra)
        predictions = model.predict(X_scaled)

        assert np.all(np.isfinite(predictions)), "Model returned NaN or Inf values."
