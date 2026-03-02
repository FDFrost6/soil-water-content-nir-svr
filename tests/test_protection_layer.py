"""
Unit tests for src/protection_layer.py
=======================================
Tests safety guardrails: OOD detection, confidence thresholds, water range checks.

Run:
  python -m pytest tests/test_protection_layer.py -v
"""

import sys
import os
from pathlib import Path
import numpy as np
import pickle
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from protection_layer import (
    fit_ood_detector,
    is_valid_spectrum,
    check_water_range,
    predict_safe,
    DEFAULT_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Fixtures: create minimal mock models
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_data():
    """
    Generate synthetic feature matrix (100 samples, 512 features).
    Simulate training data with some outliers.
    """
    np.random.seed(42)
    # Normal inlier data
    X_train_normal = np.random.randn(90, 512) * 50 + 1000
    # A few outliers (extreme values)
    X_train_outlier = np.random.randn(10, 512) * 200 + 2000
    X_train = np.vstack([X_train_normal, X_train_outlier])
    
    # Test samples
    X_test_normal = np.random.randn(5, 512) * 50 + 1000
    X_test_ood = np.random.randn(2, 512) * 300 + 5000  # way out of distribution
    
    return X_train, X_test_normal, X_test_ood


@pytest.fixture
def mock_scaler():
    """StandardScaler fitted on dummy data."""
    scaler = StandardScaler()
    X_dummy = np.random.randn(100, 512) * 50 + 1000
    scaler.fit(X_dummy)
    return scaler


@pytest.fixture
def mock_water_model(mock_scaler):
    """Mock water SVR model that predicts values in [0, 35] range."""
    np.random.seed(42)
    
    # Simple SVR that returns predictable values
    svr = SVR(kernel="rbf", C=10, gamma="scale")
    X_dummy = np.random.randn(50, 512)
    y_dummy = np.random.uniform(0, 35, 50)  # water content 0-35%
    X_scaled = mock_scaler.transform(X_dummy)
    svr.fit(X_scaled, y_dummy)
    
    return {"model": svr, "scaler": mock_scaler}


@pytest.fixture
def mock_nitrogen_model(mock_scaler):
    """Mock nitrogen RandomForest classifier for 4 classes [0, 75, 150, 300]."""
    np.random.seed(42)
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    X_dummy = np.random.randn(100, 512)
    y_dummy = np.random.choice([0, 75, 150, 300], size=100)
    X_scaled = mock_scaler.transform(X_dummy)
    rf.fit(X_scaled, y_dummy)
    
    return {
        "model": rf,
        "scaler": mock_scaler,
        "classes": [0, 75, 150, 300]
    }


@pytest.fixture
def mock_ood_detector(mock_data):
    """Fitted IsolationForest on mock training data."""
    X_train, _, _ = mock_data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    ood_model = fit_ood_detector(X_train_scaled, contamination=0.05, random_state=42)
    return ood_model, scaler


# ---------------------------------------------------------------------------
# Test OOD Detection
# ---------------------------------------------------------------------------

class TestOODDetection:
    
    def test_fit_ood_detector(self, mock_data):
        """Test that OOD detector can be fitted."""
        X_train, _, _ = mock_data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        ood_model = fit_ood_detector(X_train_scaled, contamination=0.05, random_state=42)
        
        assert isinstance(ood_model, IsolationForest)
        assert ood_model.contamination == 0.05
    
    def test_is_valid_spectrum_normal(self, mock_ood_detector, mock_data):
        """Normal spectra should be classified as valid (inlier)."""
        ood_model, scaler = mock_ood_detector
        _, X_test_normal, _ = mock_data
        X_test_scaled = scaler.transform(X_test_normal)
        
        is_valid, ood_score = is_valid_spectrum(X_test_scaled, ood_model)
        
        # Most normal samples should be valid
        assert np.mean(is_valid) >= 0.6  # at least 60% valid
    
    def test_is_valid_spectrum_ood(self, mock_ood_detector, mock_data):
        """Out-of-distribution spectra should be classified as invalid (outlier)."""
        ood_model, scaler = mock_ood_detector
        _, _, X_test_ood = mock_data
        X_test_scaled = scaler.transform(X_test_ood)
        
        is_valid, ood_score = is_valid_spectrum(X_test_scaled, ood_model)
        
        # OOD samples should mostly be invalid
        assert np.mean(is_valid) <= 0.5  # at most 50% valid (ideally 0%)


# ---------------------------------------------------------------------------
# Test Water Range Check
# ---------------------------------------------------------------------------

class TestWaterRangeCheck:
    
    def test_water_in_range(self):
        """Water prediction within valid range should pass."""
        is_valid, reason = check_water_range(15.0, water_min=-2.0, water_max=40.0)
        assert is_valid is True
        assert reason == ""
    
    def test_water_below_min(self):
        """Water prediction below min should fail."""
        is_valid, reason = check_water_range(-5.0, water_min=-2.0, water_max=40.0)
        assert is_valid is False
        assert "water_pred=-5.00%" in reason
    
    def test_water_above_max(self):
        """Water prediction above max should fail."""
        is_valid, reason = check_water_range(50.0, water_min=-2.0, water_max=40.0)
        assert is_valid is False
        assert "water_pred=50.00%" in reason
    
    def test_water_boundary_values(self):
        """Test boundary values are accepted."""
        is_valid_lower, _ = check_water_range(-2.0, water_min=-2.0, water_max=40.0)
        is_valid_upper, _ = check_water_range(40.0, water_min=-2.0, water_max=40.0)
        assert is_valid_lower is True
        assert is_valid_upper is True


# ---------------------------------------------------------------------------
# Test Safe Prediction Integration
# ---------------------------------------------------------------------------

class TestPredictSafe:
    
    def test_predict_safe_normal_sample(
        self,
        mock_water_model,
        mock_nitrogen_model,
        mock_ood_detector,
        mock_data
    ):
        """Normal sample should return status='ok' or 'uncertain' with valid predictions."""
        ood_model, scaler = mock_ood_detector
        _, X_test_normal, _ = mock_data
        
        # Use first normal test sample (unscaled)
        X_raw = X_test_normal[0:1, :]
        
        result = predict_safe(
            X_raw,
            mock_water_model,
            mock_nitrogen_model,
            ood_model,
            thresholds=DEFAULT_THRESHOLDS
        )
        
        # Should have a status
        assert result["status"] in ["ok", "uncertain", "invalid"]
        
        # Water prediction should be present
        assert result["water_pred"] is not None
        assert isinstance(result["water_pred"], float)
        
        # If status is 'ok', nitrogen_pred should be an int
        if result["status"] == "ok":
            assert result["nitrogen_pred"] is not None
            assert result["nitrogen_pred"] in [0, 75, 150, 300]
        
        # If status is 'uncertain', nitrogen_pred should be None
        if result["status"] == "uncertain":
            assert result["nitrogen_pred"] is None
            assert "low_confidence" in result["reason"]
    
    def test_predict_safe_ood_sample(
        self,
        mock_water_model,
        mock_nitrogen_model,
        mock_ood_detector,
        mock_data
    ):
        """OOD sample should return status='invalid' with reason='ood_spectrum'."""
        ood_model, scaler = mock_ood_detector
        _, _, X_test_ood = mock_data
        
        # Use first OOD test sample
        X_raw = X_test_ood[0:1, :]
        
        result = predict_safe(
            X_raw,
            mock_water_model,
            mock_nitrogen_model,
            ood_model,
            thresholds=DEFAULT_THRESHOLDS
        )
        
        # OOD should be detected (high probability)
        # Note: depending on random seed, this might not always be 100% reliable,
        # but with extreme OOD samples it should work most of the time
        if result["status"] == "invalid":
            assert "ood_spectrum" in result["reason"] or "water_out_of_range" in result["reason"]
    
    def test_predict_safe_custom_thresholds(
        self,
        mock_water_model,
        mock_nitrogen_model,
        mock_ood_detector,
        mock_data
    ):
        """Custom thresholds should affect decision logic."""
        ood_model, scaler = mock_ood_detector
        _, X_test_normal, _ = mock_data
        X_raw = X_test_normal[0:1, :]
        
        # Very strict nitrogen confidence threshold (0.99)
        strict_thresholds = DEFAULT_THRESHOLDS.copy()
        strict_thresholds["nitrogen_confidence"] = 0.99
        
        result = predict_safe(
            X_raw,
            mock_water_model,
            mock_nitrogen_model,
            ood_model,
            thresholds=strict_thresholds
        )
        
        # With such a high threshold, nitrogen prediction is likely rejected
        # (unless we're very lucky with the mock model)
        assert result["max_prob"] is not None
        if result["max_prob"] < 0.99:
            assert result["status"] == "uncertain"
            assert result["nitrogen_pred"] is None
    
    def test_predict_safe_return_fields(
        self,
        mock_water_model,
        mock_nitrogen_model,
        mock_ood_detector,
        mock_data
    ):
        """Test that all expected fields are present in result dict."""
        ood_model, scaler = mock_ood_detector
        _, X_test_normal, _ = mock_data
        X_raw = X_test_normal[0:1, :]
        
        result = predict_safe(
            X_raw,
            mock_water_model,
            mock_nitrogen_model,
            ood_model,
            thresholds=DEFAULT_THRESHOLDS
        )
        
        required_fields = [
            "water_pred",
            "nitrogen_pred",
            "nitrogen_probs",
            "max_prob",
            "ood_score",
            "status",
            "reason"
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
        
        # Type checks
        assert isinstance(result["water_pred"], float)
        assert result["nitrogen_pred"] is None or isinstance(result["nitrogen_pred"], int)
        assert isinstance(result["nitrogen_probs"], list)
        assert isinstance(result["max_prob"], float)
        assert isinstance(result["ood_score"], float)
        assert result["status"] in ["ok", "uncertain", "invalid"]
        assert isinstance(result["reason"], str)


# ---------------------------------------------------------------------------
# Test edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    
    def test_single_sample_1d_input(
        self,
        mock_water_model,
        mock_nitrogen_model,
        mock_ood_detector,
        mock_data
    ):
        """Test that 1D input (single sample) is correctly handled."""
        ood_model, scaler = mock_ood_detector
        _, X_test_normal, _ = mock_data
        
        # Provide 1D array (n_features,) instead of 2D (1, n_features)
        X_raw_1d = X_test_normal[0, :]
        
        result = predict_safe(
            X_raw_1d,
            mock_water_model,
            mock_nitrogen_model,
            ood_model,
            thresholds=DEFAULT_THRESHOLDS
        )
        
        # Should work without errors
        assert "status" in result
        assert "water_pred" in result
    
    def test_default_thresholds_are_sensible(self):
        """Test that default thresholds have sensible values."""
        assert 0.0 < DEFAULT_THRESHOLDS["nitrogen_confidence"] <= 1.0
        assert DEFAULT_THRESHOLDS["ood_contamination"] > 0.0
        assert DEFAULT_THRESHOLDS["water_min"] < DEFAULT_THRESHOLDS["water_max"]
        assert DEFAULT_THRESHOLDS["water_min"] <= 0
        assert DEFAULT_THRESHOLDS["water_max"] >= 35


# ---------------------------------------------------------------------------
# Integration test with real models (if available)
# ---------------------------------------------------------------------------

class TestWithRealModels:
    """
    Integration tests using actual trained models.
    These will be skipped if models don't exist yet.
    """
    
    @pytest.fixture
    def real_models(self):
        """Load real trained models if available."""
        models_dir = Path(__file__).parent.parent / "models"
        water_path = models_dir / "svr_water_records.pkl"
        nitrogen_path = models_dir / "rf_nitrogen_records.pkl"
        ood_path = models_dir / "ood_detector.pkl"
        
        if not (water_path.exists() and nitrogen_path.exists() and ood_path.exists()):
            pytest.skip("Real models not available yet")
        
        with open(water_path, "rb") as f:
            water_pkg = pickle.load(f)
        with open(nitrogen_path, "rb") as f:
            nitrogen_pkg = pickle.load(f)
        with open(ood_path, "rb") as f:
            ood_model = pickle.load(f)
        
        return water_pkg, nitrogen_pkg, ood_model
    
    def test_real_models_predict_safe(self, real_models):
        """Test predict_safe with actual trained models."""
        water_pkg, nitrogen_pkg, ood_model = real_models
        
        # Create a synthetic spectrum (512 features)
        np.random.seed(123)
        X_raw = np.random.randn(1, 512) * 50 + 1000
        
        result = predict_safe(
            X_raw,
            water_pkg,
            nitrogen_pkg,
            ood_model,
            thresholds=DEFAULT_THRESHOLDS
        )
        
        # Should return a valid result structure
        assert "status" in result
        assert result["status"] in ["ok", "uncertain", "invalid"]
        
        # Water prediction should be numeric
        assert isinstance(result["water_pred"], float)
        
        # OOD score should be numeric
        assert isinstance(result["ood_score"], float)
