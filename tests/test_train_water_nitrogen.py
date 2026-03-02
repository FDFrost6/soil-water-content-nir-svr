"""
Integration tests for src/train_water_nitrogen.py pipeline
============================================================
Tests data pipeline integrity, model training, and prediction sanity.

Run:
  venv/bin/python -m pytest tests/test_train_water_nitrogen.py -v
"""

import sys
import pickle
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RECORDS_CSV    = Path(__file__).parent.parent / "data" / "soil_spectral_data_records.csv"
WATER_PKL      = Path(__file__).parent.parent / "models" / "svr_water_records.pkl"
NITROGEN_PKL   = Path(__file__).parent.parent / "models" / "rf_nitrogen_records.pkl"

skip_no_csv    = pytest.mark.skipif(not RECORDS_CSV.exists(),  reason="Dataset CSV not found")
skip_no_models = pytest.mark.skipif(
    not (WATER_PKL.exists() and NITROGEN_PKL.exists()),
    reason="Trained model files not found",
)


# ---------------------------------------------------------------------------
# Helper: load & pivot (mirrors train script)
# ---------------------------------------------------------------------------

def load_and_pivot():
    df = pd.read_csv(RECORDS_CSV)
    pivot = df.pivot_table(
        index=["water_content_percent", "nitrogen_mg_kg", "replicate"],
        columns="wavelength_nm",
        values="reflectance",
        aggfunc="first",
    ).sort_index()
    X         = pivot.values
    y_water   = pivot.index.get_level_values("water_content_percent").to_numpy(dtype=float)
    y_nitrogen= pivot.index.get_level_values("nitrogen_mg_kg").to_numpy(dtype=int)
    return X, y_water, y_nitrogen


# ---------------------------------------------------------------------------
# Data pipeline tests
# ---------------------------------------------------------------------------

@skip_no_csv
class TestDataPipeline:
    @pytest.fixture(scope="class")
    def pivoted(self):
        return load_and_pivot()

    def test_pivot_shape(self, pivoted):
        X, y_water, y_nitrogen = pivoted
        n_samples, n_features = X.shape
        assert n_samples >= 300,  f"Expected >= 300 samples, got {n_samples}"
        assert n_features == 512, f"Expected 512 wavelength features, got {n_features}"

    def test_no_nans_in_X(self, pivoted):
        X, _, _ = pivoted
        assert not np.isnan(X).any(), "Feature matrix contains NaN values"

    def test_water_target_range(self, pivoted):
        _, y_water, _ = pivoted
        assert y_water.min() >= 0
        assert y_water.max() <= 35

    def test_nitrogen_classes(self, pivoted):
        _, _, y_nitrogen = pivoted
        assert set(y_nitrogen).issubset({0, 75, 150, 300})

    def test_scaler_fit(self, pivoted):
        X, _, _ = pivoted
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0),  1.0, atol=1e-10)

    def test_split_sizes(self, pivoted):
        from sklearn.model_selection import train_test_split
        X, y_water, y_nitrogen = pivoted
        strat = [f"{int(w)}_{int(n)}" for w, n in zip(y_water, y_nitrogen)]
        n = len(y_water)
        _, X_test = train_test_split(X, test_size=0.15, random_state=42, stratify=strat)
        assert abs(len(X_test) / n - 0.15) < 0.03, "Test split is not approximately 15%"


# ---------------------------------------------------------------------------
# Model training tests (lightweight synthetic data)
# ---------------------------------------------------------------------------

class TestModelTrainingSynthetic:
    """Train on tiny synthetic data to verify no crashes."""

    @pytest.fixture(scope="class")
    def synthetic_data(self):
        rng = np.random.default_rng(0)
        n, p = 60, 50
        X = rng.random((n, p))
        y_water    = rng.choice([5, 15, 25, 35], n).astype(float)
        y_nitrogen = rng.choice([0, 75, 150, 300], n)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        return X_s, y_water, y_nitrogen

    def test_svr_trains(self, synthetic_data):
        X_s, y_water, _ = synthetic_data
        svr = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)
        svr.fit(X_s, y_water)
        pred = svr.predict(X_s)
        assert pred.shape == y_water.shape
        assert not np.isnan(pred).any()

    def test_rf_trains(self, synthetic_data):
        X_s, _, y_nitrogen = synthetic_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0)
        rf.fit(X_s, y_nitrogen)
        pred = rf.predict(X_s)
        assert pred.shape == y_nitrogen.shape
        assert set(pred).issubset({0, 75, 150, 300})

    def test_svr_predictions_in_range(self, synthetic_data):
        X_s, y_water, _ = synthetic_data
        svr = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)
        svr.fit(X_s, y_water)
        pred = svr.predict(X_s)
        # Predictions should be somewhere near the training range (±10)
        assert pred.min() > y_water.min() - 10
        assert pred.max() < y_water.max() + 10

    def test_rf_feature_importances(self, synthetic_data):
        X_s, _, y_nitrogen = synthetic_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0)
        rf.fit(X_s, y_nitrogen)
        imp = rf.feature_importances_
        assert imp.shape == (X_s.shape[1],)
        assert abs(imp.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Saved model tests
# ---------------------------------------------------------------------------

@skip_no_models
class TestSavedModels:
    @pytest.fixture(scope="class")
    def water_pkg(self):
        with open(WATER_PKL, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def nitrogen_pkg(self):
        with open(NITROGEN_PKL, "rb") as f:
            return pickle.load(f)

    def test_water_pkg_keys(self, water_pkg):
        assert "model" in water_pkg
        assert "scaler" in water_pkg
        assert "wavelengths" in water_pkg

    def test_nitrogen_pkg_keys(self, nitrogen_pkg):
        assert "model" in nitrogen_pkg
        assert "classes" in nitrogen_pkg
        assert "wavelengths" in nitrogen_pkg

    def test_water_model_is_svr(self, water_pkg):
        assert isinstance(water_pkg["model"], SVR)

    def test_nitrogen_model_exists(self, nitrogen_pkg):
        # Could be RF or SVC
        assert hasattr(nitrogen_pkg["model"], "predict")

    def test_wavelength_count(self, water_pkg):
        assert len(water_pkg["wavelengths"]) == 512

    def test_nitrogen_classes(self, nitrogen_pkg):
        assert set(nitrogen_pkg["classes"]).issubset({0, 75, 150, 300})

    @skip_no_csv
    def test_water_model_r2_acceptable(self, water_pkg):
        """Water model R² on full dataset should be >= 0.70."""
        X, y_water, y_nitrogen = load_and_pivot()
        scaler = water_pkg["scaler"]
        svr    = water_pkg["model"]
        X_s    = scaler.transform(X)
        pred   = svr.predict(X_s)
        r2     = r2_score(y_water, pred)
        assert r2 >= 0.70, f"Water model R² too low: {r2:.4f}"

    @skip_no_csv
    def test_nitrogen_model_balanced_acc_acceptable(self, nitrogen_pkg):
        """Nitrogen model balanced accuracy on full dataset should be >= 0.70."""
        X, y_water, y_nitrogen = load_and_pivot()
        scaler = nitrogen_pkg["scaler"]
        clf    = nitrogen_pkg["model"]
        X_s    = scaler.transform(X)
        pred   = clf.predict(X_s)
        bal    = balanced_accuracy_score(y_nitrogen, pred)
        assert bal >= 0.70, f"Nitrogen balanced accuracy too low: {bal:.4f}"

    @skip_no_csv
    def test_predictions_no_nan(self, water_pkg, nitrogen_pkg):
        X, _, _ = load_and_pivot()
        scaler  = water_pkg["scaler"]
        X_s     = scaler.transform(X)
        w_pred  = water_pkg["model"].predict(X_s)
        n_pred  = nitrogen_pkg["model"].predict(X_s)
        assert not np.isnan(w_pred).any()
        assert not np.isnan(n_pred.astype(float)).any()
