"""
Phase 4: Safety / Guardrail Layer for NIR Predictions
======================================================
Provides Out-of-Distribution detection, confidence thresholds, and water-based
guardrails to prevent unreliable predictions on poor-quality or unusual spectra.

Usage:
  # During training:
  ood_detector = fit_ood_detector(X_train_scaled)
  
  # During prediction:
  result = predict_safe(X_raw, water_pkg, nitrogen_pkg, ood_detector, thresholds)

Returns dict with status flags ("ok"|"uncertain"|"invalid") and reason codes.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Default thresholds (tunable)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "nitrogen_confidence": 0.60,      # min max_prob for nitrogen prediction
    "ood_contamination": 0.03,        # expected outlier rate in training (3%)
    "water_min": -2.0,                # lower bound for valid water prediction (%)
    "water_max": 40.0,                # upper bound for valid water prediction (%)
}


# ---------------------------------------------------------------------------
# 1) Out-of-Distribution Detection using IsolationForest
# ---------------------------------------------------------------------------

def fit_ood_detector(
    X_train_scaled: np.ndarray,
    contamination: float = 0.03,
    random_state: int = 42
) -> IsolationForest:
    """
    Fit an IsolationForest on the training set to detect out-of-distribution spectra.
    
    Design rationale:
    - IsolationForest is robust, unsupervised, and works well in high dimensions
    - Computationally efficient (O(n log n) training, O(log n) prediction)
    - Preferred over PCA+Mahalanobis for spectral data with potential non-Gaussian structure
    
    Parameters
    ----------
    X_train_scaled : np.ndarray of shape (n_train, n_features)
        Scaled training feature matrix (after StandardScaler).
    contamination : float, default=0.03
        Expected proportion of outliers in training set. 
        3% is reasonable for clean lab data with occasional artifacts.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    ood_model : IsolationForest
        Fitted OOD detector. Predict -1 for outliers, 1 for inliers.
    """
    print(f"\n{'='*70}")
    print("FITTING OOD DETECTOR (IsolationForest)")
    print(f"{'='*70}")
    print(f"  Training samples  : {X_train_scaled.shape[0]}")
    print(f"  Features          : {X_train_scaled.shape[1]}")
    print(f"  Contamination     : {contamination*100:.1f}%")
    
    ood_model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples='auto',
        n_jobs=-1,
    )
    ood_model.fit(X_train_scaled)
    
    # Check inlier rate on training set
    train_pred = ood_model.predict(X_train_scaled)
    inlier_rate = np.mean(train_pred == 1)
    print(f"  Train inlier rate : {inlier_rate*100:.1f}%")
    print(f"  OOD detector fitted successfully.")
    
    return ood_model


def is_valid_spectrum(
    X_scaled: np.ndarray,
    ood_model: IsolationForest
) -> Tuple[bool, float]:
    """
    Check if spectrum(s) are in-distribution.
    
    Parameters
    ----------
    X_scaled : np.ndarray of shape (n_samples, n_features)
        Scaled feature matrix.
    ood_model : IsolationForest
        Fitted OOD detector.
    
    Returns
    -------
    is_valid : bool or np.ndarray of bool
        True if inlier (valid), False if outlier (OOD).
    ood_score : float or np.ndarray of float
        Anomaly score. Lower = more anomalous (outlier).
        Negative values indicate outliers, positive indicate inliers.
    """
    predictions = ood_model.predict(X_scaled)  # 1 = inlier, -1 = outlier
    scores = ood_model.decision_function(X_scaled)  # higher = more normal
    
    is_valid = predictions == 1
    
    if len(X_scaled) == 1:
        return bool(is_valid[0]), float(scores[0])
    else:
        return is_valid, scores


# ---------------------------------------------------------------------------
# 2) Water-based Guardrail
# ---------------------------------------------------------------------------

def check_water_range(
    water_pred: float,
    water_min: float = -2.0,
    water_max: float = 40.0
) -> Tuple[bool, str]:
    """
    Check if predicted water content is within acceptable range.
    
    Training range: 0-35%, but we allow slight tolerance (-2 to 40%)
    to account for extrapolation uncertainty.
    
    Parameters
    ----------
    water_pred : float
        Predicted water content (%).
    water_min : float, default=-2.0
        Lower bound (%).
    water_max : float, default=40.0
        Upper bound (%).
    
    Returns
    -------
    is_valid : bool
        True if within range.
    reason : str
        Empty string if valid, else explanation.
    """
    if water_pred < water_min:
        return False, f"water_pred={water_pred:.2f}% < {water_min}%"
    if water_pred > water_max:
        return False, f"water_pred={water_pred:.2f}% > {water_max}%"
    return True, ""


# ---------------------------------------------------------------------------
# 3) Unified Safe Prediction Function
# ---------------------------------------------------------------------------

def predict_safe(
    X_raw: np.ndarray,
    water_pkg: Dict[str, Any],
    nitrogen_pkg: Dict[str, Any],
    ood_model: IsolationForest,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Perform safe prediction with guardrails for water + nitrogen.
    
    Protection layers (in order):
    1. OOD detection: reject anomalous spectra
    2. Water range check: reject if water prediction out of bounds
    3. Nitrogen confidence: reject low-confidence nitrogen predictions
    
    Parameters
    ----------
    X_raw : np.ndarray of shape (1, n_features) or (n_features,)
        Raw (unscaled) feature vector for a single sample.
    water_pkg : dict
        Water model package with keys 'model' (SVR) and 'scaler' (StandardScaler).
    nitrogen_pkg : dict
        Nitrogen model package with keys 'model' (RF), 'scaler', 'classes'.
    ood_model : IsolationForest
        Fitted OOD detector.
    thresholds : dict, optional
        Custom thresholds. If None, uses DEFAULT_THRESHOLDS.
    
    Returns
    -------
    result : dict
        {
          "water_pred": float,
          "nitrogen_pred": int | None,
          "nitrogen_probs": list of float,
          "max_prob": float,
          "ood_score": float,
          "status": "ok" | "uncertain" | "invalid",
          "reason": str
        }
        
        Status codes:
        - "ok": all checks passed, predictions are reliable
        - "uncertain": nitrogen prediction has low confidence
        - "invalid": spectrum is OOD or water is out of range
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    # Ensure X_raw is 2D
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(1, -1)
    
    # Extract models and scalers
    water_scaler = water_pkg["scaler"]
    water_model = water_pkg["model"]
    nitrogen_scaler = nitrogen_pkg["scaler"]
    nitrogen_model = nitrogen_pkg["model"]
    nitrogen_classes = nitrogen_pkg["classes"]
    
    # Scale features
    X_scaled_water = water_scaler.transform(X_raw)
    X_scaled_nitrogen = nitrogen_scaler.transform(X_raw)
    
    # Initialize result dict
    result = {
        "water_pred": None,
        "nitrogen_pred": None,
        "nitrogen_probs": None,
        "max_prob": None,
        "ood_score": None,
        "status": "ok",
        "reason": "none"
    }
    
    # ----------------------------------------------------------------------
    # Check 1: OOD Detection
    # ----------------------------------------------------------------------
    is_valid, ood_score = is_valid_spectrum(X_scaled_nitrogen, ood_model)
    result["ood_score"] = float(ood_score)
    
    if not is_valid:
        result["status"] = "invalid"
        result["reason"] = "ood_spectrum"
        return result
    
    # ----------------------------------------------------------------------
    # Check 2: Water Prediction & Range
    # ----------------------------------------------------------------------
    water_pred = float(water_model.predict(X_scaled_water)[0])
    result["water_pred"] = water_pred
    
    water_valid, water_reason = check_water_range(
        water_pred,
        thresholds["water_min"],
        thresholds["water_max"]
    )
    
    if not water_valid:
        result["status"] = "invalid"
        result["reason"] = f"water_out_of_range ({water_reason})"
        return result
    
    # ----------------------------------------------------------------------
    # Check 3: Nitrogen Prediction with Confidence Threshold
    # ----------------------------------------------------------------------
    nitrogen_probs = nitrogen_model.predict_proba(X_scaled_nitrogen)[0]
    max_prob = float(np.max(nitrogen_probs))
    result["nitrogen_probs"] = nitrogen_probs.tolist()
    result["max_prob"] = max_prob
    
    if max_prob < thresholds["nitrogen_confidence"]:
        result["status"] = "uncertain"
        result["reason"] = f"low_confidence (max_prob={max_prob:.3f})"
        result["nitrogen_pred"] = None
    else:
        nitrogen_pred_idx = int(np.argmax(nitrogen_probs))
        result["nitrogen_pred"] = int(nitrogen_classes[nitrogen_pred_idx])
    
    return result


# ---------------------------------------------------------------------------
# 4) Batch Prediction (for evaluation)
# ---------------------------------------------------------------------------

def predict_safe_batch(
    X_raw: np.ndarray,
    water_pkg: Dict[str, Any],
    nitrogen_pkg: Dict[str, Any],
    ood_model: IsolationForest,
    thresholds: Optional[Dict[str, float]] = None
) -> list[Dict[str, Any]]:
    """
    Apply predict_safe to multiple samples.
    
    Parameters
    ----------
    X_raw : np.ndarray of shape (n_samples, n_features)
    water_pkg, nitrogen_pkg, ood_model : as in predict_safe
    thresholds : dict, optional
    
    Returns
    -------
    results : list of dict
        One result dict per sample.
    """
    results = []
    for i in range(len(X_raw)):
        result = predict_safe(
            X_raw[i:i+1, :],
            water_pkg,
            nitrogen_pkg,
            ood_model,
            thresholds
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Helper: Load all models for convenient prediction
# ---------------------------------------------------------------------------

def load_models(
    water_path: Path = Path("models/svr_water_records.pkl"),
    nitrogen_path: Path = Path("models/rf_nitrogen_records.pkl"),
    ood_path: Path = Path("models/ood_detector.pkl")
) -> Tuple[Dict, Dict, IsolationForest]:
    """
    Load all trained models for safe prediction.
    
    Returns
    -------
    water_pkg : dict
    nitrogen_pkg : dict
    ood_model : IsolationForest
    """
    with open(water_path, "rb") as f:
        water_pkg = pickle.load(f)
    with open(nitrogen_path, "rb") as f:
        nitrogen_pkg = pickle.load(f)
    with open(ood_path, "rb") as f:
        ood_model = pickle.load(f)
    return water_pkg, nitrogen_pkg, ood_model
