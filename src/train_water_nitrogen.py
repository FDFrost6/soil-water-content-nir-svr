"""
Phase 2: Water Content Regression + Nitrogen Classification
============================================================
Trains two models from data/soil_spectral_data_records.csv:
  1. Water SVR (RBF kernel) — continuous regression of water_content_percent
  2. Nitrogen RandomForest classifier — step detection of nitrogen_mg_kg class

Outputs:
  models/svr_water_records.pkl   — water SVR model + scaler
  models/rf_nitrogen_records.pkl — nitrogen RF model + scaler
  results/training_summary.txt   — printed metrics summary

Usage:
  python src/train_water_nitrogen.py
  (from project root)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, accuracy_score,
    balanced_accuracy_score,
)
import warnings
warnings.filterwarnings("ignore")

# Import protection layer for OOD detection
import sys
sys.path.insert(0, str(Path(__file__).parent))
from protection_layer import fit_ood_detector

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_CSV = Path("data/soil_spectral_data_records.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading & pivoting
# ---------------------------------------------------------------------------

def load_and_pivot(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Load long-format CSV, pivot to feature matrix.

    Returns
    -------
    X            : (n_samples, n_wavelengths) reflectance matrix
    y_water      : (n_samples,) water content percent
    y_nitrogen   : (n_samples,) nitrogen mg/kg
    wavelengths  : list of wavelength values (feature names)
    """
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}   Unique measurements: "
          f"{df.groupby(['water_content_percent', 'nitrogen_mg_kg', 'replicate']).ngroups}")

    # Pivot: rows = (water, nitrogen, replicate), cols = wavelengths
    pivot = df.pivot_table(
        index=["water_content_percent", "nitrogen_mg_kg", "replicate"],
        columns="wavelength_nm",
        values="reflectance",
        aggfunc="first",
    )
    pivot = pivot.sort_index()

    wavelengths = list(pivot.columns)
    X = pivot.values
    y_water = pivot.index.get_level_values("water_content_percent").to_numpy(dtype=float)
    y_nitrogen = pivot.index.get_level_values("nitrogen_mg_kg").to_numpy(dtype=int)

    print(f"  Feature matrix: {X.shape}  (samples × wavelengths)")
    return X, y_water, y_nitrogen, wavelengths


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred, label: str) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    tol5 = np.mean(np.abs(y_true - y_pred) <= 5.0) * 100
    tol2 = np.mean(np.abs(y_true - y_pred) <= 2.0) * 100
    print(f"\n  [{label}]")
    print(f"    R²   : {r2:.4f}")
    print(f"    RMSE : {rmse:.4f}%")
    print(f"    MAE  : {mae:.4f}%")
    print(f"    Acc ±5% : {tol5:.1f}%   Acc ±2% : {tol2:.1f}%")
    return dict(label=label, r2=r2, rmse=rmse, mae=mae, tol5=tol5, tol2=tol2)


def classification_metrics(y_true, y_pred, label: str, classes) -> dict:
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    print(f"\n  [{label}]")
    print(f"    Accuracy         : {acc*100:.2f}%")
    print(f"    Balanced Accuracy: {bal*100:.2f}%")
    print(f"\n{classification_report(y_true, y_pred, target_names=[str(c) for c in classes], zero_division=0)}")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return dict(label=label, accuracy=acc, balanced_accuracy=bal, confusion_matrix=cm)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("WATER + NITROGEN MODEL TRAINING")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Load data -----------------------------------------------------------
    X, y_water, y_nitrogen, wavelengths = load_and_pivot(DATA_CSV)
    n_samples = len(y_water)

    print(f"\n  Water levels (%)  : {sorted(np.unique(y_water).astype(int).tolist())}")
    print(f"  Nitrogen classes  : {sorted(np.unique(y_nitrogen).tolist())} mg/kg")
    print(f"  Total samples     : {n_samples}")

    # 2. Stratified split (stratify on combined label) -----------------------
    print("\n" + "=" * 70)
    print("SPLITTING DATA  (70% train / 15% val / 15% test)")
    print("=" * 70)

    # Combined stratification key
    strat_key = [f"{int(w)}_{int(n)}" for w, n in zip(y_water, y_nitrogen)]

    X_temp, X_test, yw_temp, yw_test, yn_temp, yn_test, sk_temp, _ = train_test_split(
        X, y_water, y_nitrogen, strat_key,
        test_size=0.15, random_state=42, stratify=strat_key,
    )
    # From remaining 85%, take 15/85 ≈ 17.6% for val → ~15% of total
    strat_temp = [f"{int(w)}_{int(n)}" for w, n in zip(yw_temp, yn_temp)]
    X_train, X_val, yw_train, yw_val, yn_train, yn_val = train_test_split(
        X_temp, yw_temp, yn_temp,
        test_size=0.176, random_state=42, stratify=strat_temp,
    )

    print(f"  Train : {len(yw_train):>4} samples ({len(yw_train)/n_samples*100:.1f}%)")
    print(f"  Val   : {len(yw_val):>4} samples ({len(yw_val)/n_samples*100:.1f}%)")
    print(f"  Test  : {len(yw_test):>4} samples ({len(yw_test)/n_samples*100:.1f}%)")

    # 3. Feature scaling -----------------------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    print("\n  ✓ Features standardized (zero mean, unit variance)")

    # 3B. OOD Detection (Protection Layer) -----------------------------------
    print("\n" + "=" * 70)
    print("PHASE 4: PROTECTION LAYER — OOD DETECTION")
    print("=" * 70)
    ood_detector = fit_ood_detector(
        X_train_s,
        contamination=0.03,  # expect 3% outliers in clean training data
        random_state=42
    )
    
    # Test OOD detector on validation and test sets
    ood_val_pred = ood_detector.predict(X_val_s)
    ood_test_pred = ood_detector.predict(X_test_s)
    print(f"  Val  inlier rate : {np.mean(ood_val_pred == 1)*100:.1f}%")
    print(f"  Test inlier rate : {np.mean(ood_test_pred == 1)*100:.1f}%")
    
    # Save OOD detector
    ood_path = MODELS_DIR / "ood_detector.pkl"
    with open(ood_path, "wb") as f:
        pickle.dump(ood_detector, f)
    print(f"  ✓ Saved → {ood_path}")

    # =========================================================================
    # 2A: WATER SVR
    # =========================================================================
    print("\n" + "=" * 70)
    print("2A: WATER CONTENT REGRESSION — SVR (RBF, C=100, ε=0.1)")
    print("=" * 70)

    svr = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
    print("  Training SVR …")
    svr.fit(X_train_s, yw_train)
    print(f"  ✓ Trained — {len(svr.support_)} support vectors")

    print("\n  Performance:")
    svr_train = regression_metrics(yw_train, svr.predict(X_train_s), "Train")
    svr_val   = regression_metrics(yw_val,   svr.predict(X_val_s),   "Val  ")
    svr_test  = regression_metrics(yw_test,  svr.predict(X_test_s),  "Test ")

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Use water level bins for stratification in CV
    water_bins = pd.cut(yw_train, bins=5, labels=False)
    cv_r2 = cross_val_score(svr, X_train_s, yw_train, cv=5, scoring="r2")
    print(f"\n  5-fold CV R² (train): {cv_r2.mean():.4f} ± {cv_r2.std()*2:.4f}")

    # Save water model
    water_pkg = {"model": svr, "scaler": scaler, "wavelengths": wavelengths}
    with open(MODELS_DIR / "svr_water_records.pkl", "wb") as f:
        pickle.dump(water_pkg, f)
    print("  ✓ Saved → models/svr_water_records.pkl")

    # =========================================================================
    # 2B: NITROGEN CLASSIFIER
    # =========================================================================
    print("\n" + "=" * 70)
    print("2B: NITROGEN CLASSIFICATION — RandomForest + SVC comparison")
    print("=" * 70)

    nitrogen_classes = sorted(np.unique(y_nitrogen).tolist())
    print(f"  Classes: {nitrogen_classes} mg/kg\n")

    # --- RandomForest --------------------------------------------------------
    print("  Training RandomForest …")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, yn_train)
    print(f"  ✓ Trained — {rf.n_estimators} trees")

    print("\n  RandomForest Performance:")
    rf_train = classification_metrics(yn_train, rf.predict(X_train_s), "Train [RF]", nitrogen_classes)
    rf_val   = classification_metrics(yn_val,   rf.predict(X_val_s),   "Val   [RF]", nitrogen_classes)
    rf_test  = classification_metrics(yn_test,  rf.predict(X_test_s),  "Test  [RF]", nitrogen_classes)

    # Feature importance: top 10 wavelengths
    importances = rf.feature_importances_
    top10_idx = np.argsort(importances)[::-1][:10]
    print("  Top-10 important wavelengths (RF):")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"    {rank:2d}. {wavelengths[idx]:.1f} nm  (importance={importances[idx]:.4f})")

    # --- SVC -----------------------------------------------------------------
    print("\n  Training SVC (RBF, C=10) …")
    svc = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42)
    svc.fit(X_train_s, yn_train)
    print("  ✓ Trained SVC")

    print("\n  SVC Performance:")
    svc_train = classification_metrics(yn_train, svc.predict(X_train_s), "Train [SVC]", nitrogen_classes)
    svc_val   = classification_metrics(yn_val,   svc.predict(X_val_s),   "Val   [SVC]", nitrogen_classes)
    svc_test  = classification_metrics(yn_test,  svc.predict(X_test_s),  "Test  [SVC]", nitrogen_classes)

    # Pick best classifier by val balanced accuracy
    if rf_val["balanced_accuracy"] >= svc_val["balanced_accuracy"]:
        best_clf = rf
        best_clf_name = "RandomForest"
        best_test = rf_test
    else:
        best_clf = svc
        best_clf_name = "SVC"
        best_test = svc_test

    print(f"\n  → Best nitrogen classifier: {best_clf_name}")

    # Cross-validation for best classifier
    cv_acc = cross_val_score(best_clf, X_train_s, yn_train, cv=5, scoring="balanced_accuracy")
    print(f"  5-fold CV Balanced Accuracy ({best_clf_name}): "
          f"{cv_acc.mean()*100:.2f}% ± {cv_acc.std()*200:.2f}%")

    # Save nitrogen model
    nitrogen_pkg = {
        "model": best_clf,
        "model_name": best_clf_name,
        "scaler": scaler,
        "wavelengths": wavelengths,
        "classes": nitrogen_classes,
        "rf": rf,
        "svc": svc,
    }
    with open(MODELS_DIR / "rf_nitrogen_records.pkl", "wb") as f:
        pickle.dump(nitrogen_pkg, f)
    print(f"  ✓ Saved → models/rf_nitrogen_records.pkl")

    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  Water SVR (Test) :")
    print(f"    R²   = {svr_test['r2']:.4f}")
    print(f"    RMSE = {svr_test['rmse']:.3f}%")
    print(f"    MAE  = {svr_test['mae']:.3f}%")
    print(f"    Acc ±5% = {svr_test['tol5']:.1f}%")

    water_grade = (
        "EXCELLENT" if svr_test["r2"] > 0.95 else
        "VERY GOOD" if svr_test["r2"] > 0.90 else
        "GOOD"      if svr_test["r2"] > 0.85 else
        "MODERATE"  if svr_test["r2"] > 0.70 else
        "NEEDS IMPROVEMENT"
    )
    print(f"    → {water_grade}")

    print(f"\n  Nitrogen {best_clf_name} (Test) :")
    print(f"    Accuracy          = {best_test['accuracy']*100:.2f}%")
    print(f"    Balanced Accuracy = {best_test['balanced_accuracy']*100:.2f}%")
    nitrogen_grade = (
        "STEP DETECTION: YES ✓" if best_test["balanced_accuracy"] >= 0.70 else
        "STEP DETECTION: PARTIAL ⚠" if best_test["balanced_accuracy"] >= 0.50 else
        "STEP DETECTION: NO ✗"
    )
    print(f"    → {nitrogen_grade}")

    # Write summary to file
    summary_lines = [
        "WATER + NITROGEN MODEL TRAINING SUMMARY",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Water SVR  — Test R²={svr_test['r2']:.4f}, RMSE={svr_test['rmse']:.3f}%, MAE={svr_test['mae']:.3f}%",
        f"            Acc ±5%={svr_test['tol5']:.1f}%  → {water_grade}",
        "",
        f"Nitrogen ({best_clf_name}) — Test Balanced Acc={best_test['balanced_accuracy']*100:.2f}%",
        f"            → {nitrogen_grade}",
        "",
        "Confusion matrix (test):",
    ]
    cm = best_test["confusion_matrix"]
    header = "  " + "  ".join(f"{c:>5}" for c in nitrogen_classes)
    summary_lines.append(header)
    for i, row in enumerate(cm):
        summary_lines.append(f"{nitrogen_classes[i]:3d}" + "  ".join(f"{v:>5}" for v in row))

    summary_path = RESULTS_DIR / "training_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\n  ✓ Summary saved → {summary_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
