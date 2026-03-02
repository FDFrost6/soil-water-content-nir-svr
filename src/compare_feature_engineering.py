"""
Feature Engineering Comparison for Nitrogen Classification
===========================================================
Compares 4 feature configurations:
  1. BASELINE: Raw spectral features (512 wavelengths)
  2. WATER+SPECTRAL: Spectral + predicted water content (513 features)
  3. PCA: PCA-reduced spectral features (50 components)
  4. PCA+WATER: PCA features + predicted water (51 features)

Outputs:
  - Comparison table (val + test metrics)
  - Confusion matrices for each variant
  - Best model saved to models/rf_nitrogen_best.pkl

Usage:
  python src/compare_feature_engineering.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    balanced_accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_CSV = Path("data/soil_spectral_data_records.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading (reuse from train_water_nitrogen.py)
# ---------------------------------------------------------------------------

def load_and_pivot(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Load long-format CSV, pivot to feature matrix."""
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}   Unique measurements: "
          f"{df.groupby(['water_content_percent', 'nitrogen_mg_kg', 'replicate']).ngroups}")

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
# Metrics helper
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, label: str, classes) -> dict:
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    return {
        "label": label,
        "accuracy": acc,
        "balanced_accuracy": bal,
        "confusion_matrix": cm
    }


# ---------------------------------------------------------------------------
# Feature Engineering Functions
# ---------------------------------------------------------------------------

def build_baseline_features(X_train, X_val, X_test):
    """Baseline: raw spectral features with StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler, None


def build_water_features(X_train, X_val, X_test, yw_train, yw_val, yw_test, water_model, water_scaler):
    """Add predicted water content as an additional feature."""
    # Scale spectral features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Predict water content
    X_train_water_scaled = water_scaler.transform(X_train)
    X_val_water_scaled = water_scaler.transform(X_val)
    X_test_water_scaled = water_scaler.transform(X_test)
    
    water_pred_train = water_model.predict(X_train_water_scaled).reshape(-1, 1)
    water_pred_val = water_model.predict(X_val_water_scaled).reshape(-1, 1)
    water_pred_test = water_model.predict(X_test_water_scaled).reshape(-1, 1)
    
    # Concatenate: [spectral_features, water_prediction]
    X_train_aug = np.hstack([X_train_scaled, water_pred_train])
    X_val_aug = np.hstack([X_val_scaled, water_pred_val])
    X_test_aug = np.hstack([X_test_scaled, water_pred_test])
    
    return X_train_aug, X_val_aug, X_test_aug, scaler, None


def build_pca_features(X_train, X_val, X_test, n_components=50):
    """PCA-reduced spectral features."""
    # Scale first
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"    PCA: {n_components} components explain {explained_var*100:.2f}% variance")
    
    return X_train_pca, X_val_pca, X_test_pca, scaler, pca


def build_pca_water_features(X_train, X_val, X_test, yw_train, yw_val, yw_test, 
                              water_model, water_scaler, n_components=50):
    """PCA features + predicted water content."""
    # Get PCA features
    X_train_pca, X_val_pca, X_test_pca, scaler, pca = build_pca_features(
        X_train, X_val, X_test, n_components
    )
    
    # Predict water
    X_train_water_scaled = water_scaler.transform(X_train)
    X_val_water_scaled = water_scaler.transform(X_val)
    X_test_water_scaled = water_scaler.transform(X_test)
    
    water_pred_train = water_model.predict(X_train_water_scaled).reshape(-1, 1)
    water_pred_val = water_model.predict(X_val_water_scaled).reshape(-1, 1)
    water_pred_test = water_model.predict(X_test_water_scaled).reshape(-1, 1)
    
    # Concatenate
    X_train_aug = np.hstack([X_train_pca, water_pred_train])
    X_val_aug = np.hstack([X_val_pca, water_pred_val])
    X_test_aug = np.hstack([X_test_pca, water_pred_test])
    
    return X_train_aug, X_val_aug, X_test_aug, scaler, pca


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_variant(
    X_train, X_val, X_test,
    yn_train, yn_val, yn_test,
    variant_name: str,
    classes: list
):
    """Train RandomForest and evaluate on val + test."""
    print(f"\n{'='*70}")
    print(f"VARIANT: {variant_name}")
    print(f"{'='*70}")
    print(f"  Feature dimensions: {X_train.shape[1]}")
    
    # Train RandomForest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, yn_train)
    
    # Predictions
    yn_pred_train = rf.predict(X_train)
    yn_pred_val = rf.predict(X_val)
    yn_pred_test = rf.predict(X_test)
    
    # Metrics
    metrics_train = classification_metrics(yn_train, yn_pred_train, "Train", classes)
    metrics_val = classification_metrics(yn_val, yn_pred_val, "Val", classes)
    metrics_test = classification_metrics(yn_test, yn_pred_test, "Test", classes)
    
    print(f"  Train: Acc={metrics_train['accuracy']*100:.2f}% | Bal={metrics_train['balanced_accuracy']*100:.2f}%")
    print(f"  Val  : Acc={metrics_val['accuracy']*100:.2f}% | Bal={metrics_val['balanced_accuracy']*100:.2f}%")
    print(f"  Test : Acc={metrics_test['accuracy']*100:.2f}% | Bal={metrics_test['balanced_accuracy']*100:.2f}%")
    
    # Cross-validation on training set
    cv_scores = cross_val_score(rf, X_train, yn_train, cv=5, scoring="balanced_accuracy")
    print(f"  5-fold CV Balanced Acc: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*200:.2f}%")
    
    return {
        "variant": variant_name,
        "model": rf,
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrices(results_list, classes, savepath: Path):
    """Plot confusion matrices for all variants in a grid."""
    n_variants = len(results_list)
    fig, axes = plt.subplots(1, n_variants, figsize=(5*n_variants, 5))
    if n_variants == 1:
        axes = [axes]
    
    for i, result in enumerate(results_list):
        ax = axes[i]
        cm = result["metrics_test"]["confusion_matrix"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        
        tick_labels = [str(c) for c in classes]
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)
        
        for ii in range(len(classes)):
            for jj in range(len(classes)):
                color = "white" if cm_norm[ii, jj] > 0.6 else "black"
                ax.text(jj, ii, f"{cm[ii,jj]}\n({cm_norm[ii,jj]*100:.0f}%)",
                        ha="center", va="center", fontsize=8, color=color, fontweight="bold")
        
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True" if i == 0 else "", fontsize=10)
        
        variant_name = result["variant"]
        bal_acc = result["metrics_test"]["balanced_accuracy"]
        ax.set_title(f"{variant_name}\nBal Acc = {bal_acc*100:.1f}%", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Confusion matrices saved → {savepath}")


def plot_performance_comparison(results_list, savepath: Path):
    """Bar plot comparing balanced accuracy across variants."""
    variants = [r["variant"] for r in results_list]
    val_acc = [r["metrics_val"]["balanced_accuracy"] * 100 for r in results_list]
    test_acc = [r["metrics_test"]["balanced_accuracy"] * 100 for r in results_list]
    
    x = np.arange(len(variants))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, val_acc, width, label="Validation", color="#4e79a7", alpha=0.8)
    bars2 = ax.bar(x + width/2, test_acc, width, label="Test", color="#e15759", alpha=0.8)
    
    ax.set_xlabel("Feature Configuration", fontsize=12)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=12)
    ax.set_title("Nitrogen Classification: Feature Engineering Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Performance comparison saved → {savepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FEATURE ENGINEERING COMPARISON FOR NITROGEN CLASSIFICATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Load data
    X, y_water, y_nitrogen, wavelengths = load_and_pivot(DATA_CSV)
    n_samples = len(y_water)
    nitrogen_classes = sorted(np.unique(y_nitrogen).tolist())
    
    print(f"\n  Nitrogen classes: {nitrogen_classes} mg/kg")
    print(f"  Total samples: {n_samples}")

    # 2. Split data (same as training script)
    print("\n" + "=" * 70)
    print("SPLITTING DATA  (70% train / 15% val / 15% test)")
    print("=" * 70)
    
    strat_key = [f"{int(w)}_{int(n)}" for w, n in zip(y_water, y_nitrogen)]
    X_temp, X_test, yw_temp, yw_test, yn_temp, yn_test, sk_temp, _ = train_test_split(
        X, y_water, y_nitrogen, strat_key,
        test_size=0.15, random_state=42, stratify=strat_key,
    )
    strat_temp = [f"{int(w)}_{int(n)}" for w, n in zip(yw_temp, yn_temp)]
    X_train, X_val, yw_train, yw_val, yn_train, yn_val = train_test_split(
        X_temp, yw_temp, yn_temp,
        test_size=0.176, random_state=42, stratify=strat_temp,
    )
    
    print(f"  Train : {len(yw_train):>4} samples")
    print(f"  Val   : {len(yw_val):>4} samples")
    print(f"  Test  : {len(yw_test):>4} samples")

    # 3. Load water model (needed for water-as-feature variants)
    print("\n" + "=" * 70)
    print("LOADING WATER MODEL")
    print("=" * 70)
    
    water_model_path = MODELS_DIR / "svr_water_records.pkl"
    with open(water_model_path, "rb") as f:
        water_pkg = pickle.load(f)
    water_model = water_pkg["model"]
    water_scaler = water_pkg["scaler"]
    print(f"  ✓ Loaded {water_model_path}")

    # 4. Train all variants
    results = []
    
    # VARIANT 1: BASELINE (raw spectral features)
    print("\n" + "=" * 70)
    print("VARIANT 1: BASELINE (Spectral Features Only)")
    print("=" * 70)
    X_train_v1, X_val_v1, X_test_v1, scaler_v1, pca_v1 = build_baseline_features(
        X_train, X_val, X_test
    )
    result_v1 = train_and_evaluate_variant(
        X_train_v1, X_val_v1, X_test_v1,
        yn_train, yn_val, yn_test,
        "Baseline (512 features)",
        nitrogen_classes
    )
    result_v1["scaler"] = scaler_v1
    result_v1["pca"] = pca_v1
    results.append(result_v1)
    
    # VARIANT 2: SPECTRAL + WATER
    print("\n" + "=" * 70)
    print("VARIANT 2: SPECTRAL + WATER (513 features)")
    print("=" * 70)
    X_train_v2, X_val_v2, X_test_v2, scaler_v2, pca_v2 = build_water_features(
        X_train, X_val, X_test,
        yw_train, yw_val, yw_test,
        water_model, water_scaler
    )
    result_v2 = train_and_evaluate_variant(
        X_train_v2, X_val_v2, X_test_v2,
        yn_train, yn_val, yn_test,
        "Spectral + Water (513)",
        nitrogen_classes
    )
    result_v2["scaler"] = scaler_v2
    result_v2["pca"] = pca_v2
    result_v2["uses_water"] = True
    results.append(result_v2)
    
    # VARIANT 3: PCA (50 components)
    print("\n" + "=" * 70)
    print("VARIANT 3: PCA (50 components)")
    print("=" * 70)
    X_train_v3, X_val_v3, X_test_v3, scaler_v3, pca_v3 = build_pca_features(
        X_train, X_val, X_test, n_components=50
    )
    result_v3 = train_and_evaluate_variant(
        X_train_v3, X_val_v3, X_test_v3,
        yn_train, yn_val, yn_test,
        "PCA-50 (50 features)",
        nitrogen_classes
    )
    result_v3["scaler"] = scaler_v3
    result_v3["pca"] = pca_v3
    results.append(result_v3)
    
    # VARIANT 4: PCA + WATER
    print("\n" + "=" * 70)
    print("VARIANT 4: PCA + WATER (51 features)")
    print("=" * 70)
    X_train_v4, X_val_v4, X_test_v4, scaler_v4, pca_v4 = build_pca_water_features(
        X_train, X_val, X_test,
        yw_train, yw_val, yw_test,
        water_model, water_scaler,
        n_components=50
    )
    result_v4 = train_and_evaluate_variant(
        X_train_v4, X_val_v4, X_test_v4,
        yn_train, yn_val, yn_test,
        "PCA-50 + Water (51)",
        nitrogen_classes
    )
    result_v4["scaler"] = scaler_v4
    result_v4["pca"] = pca_v4
    result_v4["uses_water"] = True
    results.append(result_v4)

    # 5. Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: FEATURE ENGINEERING COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Variant':<30} {'Val Bal Acc':<15} {'Test Bal Acc':<15} {'CV Mean±SD':<20}")
    print("-" * 80)
    for r in results:
        variant = r["variant"]
        val_acc = r["metrics_val"]["balanced_accuracy"] * 100
        test_acc = r["metrics_test"]["balanced_accuracy"] * 100
        cv_mean = r["cv_mean"] * 100
        cv_std = r["cv_std"] * 100
        print(f"{variant:<30} {val_acc:>6.2f}%         {test_acc:>6.2f}%         {cv_mean:>6.2f}% ± {cv_std:>5.2f}%")
    
    # 6. Find best variant (by val balanced accuracy)
    best_idx = np.argmax([r["metrics_val"]["balanced_accuracy"] for r in results])
    best_result = results[best_idx]
    best_variant = best_result["variant"]
    best_val_acc = best_result["metrics_val"]["balanced_accuracy"] * 100
    best_test_acc = best_result["metrics_test"]["balanced_accuracy"] * 100
    
    print("\n" + "=" * 70)
    print(f"BEST VARIANT: {best_variant}")
    print("=" * 70)
    print(f"  Val Balanced Accuracy  : {best_val_acc:.2f}%")
    print(f"  Test Balanced Accuracy : {best_test_acc:.2f}%")
    
    # Check if best variant is better than baseline
    baseline_val_acc = results[0]["metrics_val"]["balanced_accuracy"] * 100
    baseline_test_acc = results[0]["metrics_test"]["balanced_accuracy"] * 100
    improvement_val = best_val_acc - baseline_val_acc
    improvement_test = best_test_acc - baseline_test_acc
    
    print(f"\n  Improvement over baseline:")
    print(f"    Val  : {improvement_val:+.2f}% ({baseline_val_acc:.2f}% → {best_val_acc:.2f}%)")
    print(f"    Test : {improvement_test:+.2f}% ({baseline_test_acc:.2f}% → {best_test_acc:.2f}%)")
    
    if improvement_val >= 1.0:
        print(f"  ✓ Significant improvement (≥1%) — saving best model")
        best_model_pkg = {
            "model": best_result["model"],
            "scaler": best_result["scaler"],
            "pca": best_result["pca"],
            "classes": nitrogen_classes,
            "variant": best_variant,
            "uses_water": best_result.get("uses_water", False),
            "metrics_val": best_result["metrics_val"],
            "metrics_test": best_result["metrics_test"],
        }
        best_model_path = MODELS_DIR / "rf_nitrogen_best.pkl"
        with open(best_model_path, "wb") as f:
            pickle.dump(best_model_pkg, f)
        print(f"  ✓ Saved → {best_model_path}")
    else:
        print(f"  → Improvement <1% — baseline remains best")

    # 7. Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_confusion_matrices(
        results, nitrogen_classes,
        RESULTS_DIR / "plot7_feature_comparison_confusion.png"
    )
    
    plot_performance_comparison(
        results,
        RESULTS_DIR / "plot8_feature_comparison_barplot.png"
    )
    
    # 8. Write detailed report
    report_path = RESULTS_DIR / "feature_comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("FEATURE ENGINEERING COMPARISON REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("VARIANTS TESTED:\n")
        f.write("-" * 70 + "\n")
        for i, r in enumerate(results, 1):
            f.write(f"{i}. {r['variant']}\n")
        f.write("\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Variant':<30} {'Val Bal Acc':<15} {'Test Bal Acc':<15}\n")
        for r in results:
            val_acc = r["metrics_val"]["balanced_accuracy"] * 100
            test_acc = r["metrics_test"]["balanced_accuracy"] * 100
            f.write(f"{r['variant']:<30} {val_acc:>6.2f}%         {test_acc:>6.2f}%\n")
        f.write("\n")
        
        f.write(f"BEST VARIANT: {best_variant}\n")
        f.write(f"  Val Balanced Accuracy  : {best_val_acc:.2f}%\n")
        f.write(f"  Test Balanced Accuracy : {best_test_acc:.2f}%\n")
        f.write(f"  Improvement over baseline: {improvement_val:+.2f}% (val), {improvement_test:+.2f}% (test)\n")
        
    print(f"  ✓ Report saved → {report_path}")
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPARISON COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
