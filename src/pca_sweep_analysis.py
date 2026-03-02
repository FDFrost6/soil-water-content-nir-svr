"""
PCA Component Analysis: Finding Optimal Dimensionality
======================================================
Tests multiple PCA component counts to find the optimal balance between:
  - Dimensionality reduction (computational efficiency)
  - Variance retention (information preservation)
  - Classification accuracy (nitrogen prediction)

Tested configurations:
  - PCA-10, PCA-25, PCA-50, PCA-100, PCA-200 (spectral only)
  - PCA-10+W, PCA-25+W, PCA-50+W, PCA-100+W, PCA-200+W (with water)

Outputs:
  - Elbow curve plot (variance explained vs components)
  - Accuracy vs components plot
  - Comparison table
  - Best PCA configuration saved

Usage:
  python src/pca_sweep_analysis.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
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
# Data loading (reuse from previous scripts)
# ---------------------------------------------------------------------------

def load_and_pivot(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Load long-format CSV, pivot to feature matrix."""
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    
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

    print(f"  Feature matrix: {X.shape}")
    return X, y_water, y_nitrogen, wavelengths


# ---------------------------------------------------------------------------
# PCA sweep functions
# ---------------------------------------------------------------------------

def train_pca_variant(X_train, X_val, X_test, yn_train, yn_val, yn_test,
                      n_components: int, add_water: bool = False,
                      yw_train=None, yw_val=None, yw_test=None,
                      water_model=None, water_scaler=None):
    """Train and evaluate a single PCA configuration."""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    
    # Optionally add water
    if add_water and water_model is not None:
        X_train_water_scaled = water_scaler.transform(X_train)
        X_val_water_scaled = water_scaler.transform(X_val)
        X_test_water_scaled = water_scaler.transform(X_test)
        
        water_pred_train = water_model.predict(X_train_water_scaled).reshape(-1, 1)
        water_pred_val = water_model.predict(X_val_water_scaled).reshape(-1, 1)
        water_pred_test = water_model.predict(X_test_water_scaled).reshape(-1, 1)
        
        X_train_pca = np.hstack([X_train_pca, water_pred_train])
        X_val_pca = np.hstack([X_val_pca, water_pred_val])
        X_test_pca = np.hstack([X_test_pca, water_pred_test])
    
    # Train RandomForest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_pca, yn_train)
    
    # Evaluate
    yn_pred_train = rf.predict(X_train_pca)
    yn_pred_val = rf.predict(X_val_pca)
    yn_pred_test = rf.predict(X_test_pca)
    
    bal_acc_train = balanced_accuracy_score(yn_train, yn_pred_train)
    bal_acc_val = balanced_accuracy_score(yn_val, yn_pred_val)
    bal_acc_test = balanced_accuracy_score(yn_test, yn_pred_test)
    
    n_features = X_train_pca.shape[1]
    
    return {
        "n_components": n_components,
        "add_water": add_water,
        "n_features": n_features,
        "variance_explained": variance_explained,
        "bal_acc_train": bal_acc_train,
        "bal_acc_val": bal_acc_val,
        "bal_acc_test": bal_acc_test,
        "model": rf,
        "scaler": scaler,
        "pca": pca,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_variance_curve(results, savepath: Path):
    """Plot variance explained vs number of PCA components."""
    # Get unique n_components
    pca_only = [r for r in results if not r["add_water"]]
    pca_only = sorted(pca_only, key=lambda x: x["n_components"])
    
    n_comps = [r["n_components"] for r in pca_only]
    variance = [r["variance_explained"] * 100 for r in pca_only]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_comps, variance, marker='o', linewidth=2, markersize=8, color='#4e79a7')
    ax.axhline(99, color='red', linestyle='--', linewidth=1, alpha=0.7, label='99% threshold')
    ax.axhline(95, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='95% threshold')
    
    ax.set_xlabel("Number of PCA Components", fontsize=12)
    ax.set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
    ax.set_title("PCA Elbow Curve: Variance Retention vs Dimensionality", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(90, 100.5)
    
    # Annotate points
    for nc, var in zip(n_comps, variance):
        ax.annotate(f'{var:.2f}%', (nc, var), textcoords="offset points",
                   xytext=(0, 8), ha='center', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Variance curve saved → {savepath}")


def plot_accuracy_vs_components(results, savepath: Path):
    """Plot balanced accuracy vs PCA components (with and without water)."""
    # Separate by water addition
    pca_only = [r for r in results if not r["add_water"]]
    pca_water = [r for r in results if r["add_water"]]
    
    pca_only = sorted(pca_only, key=lambda x: x["n_components"])
    pca_water = sorted(pca_water, key=lambda x: x["n_components"])
    
    n_comps = [r["n_components"] for r in pca_only]
    
    val_acc_pca = [r["bal_acc_val"] * 100 for r in pca_only]
    test_acc_pca = [r["bal_acc_test"] * 100 for r in pca_only]
    
    val_acc_pca_w = [r["bal_acc_val"] * 100 for r in pca_water]
    test_acc_pca_w = [r["bal_acc_test"] * 100 for r in pca_water]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left panel: PCA only
    ax1.plot(n_comps, val_acc_pca, marker='o', linewidth=2, markersize=8,
             color='#4e79a7', label='Validation')
    ax1.plot(n_comps, test_acc_pca, marker='s', linewidth=2, markersize=8,
             color='#e15759', label='Test')
    ax1.set_xlabel("Number of PCA Components", fontsize=12)
    ax1.set_ylabel("Balanced Accuracy (%)", fontsize=12)
    ax1.set_title("PCA Only (Spectral Features)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim(70, 105)
    
    # Right panel: PCA + Water
    ax2.plot(n_comps, val_acc_pca_w, marker='o', linewidth=2, markersize=8,
             color='#4e79a7', label='Validation')
    ax2.plot(n_comps, test_acc_pca_w, marker='s', linewidth=2, markersize=8,
             color='#e15759', label='Test')
    ax2.set_xlabel("Number of PCA Components", fontsize=12)
    ax2.set_ylabel("Balanced Accuracy (%)", fontsize=12)
    ax2.set_title("PCA + Water Prediction", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim(70, 105)
    
    plt.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Accuracy curves saved → {savepath}")


def plot_combined_comparison(results, savepath: Path):
    """Bar plot comparing all PCA variants."""
    results_sorted = sorted(results, key=lambda x: (x["add_water"], x["n_components"]))
    
    labels = []
    val_acc = []
    test_acc = []
    colors = []
    
    for r in results_sorted:
        water_str = "+W" if r["add_water"] else ""
        label = f"PCA-{r['n_components']}{water_str}"
        labels.append(label)
        val_acc.append(r["bal_acc_val"] * 100)
        test_acc.append(r["bal_acc_test"] * 100)
        colors.append('#59a14f' if r["add_water"] else '#4e79a7')
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, val_acc, width, label="Validation", alpha=0.8, color=colors)
    bars2 = ax.bar(x + width/2, test_acc, width, label="Test", alpha=0.6, color=colors)
    
    ax.set_xlabel("PCA Configuration", fontsize=12)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=12)
    ax.set_title("PCA Sweep: Component Count Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(70, 105)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Combined comparison saved → {savepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PCA COMPONENT SWEEP ANALYSIS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Load data
    X, y_water, y_nitrogen, wavelengths = load_and_pivot(DATA_CSV)
    nitrogen_classes = sorted(np.unique(y_nitrogen).tolist())
    
    print(f"  Nitrogen classes: {nitrogen_classes} mg/kg")
    print(f"  Total samples: {len(y_water)}")

    # 2. Split data
    print("\n" + "=" * 70)
    print("SPLITTING DATA")
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

    # 3. Load water model
    print("\n" + "=" * 70)
    print("LOADING WATER MODEL")
    print("=" * 70)
    
    water_model_path = MODELS_DIR / "svr_water_records.pkl"
    with open(water_model_path, "rb") as f:
        water_pkg = pickle.load(f)
    water_model = water_pkg["model"]
    water_scaler = water_pkg["scaler"]
    print(f"  ✓ Loaded {water_model_path}")

    # 4. PCA sweep: test multiple component counts
    print("\n" + "=" * 70)
    print("PCA SWEEP: TESTING MULTIPLE COMPONENT COUNTS")
    print("=" * 70)
    
    component_counts = [10, 25, 50, 100, 200]
    results = []
    
    for n_comp in component_counts:
        # PCA only
        print(f"\n  → Training PCA-{n_comp} (spectral only)")
        result = train_pca_variant(
            X_train, X_val, X_test,
            yn_train, yn_val, yn_test,
            n_components=n_comp,
            add_water=False
        )
        results.append(result)
        print(f"     Variance: {result['variance_explained']*100:.2f}% | "
              f"Val: {result['bal_acc_val']*100:.2f}% | "
              f"Test: {result['bal_acc_test']*100:.2f}%")
        
        # PCA + Water
        print(f"  → Training PCA-{n_comp}+Water")
        result_w = train_pca_variant(
            X_train, X_val, X_test,
            yn_train, yn_val, yn_test,
            n_components=n_comp,
            add_water=True,
            yw_train=yw_train, yw_val=yw_val, yw_test=yw_test,
            water_model=water_model, water_scaler=water_scaler
        )
        results.append(result_w)
        print(f"     Variance: {result_w['variance_explained']*100:.2f}% | "
              f"Val: {result_w['bal_acc_val']*100:.2f}% | "
              f"Test: {result_w['bal_acc_test']*100:.2f}%")

    # 5. Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: PCA COMPONENT SWEEP")
    print("=" * 70)
    
    print(f"\n{'Config':<20} {'Features':<10} {'Var %':<10} {'Val Acc':<12} {'Test Acc':<12}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: (x["add_water"], x["n_components"])):
        water_str = "+W" if r["add_water"] else "  "
        config = f"PCA-{r['n_components']}{water_str}"
        var_pct = f"{r['variance_explained']*100:.2f}%"
        val_acc = f"{r['bal_acc_val']*100:.2f}%"
        test_acc = f"{r['bal_acc_test']*100:.2f}%"
        print(f"{config:<20} {r['n_features']:<10} {var_pct:<10} {val_acc:<12} {test_acc:<12}")
    
    # 6. Find best configuration
    best_result = max(results, key=lambda x: x["bal_acc_val"])
    water_str = "+Water" if best_result["add_water"] else ""
    best_config = f"PCA-{best_result['n_components']}{water_str}"
    
    print("\n" + "=" * 70)
    print(f"BEST CONFIGURATION: {best_config}")
    print("=" * 70)
    print(f"  Features         : {best_result['n_features']}")
    print(f"  Variance         : {best_result['variance_explained']*100:.2f}%")
    print(f"  Val Bal Accuracy : {best_result['bal_acc_val']*100:.2f}%")
    print(f"  Test Bal Accuracy: {best_result['bal_acc_test']*100:.2f}%")
    
    # 7. Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_variance_curve(
        results,
        RESULTS_DIR / "plot9_pca_variance_curve.png"
    )
    
    plot_accuracy_vs_components(
        results,
        RESULTS_DIR / "plot10_pca_accuracy_curves.png"
    )
    
    plot_combined_comparison(
        results,
        RESULTS_DIR / "plot11_pca_sweep_comparison.png"
    )
    
    # 8. Write detailed report
    report_path = RESULTS_DIR / "pca_sweep_report.txt"
    with open(report_path, "w") as f:
        f.write("PCA COMPONENT SWEEP ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("COMPONENT COUNTS TESTED:\n")
        f.write(f"  {component_counts}\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Config':<20} {'Features':<10} {'Var %':<10} {'Val Acc':<12} {'Test Acc':<12}\n")
        for r in sorted(results, key=lambda x: (x["add_water"], x["n_components"])):
            water_str = "+W" if r["add_water"] else "  "
            config = f"PCA-{r['n_components']}{water_str}"
            f.write(f"{config:<20} {r['n_features']:<10} "
                   f"{r['variance_explained']*100:.2f}%    "
                   f"{r['bal_acc_val']*100:.2f}%      "
                   f"{r['bal_acc_test']*100:.2f}%\n")
        f.write("\n")
        
        f.write(f"BEST CONFIGURATION: {best_config}\n")
        f.write(f"  Features         : {best_result['n_features']}\n")
        f.write(f"  Variance         : {best_result['variance_explained']*100:.2f}%\n")
        f.write(f"  Val Bal Accuracy : {best_result['bal_acc_val']*100:.2f}%\n")
        f.write(f"  Test Bal Accuracy: {best_result['bal_acc_test']*100:.2f}%\n")
        
    print(f"  ✓ Report saved → {report_path}")
    
    print("\n" + "=" * 70)
    print("PCA SWEEP ANALYSIS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
