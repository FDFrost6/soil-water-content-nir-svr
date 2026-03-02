"""
Random Seed Stability Analysis
================================
Tests how val/test accuracy varies across different stratified splits.
This checks if the "test > val" pattern is consistent or just lucky.

Tests 5 different random seeds for the train/val/test split.

Usage:
  python src/test_random_seeds.py
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
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_CSV = Path("data/soil_spectral_data_records.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_and_pivot(csv_path: Path):
    """Load long-format CSV, pivot to feature matrix."""
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

    return X, y_water, y_nitrogen, wavelengths


def train_best_model(X_train, X_val, X_test, yn_train, yn_val, yn_test,
                     yw_train, yw_val, yw_test, water_model, water_scaler):
    """Train the best model: PCA-50 + Water."""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA
    pca = PCA(n_components=50, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Add water predictions
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
    bal_acc_train = balanced_accuracy_score(yn_train, rf.predict(X_train_pca))
    bal_acc_val = balanced_accuracy_score(yn_val, rf.predict(X_val_pca))
    bal_acc_test = balanced_accuracy_score(yn_test, rf.predict(X_test_pca))
    
    return bal_acc_train, bal_acc_val, bal_acc_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("RANDOM SEED STABILITY ANALYSIS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("Testing PCA-50+Water model with different train/val/test splits")
    print("Random seeds: [42, 123, 456, 789, 2024]\n")

    # 1. Load data
    X, y_water, y_nitrogen, wavelengths = load_and_pivot(DATA_CSV)
    nitrogen_classes = sorted(np.unique(y_nitrogen).tolist())
    
    print(f"  Total samples: {len(y_water)}")

    # 2. Load water model
    water_model_path = MODELS_DIR / "svr_water_records.pkl"
    with open(water_model_path, "rb") as f:
        water_pkg = pickle.load(f)
    water_model = water_pkg["model"]
    water_scaler = water_pkg["scaler"]
    print(f"  ✓ Loaded water model\n")

    # 3. Test multiple random seeds
    seeds = [42, 123, 456, 789, 2024]
    results = []
    
    print("=" * 70)
    print("TESTING MULTIPLE RANDOM SPLITS")
    print("=" * 70)
    
    for seed in seeds:
        print(f"\n  Random Seed: {seed}")
        
        # Split data with this seed
        strat_key = [f"{int(w)}_{int(n)}" for w, n in zip(y_water, y_nitrogen)]
        X_temp, X_test, yw_temp, yw_test, yn_temp, yn_test, sk_temp, _ = train_test_split(
            X, y_water, y_nitrogen, strat_key,
            test_size=0.15, random_state=seed, stratify=strat_key,
        )
        strat_temp = [f"{int(w)}_{int(n)}" for w, n in zip(yw_temp, yn_temp)]
        X_train, X_val, yw_train, yw_val, yn_train, yn_val = train_test_split(
            X_temp, yw_temp, yn_temp,
            test_size=0.176, random_state=seed, stratify=strat_temp,
        )
        
        print(f"    Train: {len(yn_train)} | Val: {len(yn_val)} | Test: {len(yn_test)}")
        
        # Train best model
        bal_train, bal_val, bal_test = train_best_model(
            X_train, X_val, X_test,
            yn_train, yn_val, yn_test,
            yw_train, yw_val, yw_test,
            water_model, water_scaler
        )
        
        results.append({
            "seed": seed,
            "train": bal_train * 100,
            "val": bal_val * 100,
            "test": bal_test * 100,
            "val_test_diff": (bal_test - bal_val) * 100
        })
        
        print(f"    Train: {bal_train*100:>6.2f}% | Val: {bal_val*100:>6.2f}% | Test: {bal_test*100:>6.2f}%")
        print(f"    Val-Test Diff: {(bal_test - bal_val)*100:+6.2f}%")

    # 4. Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: STABILITY ACROSS RANDOM SEEDS")
    print("=" * 70)
    
    print(f"\n{'Seed':<10} {'Train %':<10} {'Val %':<10} {'Test %':<10} {'Test-Val':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['seed']:<10} {r['train']:>6.2f}    {r['val']:>6.2f}    {r['test']:>6.2f}    {r['val_test_diff']:>+6.2f}")
    
    # Statistics
    val_mean = np.mean([r["val"] for r in results])
    val_std = np.std([r["val"] for r in results])
    test_mean = np.mean([r["test"] for r in results])
    test_std = np.std([r["test"] for r in results])
    diff_mean = np.mean([r["val_test_diff"] for r in results])
    diff_std = np.std([r["val_test_diff"] for r in results])
    
    print("-" * 70)
    print(f"{'MEAN':<10} {'—':<10} {val_mean:>6.2f}    {test_mean:>6.2f}    {diff_mean:>+6.2f}")
    print(f"{'STD':<10} {'—':<10} {val_std:>6.2f}    {test_std:>6.2f}    {diff_std:>6.2f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if abs(diff_mean) < 2.0:
        print("✓ Val and Test accuracies are BALANCED on average")
        print(f"  Average difference: {diff_mean:+.2f}% ± {diff_std:.2f}%")
    elif diff_mean > 2.0:
        print("⚠ Test tends to be HIGHER than Val on average")
        print(f"  Average difference: {diff_mean:+.2f}% ± {diff_std:.2f}%")
    else:
        print("⚠ Val tends to be HIGHER than Test on average")
        print(f"  Average difference: {diff_mean:+.2f}% ± {diff_std:.2f}%")
    
    print(f"\nVariability:")
    print(f"  Val accuracy range: {val_mean-val_std:.2f}% to {val_mean+val_std:.2f}%")
    print(f"  Test accuracy range: {test_mean-test_std:.2f}% to {test_mean+test_std:.2f}%")
    
    if val_std < 3.0 and test_std < 3.0:
        print("\n✓ Model is STABLE across different random splits")
        print("  Both val and test have low variance (<3%)")
    else:
        print("\n⚠ Model shows MODERATE sensitivity to split choice")
        print(f"  Variance: Val±{val_std:.2f}%, Test±{test_std:.2f}%")
    
    # Write report
    report_path = RESULTS_DIR / "random_seed_stability_report.txt"
    with open(report_path, "w") as f:
        f.write("RANDOM SEED STABILITY ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Model: PCA-50 + Water (best configuration)\n")
        f.write(f"Seeds tested: {seeds}\n\n")
        
        f.write(f"{'Seed':<10} {'Train %':<10} {'Val %':<10} {'Test %':<10} {'Test-Val':<10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['seed']:<10} {r['train']:>6.2f}    {r['val']:>6.2f}    {r['test']:>6.2f}    {r['val_test_diff']:>+6.2f}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'MEAN':<10} {'—':<10} {val_mean:>6.2f}    {test_mean:>6.2f}    {diff_mean:>+6.2f}\n")
        f.write(f"{'STD':<10} {'—':<10} {val_std:>6.2f}    {test_std:>6.2f}    {diff_std:>6.2f}\n")
        
    print(f"\n  ✓ Report saved → {report_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
