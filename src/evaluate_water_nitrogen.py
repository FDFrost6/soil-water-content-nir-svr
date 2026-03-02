"""
Phase 3: Evaluation & Visualisation
=====================================
Loads trained models and raw data, then generates publication-quality plots:

  results/plot1_water_actual_vs_predicted.png  — water scatter (coloured by N class)
  results/plot2_nitrogen_confusion_matrix.png  — nitrogen confusion matrix heatmap
  results/plot3_spectral_signatures.png        — mean spectra per nitrogen class
  results/plot4_water_residuals.png            — water residual analysis by water level

Usage:
  python src/evaluate_water_nitrogen.py
  (from project root)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error, balanced_accuracy_score

# Import protection layer
import sys
sys.path.insert(0, str(Path(__file__).parent))
from protection_layer import predict_safe_batch, DEFAULT_THRESHOLDS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_CSV      = Path("data/soil_spectral_data_records.csv")
WATER_MODEL   = Path("models/svr_water_records.pkl")
NITROGEN_MODEL= Path("models/rf_nitrogen_records.pkl")
OOD_DETECTOR  = Path("models/ood_detector.pkl")
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Colour maps
NITROGEN_COLOURS = {0: "#4e79a7", 75: "#f28e2b", 150: "#e15759", 300: "#76b7b2"}
WATER_CMAP = plt.cm.viridis


# ---------------------------------------------------------------------------
# Helper: reproduce the same train/val/test split as training script
# ---------------------------------------------------------------------------

def reproduce_split(X, y_water, y_nitrogen):
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
    return X_train, X_val, X_test, yw_train, yw_val, yw_test, yn_train, yn_val, yn_test


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

def load_data_and_models():
    print("Loading data …")
    df = pd.read_csv(DATA_CSV)

    pivot = df.pivot_table(
        index=["water_content_percent", "nitrogen_mg_kg", "replicate"],
        columns="wavelength_nm",
        values="reflectance",
        aggfunc="first",
    ).sort_index()

    wavelengths = np.array(pivot.columns, dtype=float)
    X = pivot.values
    y_water    = pivot.index.get_level_values("water_content_percent").to_numpy(dtype=float)
    y_nitrogen = pivot.index.get_level_values("nitrogen_mg_kg").to_numpy(dtype=int)

    print("Loading models …")
    with open(WATER_MODEL, "rb") as f:
        water_pkg = pickle.load(f)
    with open(NITROGEN_MODEL, "rb") as f:
        nitrogen_pkg = pickle.load(f)
    with open(OOD_DETECTOR, "rb") as f:
        ood_model = pickle.load(f)

    svr    = water_pkg["model"]
    scaler = water_pkg["scaler"]
    rf     = nitrogen_pkg["model"]
    classes= nitrogen_pkg["classes"]

    return df, X, y_water, y_nitrogen, wavelengths, svr, scaler, rf, classes, ood_model


# ---------------------------------------------------------------------------
# Plot 1: Water actual vs predicted (scatter, coloured by nitrogen class)
# ---------------------------------------------------------------------------

def plot_water_scatter(yw_all, yw_pred_all, yn_all, split_labels, savepath: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Water Content — Actual vs Predicted (SVR)", fontsize=14, fontweight="bold")

    for ax, split in zip(axes, ["Train", "Val", "Test"]):
        mask = np.array(split_labels) == split
        yw_t = yw_all[mask]
        yw_p = yw_pred_all[mask]
        yn_t = yn_all[mask]

        for n_val, colour in NITROGEN_COLOURS.items():
            nm = yn_t == n_val
            if nm.sum() == 0:
                continue
            ax.scatter(yw_t[nm], yw_p[nm], color=colour, alpha=0.7, s=55,
                       label=f"N={n_val} mg/kg", edgecolors="white", linewidths=0.4)

        lim_min = min(yw_t.min(), yw_p.min()) - 1
        lim_max = max(yw_t.max(), yw_p.max()) + 1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.5, label="Ideal")
        ax.fill_between([lim_min, lim_max],
                        [lim_min - 5, lim_max - 5],
                        [lim_min + 5, lim_max + 5],
                        alpha=0.08, color="red", label="±5% band")

        r2  = r2_score(yw_t, yw_p)
        mae = mean_absolute_error(yw_t, yw_p)
        ax.set_title(f"{split}  (n={mask.sum()})\nR²={r2:.3f}  MAE={mae:.2f}%", fontsize=11)
        ax.set_xlabel("Actual Water Content (%)", fontsize=10)
        ax.set_ylabel("Predicted Water Content (%)" if split == "Train" else "", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect("equal", adjustable="box")

    # Shared legend
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8,
                      label=f"N={n} mg/kg") for n, c in NITROGEN_COLOURS.items()]
    handles += [Line2D([0], [0], color="red", linestyle="--", label="Ideal")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {savepath}")


# ---------------------------------------------------------------------------
# Plot 2: Nitrogen confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(yn_true, yn_pred, classes, title_suffix: str, savepath: Path):
    cm = confusion_matrix(yn_true, yn_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")

    tick_labels = [f"{c}" for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_yticklabels(tick_labels, fontsize=11)

    for i in range(len(classes)):
        for j in range(len(classes)):
            colour = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                    ha="center", va="center", fontsize=10, color=colour, fontweight="bold")

    ax.set_xlabel("Predicted N class (mg/kg)", fontsize=12)
    ax.set_ylabel("True N class (mg/kg)", fontsize=12)
    ax.set_title(f"Nitrogen Confusion Matrix — {title_suffix}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {savepath}")


# ---------------------------------------------------------------------------
# Plot 3: Mean spectral signatures per nitrogen class (all water levels combined)
# ---------------------------------------------------------------------------

def plot_spectral_signatures(df: pd.DataFrame, wavelengths: np.ndarray, savepath: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Panel A: mean spectrum per nitrogen class
    ax = axes[0]
    nitrogen_levels = sorted(df["nitrogen_mg_kg"].unique())
    for n_val in nitrogen_levels:
        sub = df[df["nitrogen_mg_kg"] == n_val]
        pivot = sub.pivot_table(index=["water_content_percent", "replicate"],
                                columns="wavelength_nm", values="reflectance", aggfunc="first")
        mean_spec = pivot.mean(axis=0).values
        std_spec  = pivot.std(axis=0).values
        wl = np.array(pivot.columns, dtype=float)
        colour = NITROGEN_COLOURS[n_val]
        ax.plot(wl, mean_spec, color=colour, lw=1.5, label=f"N={n_val} mg/kg")
        ax.fill_between(wl, mean_spec - std_spec, mean_spec + std_spec,
                        alpha=0.15, color=colour)

    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance (counts)", fontsize=11)
    ax.set_title("Mean Spectral Signatures by Nitrogen Class (± 1 SD, all water levels)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel B: difference spectra relative to N=0
    ax2 = axes[1]
    sub0 = df[df["nitrogen_mg_kg"] == 0]
    pivot0 = sub0.pivot_table(index=["water_content_percent", "replicate"],
                              columns="wavelength_nm", values="reflectance", aggfunc="first")
    mean0 = pivot0.mean(axis=0).values
    wl0 = np.array(pivot0.columns, dtype=float)

    for n_val in [75, 150, 300]:
        sub = df[df["nitrogen_mg_kg"] == n_val]
        pivot = sub.pivot_table(index=["water_content_percent", "replicate"],
                                columns="wavelength_nm", values="reflectance", aggfunc="first")
        mean_n = pivot.mean(axis=0).values
        diff   = mean_n - mean0
        ax2.plot(wl0, diff, color=NITROGEN_COLOURS[n_val], lw=1.5, label=f"N={n_val} − N=0")
        ax2.axhline(0, color="black", lw=0.8, linestyle="--")

    ax2.set_xlabel("Wavelength (nm)", fontsize=11)
    ax2.set_ylabel("ΔReflectance (counts)", fontsize=11)
    ax2.set_title("Difference Spectra: N>0 minus N=0 (all water levels)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {savepath}")


# ---------------------------------------------------------------------------
# Plot 4: Water residuals by water level
# ---------------------------------------------------------------------------

def plot_water_residuals(yw_true, yw_pred, savepath: Path):
    water_levels = sorted(np.unique(yw_true).astype(int).tolist())
    residuals = yw_pred - yw_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Water Content — Residual Analysis", fontsize=13, fontweight="bold")

    # Box plot per water level
    ax = axes[0]
    data_by_level = [residuals[yw_true == w] for w in water_levels]
    bp = ax.boxplot(data_by_level, labels=[f"{w}%" for w in water_levels],
                    patch_artist=True, notch=False)
    colours = plt.cm.plasma(np.linspace(0.1, 0.9, len(water_levels)))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.axhline(5, color="orange", linestyle=":", lw=1, alpha=0.7)
    ax.axhline(-5, color="orange", linestyle=":", lw=1, alpha=0.7)
    ax.set_xlabel("Actual Water Content (%)", fontsize=11)
    ax.set_ylabel("Residual (predicted − actual) %", fontsize=11)
    ax.set_title("Residuals by Water Level", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Histogram of all residuals
    ax2 = axes[1]
    ax2.hist(residuals, bins=20, color="#4e79a7", edgecolor="white", alpha=0.8)
    ax2.axvline(0, color="red", linestyle="--", lw=1.5)
    ax2.axvline(residuals.mean(), color="orange", linestyle="-", lw=1.5,
                label=f"Mean={residuals.mean():.2f}%")
    ax2.set_xlabel("Residual (predicted − actual) %", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Residual Distribution (all samples)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {savepath}")


# ---------------------------------------------------------------------------
# Plot 5: Guardrail Coverage vs Confidence Threshold
# ---------------------------------------------------------------------------

def plot_guardrail_coverage(
    X_test: np.ndarray,
    yn_test: np.ndarray,
    water_pkg: dict,
    nitrogen_pkg: dict,
    ood_model,
    savepath: Path
):
    """
    Plot coverage (% samples with status='ok') vs nitrogen confidence threshold.
    Also show accuracy on accepted samples.
    
    Optimized version: precompute predictions once, then vary threshold.
    """
    # Reduced number of thresholds for faster computation
    thresholds_to_test = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    coverages = []
    accuracies_on_accepted = []
    
    print("\n  Computing coverage vs threshold (optimized) …")
    
    # Precompute OOD checks and nitrogen probabilities (expensive part)
    scaler = nitrogen_pkg["scaler"]
    rf_model = nitrogen_pkg["model"]
    X_test_scaled = scaler.transform(X_test)
    
    # OOD check (do once)
    ood_valid = ood_model.predict(X_test_scaled) == 1
    
    # Nitrogen probabilities (do once)
    nitrogen_probs = rf_model.predict_proba(X_test_scaled)
    max_probs = np.max(nitrogen_probs, axis=1)
    nitrogen_preds = rf_model.predict(X_test_scaled)
    
    # Now iterate over thresholds (fast)
    for thresh in thresholds_to_test:
        # Check which samples pass the confidence threshold
        conf_ok = max_probs >= thresh
        
        # Combined: OOD valid AND confidence ok
        status_ok = ood_valid & conf_ok
        
        coverage = np.mean(status_ok) * 100
        coverages.append(coverage)
        
        # Accuracy on accepted samples
        if np.sum(status_ok) > 0:
            yn_pred_accepted = nitrogen_preds[status_ok]
            yn_true_accepted = yn_test[status_ok]
            acc = np.mean(yn_pred_accepted == yn_true_accepted) * 100
            accuracies_on_accepted.append(acc)
        else:
            accuracies_on_accepted.append(np.nan)
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = "#4e79a7"
    ax1.plot(thresholds_to_test, coverages, marker="o", color=color1, linewidth=2, markersize=8, label="Coverage")
    ax1.set_xlabel("Nitrogen Confidence Threshold", fontsize=12)
    ax1.set_ylabel("Coverage (% samples accepted)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.set_ylim(0, 105)
    
    # Secondary axis for accuracy
    ax2 = ax1.twinx()
    color2 = "#e15759"
    ax2.plot(thresholds_to_test, accuracies_on_accepted, marker="s", color=color2, 
             linewidth=2, markersize=8, linestyle="--", label="Accuracy (on accepted)")
    ax2.set_ylabel("Accuracy on Accepted Samples (%)", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(80, 105)
    
    # Mark default threshold
    default_thresh = DEFAULT_THRESHOLDS["nitrogen_confidence"]
    ax1.axvline(default_thresh, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    # Place label at top, slightly offset to avoid covering the line
    ax1.text(default_thresh + 0.015, 98, f"Default\n({default_thresh:.2f})", 
             horizontalalignment="left", verticalalignment="top", 
             fontsize=10, color="gray", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
    
    plt.title("Guardrail Performance: Coverage vs Confidence Threshold\n(Test Set, OOD + Water checks active)",
              fontsize=13, fontweight="bold")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=11)
    
    plt.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {savepath}")


# ---------------------------------------------------------------------------
# Plot 6: Nitrogen Balanced Accuracy per Water Level
# ---------------------------------------------------------------------------

def plot_nitrogen_by_water_level(
    yw_test: np.ndarray,
    yn_test: np.ndarray,
    yn_pred_test: np.ndarray,
    classes: list,
    savepath: Path
):
    """
    Show nitrogen balanced accuracy stratified by water content level.
    """
    water_levels = sorted(np.unique(yw_test).astype(int).tolist())
    bal_accs = []
    counts = []
    
    print("\n  Computing nitrogen accuracy per water level …")
    for w in water_levels:
        mask = yw_test == w
        if mask.sum() < 2:
            bal_accs.append(np.nan)
            counts.append(0)
            continue
        yn_t = yn_test[mask]
        yn_p = yn_pred_test[mask]
        bal_acc = balanced_accuracy_score(yn_t, yn_p) * 100
        bal_accs.append(bal_acc)
        counts.append(mask.sum())
        print(f"    Water {w:2d}%: Balanced Acc = {bal_acc:.1f}%  (n={mask.sum()})")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colours = plt.cm.viridis(np.linspace(0.2, 0.9, len(water_levels)))
    bars = ax.bar([str(w) for w in water_levels], bal_accs, color=colours, 
                  edgecolor="white", linewidth=1.5, alpha=0.8)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f"n={count}", ha="center", va="bottom", fontsize=10)
    
    # Reference lines
    ax.axhline(100, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Perfect")
    ax.axhline(80, color="orange", linestyle=":", linewidth=1, alpha=0.5, label="Good (≥80%)")
    
    ax.set_xlabel("Water Content Level (%)", fontsize=12)
    ax.set_ylabel("Nitrogen Balanced Accuracy (%)", fontsize=12)
    ax.set_title("Nitrogen Classification Performance by Water Level\n(Test Set)", 
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {savepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EVALUATION & VISUALISATION")
    print("=" * 70 + "\n")

    df, X, y_water, y_nitrogen, wavelengths, svr, scaler, rf, classes, ood_model = load_data_and_models()

    # Reproduce splits
    (X_train, X_val, X_test,
     yw_train, yw_val, yw_test,
     yn_train, yn_val, yn_test) = reproduce_split(X, y_water, y_nitrogen)

    # Scale
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Predictions
    yw_pred_train = svr.predict(X_train_s)
    yw_pred_val   = svr.predict(X_val_s)
    yw_pred_test  = svr.predict(X_test_s)

    yn_pred_train = rf.predict(X_train_s)
    yn_pred_val   = rf.predict(X_val_s)
    yn_pred_test  = rf.predict(X_test_s)

    # Combine all splits for multi-split plots
    yw_all   = np.concatenate([yw_train, yw_val, yw_test])
    yw_pred  = np.concatenate([yw_pred_train, yw_pred_val, yw_pred_test])
    yn_all   = np.concatenate([yn_train, yn_val, yn_test])
    yn_pred  = np.concatenate([yn_pred_train, yn_pred_val, yn_pred_test])
    splits   = (["Train"] * len(yw_train) + ["Val"] * len(yw_val) + ["Test"] * len(yw_test))

    print("\nGenerating plots …")

    # Plot 1
    plot_water_scatter(
        yw_all, yw_pred, yn_all, splits,
        RESULTS_DIR / "plot1_water_actual_vs_predicted.png",
    )

    # Plot 2 — test set confusion matrix
    plot_confusion_matrix(
        yn_test, yn_pred_test, classes,
        title_suffix="Test Set (RandomForest)",
        savepath=RESULTS_DIR / "plot2_nitrogen_confusion_matrix.png",
    )

    # Plot 3 — spectral signatures
    plot_spectral_signatures(df, wavelengths, RESULTS_DIR / "plot3_spectral_signatures.png")

    # Plot 4 — water residuals (all data combined)
    plot_water_residuals(yw_all, yw_pred, RESULTS_DIR / "plot4_water_residuals.png")

    # Plot 5 — guardrail coverage vs threshold
    water_pkg = {"model": svr, "scaler": scaler}
    nitrogen_pkg = {"model": rf, "scaler": scaler, "classes": classes}
    plot_guardrail_coverage(
        X_test, yn_test, water_pkg, nitrogen_pkg, ood_model,
        RESULTS_DIR / "plot5_guardrail_coverage.png"
    )

    # Plot 6 — nitrogen accuracy per water level
    plot_nitrogen_by_water_level(
        yw_test, yn_test, yn_pred_test, classes,
        RESULTS_DIR / "plot6_nitrogen_by_water_level.png"
    )

    print("\n" + "=" * 70)
    print("ALL PLOTS SAVED TO results/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
