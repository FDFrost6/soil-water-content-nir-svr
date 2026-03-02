# Soil Water Content & Nitrogen Prediction using NIR Spectroscopy and Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine learning models to predict soil water content and detect nitrogen levels from near-infrared (NIR) spectral reflectance measurements using Support Vector Regression and Random Forest classification.

## Project Overview

This project demonstrates the use of NIR spectroscopy combined with machine learning for rapid, non-destructive soil moisture and nutrient assessment. Two separate models are trained:

### Water Content Model (SVR)
- **R² Score: 0.84** on held-out test set
- **MAE: 1.55%** water content
- **Accuracy: 92.3%** of predictions within ±5%

### Nitrogen Classification Model (RandomForest + PCA)
- **Step detection: YES ✓**
- **Balanced Accuracy: 98.1%** across 4 nitrogen classes (with PCA-50 + Water features)
- **Baseline Accuracy: 88.1%** (raw 512 wavelengths)
- **Classes:** 0 / 75 / 150 / 300 mg N kg⁻¹ (NH₄NO₃)

## Datasets

### Original Dataset (baseline)
- **Source:** NIR Spectrometer (NQ5500316), University of Bonn
- **Measurement Date:** March 24, 2023
- **Samples:** 72 independent spectral measurements
- **Water Content Levels:** 9 levels (0%–40%)
- **Spectral Range:** 189.85–2514.35 nm (~2,557 wavelengths)
- **Replicates:** 8 measurements per water content level

### NIRQuest Records Dataset (new — used for dual models)
- **Measurement Date:** February 2026
- **Spectrometer:** NQ5500316 (NIRQuest, 898–2514 nm, 512 pixels)
- **Water levels:** 0 / 5 / 15 / 25 / 35 %
- **Nitrogen levels:** 0 / 75 / 150 / 300 mg N kg⁻¹
- **Conditions:** 17 (W×N combinations)
- **Total measurements:** 343 (≥20 replicates per condition)
- **Nitrogen source:** NH₄NO₃, background soil N ≈ 0.06%

## Repository Structure

```
.
├── data/
│   ├── soil_spectral_data_individual.csv       # Baseline individual measurements
│   ├── soil_spectral_data_summary.csv          # Baseline summary statistics
│   ├── soil_spectral_data_wide.csv             # Baseline wide format
│   └── soil_spectral_data_records.csv          # NEW: long-format records dataset
│   └── records/
│       └── W{water}N{nitrogen}/                # NIRQuest text files per condition
│           └── soil[_NN]_rep_Reflection__*.txt
│
├── models/
│   ├── svm_water_content_model.pkl             # Baseline SVM model (water only)
│   ├── svr_water_records.pkl                   # NEW: SVR model (records data)
│   ├── rf_nitrogen_records.pkl                 # NEW: RF nitrogen classifier (baseline)
│   ├── rf_nitrogen_best.pkl                    # NEW: Best model (PCA-50 + Water)
│   └── ood_detector.pkl                        # NEW: OOD detector (IsolationForest)
│
├── results/
│   ├── plot1_water_actual_vs_predicted.png     # Water scatter by N class
│   ├── plot2_nitrogen_confusion_matrix.png     # Nitrogen confusion matrix
│   ├── plot3_spectral_signatures.png           # Mean spectra per N level
│   ├── plot4_water_residuals.png               # Water residual analysis
│   ├── plot5_guardrail_coverage.png            # Coverage vs confidence threshold
│   ├── plot6_nitrogen_by_water_level.png       # Nitrogen acc per water level
│   ├── plot7_feature_comparison_confusion.png  # NEW: Feature engineering comparison
│   ├── plot8_feature_comparison_barplot.png    # NEW: Accuracy vs feature config
│   ├── plot9_pca_variance_curve.png            # NEW: PCA elbow curve
│   ├── plot10_pca_accuracy_curves.png          # NEW: Accuracy vs PCA components
│   ├── plot11_pca_sweep_comparison.png         # NEW: All PCA configs compared
│   ├── training_summary.txt                    # Key metrics summary
│   ├── feature_comparison_report.txt           # NEW: Feature engineering report
│   ├── pca_sweep_report.txt                    # NEW: PCA component analysis
│   └── random_seed_stability_report.txt        # NEW: Cross-split validation
│
├── src/
│   ├── ingest_records.py                       # Parse NIRQuest text files
│   ├── train_water_nitrogen.py                 # Train water SVR + N classifier (baseline)
│   ├── evaluate_water_nitrogen.py              # Generate evaluation plots (6 plots)
│   ├── protection_layer.py                     # Safety guardrails (OOD, confidence)
│   ├── compare_feature_engineering.py          # NEW: Compare 4 feature variants
│   ├── pca_sweep_analysis.py                   # NEW: Test PCA component counts
│   ├── test_random_seeds.py                    # NEW: Cross-split stability validation
│   ├── train_svm_individual.py                 # Baseline training script
│   └── generate_accuracy_plot.py              # Baseline plot script
│
├── tests/
│   ├── test_ingest_records.py                  # NEW: unit tests for ingestion
│   ├── test_train_water_nitrogen.py            # NEW: integration tests for training
│   └── test_protection_layer.py                # NEW: safety layer unit tests
│
├── main.py                                      # NEW: Unified CLI entry point
├── PROJECT_REPORT.md                            # Detailed project report
├── PRESENTATION_SLIDES.md                       # Google Slides deck
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
└── .gitignore                                   # Git ignore rules
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/soil-water-content-prediction.git
cd soil-water-content-prediction
```

### 2. Set Up Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Full Dual-Model Pipeline (Water + Nitrogen)

#### Quick Start (Recommended - CLI Interface)

The easiest way to use the pipeline is via the unified CLI:

```bash
# Run complete pipeline: ingest → train → evaluate
python main.py full

# Run with feature engineering comparison
python main.py full --with-comparison

# Individual commands
python main.py ingest         # Parse NIRQuest files
python main.py train          # Train models
python main.py evaluate       # Generate plots
python main.py compare        # Feature engineering analysis
python main.py pca-sweep      # PCA component optimization
python main.py test-seeds     # Cross-split validation

# Make predictions on new spectra
python main.py predict --spectrum path/to/spectrum.csv

# Get help
python main.py --help
python main.py <command> --help
```

#### All Available Commands

| Command | Description |
|---------|-------------|
| `ingest` | Parse NIRQuest .txt files → CSV |
| `train` | Train water SVR + nitrogen RF |
| `evaluate` | Generate 6 evaluation plots |
| `predict` | Make predictions with safety checks |
| `compare` | Compare feature engineering (4 variants) |
| `pca-sweep` | Analyze PCA components (plot9-11) |
| `test-seeds` | Cross-split validation (5 seeds) |
| `test` | Run pytest unit tests |
| `full` | Complete pipeline (ingest→train→eval) |

#### Advanced Usage Examples

```bash
# Skip steps already completed
python main.py full --skip-ingest  # Use existing CSV
python main.py full --skip-train   # Use existing models

# Quick mode (skip re-training)
python main.py train --quick

# Custom prediction confidence
python main.py predict --spectrum data.csv --confidence 0.70

# Tests with coverage
python main.py test --coverage
```

**Troubleshooting:**
- **ModuleNotFoundError:** Activate venv with `source venv/bin/activate`
- **File not found:** Make sure you're in project root directory
- **Models missing:** Run `python main.py train` or `python main.py full`

#### Alternative: Manual Step-by-Step

```bash
# Step 1: Parse all NIRQuest text files → data/soil_spectral_data_records.csv
python src/ingest_records.py

# Step 2: Train baseline models (water SVR + nitrogen RandomForest + OOD detector)
python src/train_water_nitrogen.py

# Step 3: Generate evaluation plots (6 plots)
python src/evaluate_water_nitrogen.py

# Step 4: Compare feature engineering approaches (PCA, water-as-feature)
python src/compare_feature_engineering.py

# Step 5: Analyze optimal PCA component count (10, 25, 50, 100, 200)
python src/pca_sweep_analysis.py

# Step 6: Validate model stability across random splits
python src/test_random_seeds.py

# Step 7: Run all tests
python -m pytest tests/ -v
```

**Models trained:**
- `models/svr_water_records.pkl` - Water content SVR (R²=0.844)
- `models/rf_nitrogen_records.pkl` - Nitrogen classifier baseline (88.1%)
- `models/rf_nitrogen_best.pkl` - **Best nitrogen model with PCA-50 + Water (98.08%)**
- `models/ood_detector.pkl` - Out-of-Distribution detector (IsolationForest)

**Plots generated (11 total):**
- `results/plot1-6` - Main evaluation plots (water, nitrogen, spectra, safety)
- `results/plot7-8` - Feature engineering comparison (confusion matrices, bar chart)
- `results/plot9-11` - PCA analysis (variance curve, accuracy curves, full comparison)

### 4. Train the Baseline Water-Only Model

```bash
python src/train_svm_individual.py
```

This will:
- Load the cleaned spectral data
- Split into train/validation/test sets (70%/15%/15%)
- Train an SVM regression model
- Save the trained model to `models/svm_water_content_model.pkl`
- Generate performance visualizations

### 5. Generate Baseline Plots

```bash
python src/generate_accuracy_plot.py
```

## Results

### Dual-Model Results (NIRQuest Records, Feb 2026)

#### Water Content SVR

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| R² Score | 0.9359 | 0.9920 | **0.8437** |
| RMSE (%) | 3.00 | 1.07 | **4.71** |
| MAE (%) | 0.76 | 0.80 | **1.55** |
| Accuracy (±5%) | 99.2% | 100.0% | **92.3%** |

#### Nitrogen Classification (RandomForest, 4 classes)

**Baseline Model (512 raw wavelengths):**

| Metric | Test |
|--------|------|
| Accuracy | 88.5% |
| Balanced Accuracy | **88.1%** |
| Step Detection | **YES ✓** |

Per-class F1 scores (test): N=0 → 0.90 | N=75 → 0.83 | N=150 → 0.87 | N=300 → 0.92

**Best Model (PCA-50 + Water features):**

| Metric | Validation | Test | Cross-Val (5-fold) |
|--------|------------|------|--------------------|
| Balanced Accuracy | **97.92%** | **98.08%** | 92.13% ± 2.40% |
| Step Detection | **YES ✓** | **YES ✓** | **YES ✓** |
| Feature Count | 51 (50 PCA + 1 water) | 51 | 51 |
| Improvement | +9.84% | +10.00% | — |

**Feature Engineering Comparison:**
- Baseline (512 features): 88.08% test
- Spectral + Water (513): 90.00% test (+1.92%)
- **PCA-50 (50 features): 98.08% test (+10.00%)** ⭐
- **PCA-50 + Water (51): 97.92% val, 98.08% test** 🏆

### Baseline Water-Only Model (March 2023 data)

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| R² Score | 0.9701 | 0.9488 | **0.9398** |
| RMSE (%) | 2.30 | 2.66 | **2.98** |
| MAE (%) | 1.38 | 1.93 | **2.38** |
| Accuracy (±5%) | 92.00% | 90.91% | **90.91%** |

## Methodology

### Data Ingestion (`src/ingest_records.py`)
- Walks all `data/records/W{W}N{N}/` subdirectories
- Parses NIRQuest text files (header metadata + 512-pixel spectral data)
- Handles two filename variants: `soil_rep_*` and `soil_NN_rep_*`
- Builds long-format CSV with columns: `water_content_percent`, `nitrogen_mg_kg`, `replicate`, `wavelength_nm`, `reflectance`
- Runs 6 quality checks (NaN, water levels, nitrogen levels, wavelength consistency, replicate counts)

### Water Content Model (`src/train_water_nitrogen.py`)
- Pivots long-format data to feature matrix (343 samples × 512 wavelengths)
- Stratified split: 70% train / 15% val / 15% test (stratified on W×N combination)
- StandardScaler normalisation
- **Algorithm:** SVR with RBF kernel, C=100, ε=0.1
- **Evaluation:** R², RMSE, MAE, ±5%/±2% accuracy, 5-fold CV

### Nitrogen Classification Model

**Baseline (`src/train_water_nitrogen.py`):**
- Same feature matrix and split as water model (512 raw wavelengths)
- **Algorithm:** RandomForest (300 trees, balanced class weights) vs SVC (RBF, C=10)
- Best baseline model selected by validation balanced accuracy
- **Test Result:** 88.1% balanced accuracy

**Feature Engineering (`src/compare_feature_engineering.py`):**
- Compares 4 variants: Baseline, Spectral+Water, PCA-50, PCA-50+Water
- **Best:** PCA-50 + Water (51 features) → 97.92% val, 98.08% test
- PCA trained on standardized spectra, then water prediction added as 51st feature
- **Key insight:** Dimensionality reduction removes noise, improves generalization

**PCA Component Optimization (`src/pca_sweep_analysis.py`):**
- Tests 10, 25, 50, 100, 200 PCA components
- **Optimal:** 50 components (99.95% variance, 98.08% test accuracy)
- Beyond 50 components: no accuracy gain, higher computational cost

**Cross-Split Validation (`src/test_random_seeds.py`):**
- Tests model with 5 different random seeds (42, 123, 456, 789, 2024)
- **Result:** Stable performance (Val: 96.4% ± 2.3%, Test: 97.6% ± 1.5%)
- Confirms model is robust, not overfitting to a particular split

### Evaluation & Plots (`src/evaluate_water_nitrogen.py`)
- Plot 1: Water actual vs predicted scatter (coloured by nitrogen class)
- Plot 2: Nitrogen confusion matrix heatmap (row-normalised recall)
- Plot 3: Mean spectral signatures per nitrogen class + difference spectra
- Plot 4: Water residual boxplots by water level + histogram
- Plot 5: Safety layer coverage vs confidence threshold
- Plot 6: Nitrogen accuracy stratified by water level

### Feature Engineering Analysis
- Plot 7-8 (`src/compare_feature_engineering.py`): Confusion matrices and bar chart comparing 4 feature variants
- Plot 9-11 (`src/pca_sweep_analysis.py`): PCA variance curve, accuracy vs components, full comparison

## Visualizations

### 1. Water Content: Actual vs Predicted
`results/plot1_water_actual_vs_predicted.png` — Three-panel scatter plot for train/val/test, each point coloured by nitrogen class.

### 2. Nitrogen Confusion Matrix
`results/plot2_nitrogen_confusion_matrix.png` — Heatmap showing per-class recall and raw counts (baseline model).

### 3. Spectral Signatures
`results/plot3_spectral_signatures.png` — Mean ± SD spectra per nitrogen class + difference spectra (N>0 − N=0).

### 4. Water Residual Analysis
`results/plot4_water_residuals.png` — Residual boxplots by water level and overall residual histogram.

### 5. Safety Layer Coverage
`results/plot5_guardrail_coverage.png` — Coverage vs confidence threshold trade-off curve.

### 6. Nitrogen by Water Level
`results/plot6_nitrogen_by_water_level.png` — Nitrogen accuracy stratified by water content level.

### 7-8. Feature Engineering Comparison
`results/plot7_feature_comparison_confusion.png` — Confusion matrices for 4 variants side-by-side.  
`results/plot8_feature_comparison_barplot.png` — Balanced accuracy bar chart (val vs test).

### 9-11. PCA Component Analysis
`results/plot9_pca_variance_curve.png` — Elbow curve: variance explained vs component count.  
`results/plot10_pca_accuracy_curves.png` — Accuracy vs PCA components (with and without water).  
`results/plot11_pca_sweep_comparison.png` — Bar chart comparing all 10 PCA configurations.

## Usage Example

### Load and Use the Best Model (PCA-50 + Water)

```python
import pickle
import numpy as np

# Load best nitrogen model (PCA-50 + Water)
with open('models/rf_nitrogen_best.pkl', 'rb') as f:
    best_pkg = pickle.load(f)

rf = best_pkg['model']           # RandomForest classifier
scaler = best_pkg['scaler']      # StandardScaler for raw features
pca = best_pkg['pca']            # PCA(n_components=50)
classes = best_pkg['classes']    # [0, 75, 150, 300]
uses_water = best_pkg['uses_water']  # True

# Load water model (for predicting water content as 51st feature)
with open('models/svr_water_records.pkl', 'rb') as f:
    water_pkg = pickle.load(f)
water_svr = water_pkg['model']
water_scaler = water_pkg['scaler']

# X_new: shape (n_samples, 512) — reflectance at 512 NIR wavelengths (898–2514 nm)
X_new = np.array([...])  # your raw spectral data

# Step 1: Predict water content
X_water_scaled = water_scaler.transform(X_new)
water_pred = water_svr.predict(X_water_scaled)  # shape: (n_samples,)

# Step 2: Transform spectral features with PCA
X_scaled = scaler.transform(X_new)
X_pca = pca.transform(X_scaled)  # shape: (n_samples, 50)

# Step 3: Add water as 51st feature
X_augmented = np.hstack([X_pca, water_pred.reshape(-1, 1)])  # shape: (n_samples, 51)

# Step 4: Predict nitrogen class
nitrogen_pred = rf.predict(X_augmented)     # int, nitrogen class (mg/kg)
nitrogen_prob = rf.predict_proba(X_augmented)  # class probabilities

print(f"Water: {water_pred[0]:.1f}%")
print(f"Nitrogen: {nitrogen_pred[0]} mg/kg (confidence: {nitrogen_prob[0].max():.2f})")
```

### Load and Use the Baseline Models (512 raw wavelengths)

```python
import pickle
import numpy as np

# Load water SVR
with open('models/svr_water_records.pkl', 'rb') as f:
    water_pkg = pickle.load(f)
svr    = water_pkg['model']
scaler = water_pkg['scaler']

# Load nitrogen RandomForest
with open('models/rf_nitrogen_records.pkl', 'rb') as f:
    nitrogen_pkg = pickle.load(f)
rf      = nitrogen_pkg['model']
classes = nitrogen_pkg['classes']  # [0, 75, 150, 300]

# X_new: shape (n_samples, 512) — reflectance at 512 NIR wavelengths (898–2514 nm)
X_new = np.array([...])
X_scaled = scaler.transform(X_new)

water_pred    = svr.predict(X_scaled)          # float, water content (%)
nitrogen_pred = rf.predict(X_scaled)           # int, nitrogen class (mg/kg)
nitrogen_prob = rf.predict_proba(X_scaled)     # class probabilities

print(f"Water: {water_pred[0]:.1f}%   Nitrogen: {nitrogen_pred[0]} mg/kg")
```

### Load and Use with Safety Layer (Protection/Guardrails)

The safety layer provides **Out-of-Distribution (OOD) detection**, **confidence thresholds**, and **water-based guardrails** to prevent unreliable predictions on poor-quality or unusual spectra.

```python
import pickle
import numpy as np
from src.protection_layer import predict_safe, load_models

# Load all models (water SVR, nitrogen RF, OOD detector)
water_pkg, nitrogen_pkg, ood_model = load_models()

# X_new: shape (1, 512) — single spectrum (raw, unscaled)
X_new = np.array([[...]])  # your 512-pixel NIR spectrum

# Safe prediction with guardrails
result = predict_safe(
    X_new,
    water_pkg,
    nitrogen_pkg,
    ood_model,
    thresholds=None  # uses defaults: N_conf=0.60, water_range=[-2, 40]
)

print(f"Status: {result['status']}")  # 'ok', 'uncertain', or 'invalid'
print(f"Water:  {result['water_pred']:.1f}%")
print(f"Nitrogen: {result['nitrogen_pred']} mg/kg")  # None if rejected
print(f"Confidence: {result['max_prob']:.2f}")
print(f"Reason: {result['reason']}")
```

**Status codes:**
- **`ok`**: All checks passed, predictions are reliable
- **`uncertain`**: Nitrogen prediction has low confidence (< threshold)
- **`invalid`**: Spectrum is out-of-distribution OR water is out of range

**Protection layers:**
1. **OOD Detection**: IsolationForest rejects anomalous spectra (trained on 3% contamination rate)
2. **Water Range Check**: Rejects predictions outside [-2%, 40%] range
3. **Nitrogen Confidence Threshold**: Rejects nitrogen predictions with `max_prob < 0.60`

**Tunable thresholds:**
```python
custom_thresholds = {
    "nitrogen_confidence": 0.70,  # stricter confidence (default: 0.60)
    "ood_contamination": 0.05,    # expected outlier rate (default: 0.03)
    "water_min": 0.0,             # tighter water bounds (default: -2.0)
    "water_max": 35.0,            # (default: 40.0)
}
result = predict_safe(X_new, water_pkg, nitrogen_pkg, ood_model, custom_thresholds)
```


## Contributors

- Nelson Pinheiro
- University of Bonn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- University of Bonn for providing the spectroscopy equipment
- Baseline data collection: March 24, 2023
- NIRQuest records dataset: February 2026
- Feature engineering analysis: February 23, 2026

---

**Citation:**
If you use this code or data, please cite:
```
Pinheiro, N. (2026). Soil Water Content & Nitrogen Prediction using NIR Spectroscopy 
and Machine Learning with PCA Feature Engineering. University of Bonn. 
https://github.com/YOUR_USERNAME/soil-water-content-prediction
```
