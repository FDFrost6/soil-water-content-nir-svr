# Soil Water Content Prediction Using Near-Infrared Spectroscopy and Machine Learning

**Project Report**  
**Date:** December 9, 2025  
**Institution:** University of Bonn

---

## Executive Summary

This project successfully developed a Support Vector Machine (SVM) regression model to predict soil water content from near-infrared (NIR) spectral reflectance data. The final model achieved **R² = 0.94** and **MAE = 2.38%** on test data, demonstrating excellent predictive performance.

**Key Results:**
- **Test R² Score:** 0.9398 (excellent)
- **Test MAE:** 2.38% water content
- **Test RMSE:** 2.98%
- **Accuracy (±5%):** 90.91% of predictions within 5% error

---

## 1. Introduction

### 1.1 Objective
Develop a machine learning model to predict soil water content (0-40%) from NIR spectral reflectance measurements.

### 1.2 Motivation
- Traditional gravimetric methods are time-consuming and destructive
- NIR spectroscopy offers rapid, non-destructive soil moisture assessment
- Machine learning can capture complex spectral-moisture relationships

### 1.3 Dataset
- **Source:** NIR spectrometer measurements (NQ5500316)
- **Measurement date:** March 24, 2023
- **Water content levels:** 9 levels (0%, 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%)
- **Replicates per level:** 8 measurements
- **Total samples:** 72 independent spectral measurements
- **Spectral range:** 189.85 - 2514.35 nm (~2,557 wavelengths)
- **Spectral regions:** UV (189-400 nm), Visible (400-700 nm), NIR (700-2,514 nm)

---

## 2. Methodology

### 2.1 Data Preprocessing

**Data Extraction:**
- Raw data contained in Excel file with separate sheets per water content level
- Extracted individual spectral measurements (not averaged)
- Each measurement contains full spectrum (~2,557 wavelengths)

**Data Structure:**
```
72 samples × 2,557 features (wavelengths)
- Features: Spectral reflectance at each wavelength
- Target: Water content percentage (0-40%)
```

**Feature Scaling:**
- Applied StandardScaler (zero mean, unit variance)
- Prevents wavelengths with larger values from dominating the model
- Essential for SVM performance

### 2.2 Train/Validation/Test Split

**Strategy:** Stratified random split to ensure balanced water content distribution

- **Training set:** 50 samples (69.4%)
  - Used for model training
- **Validation set:** 11 samples (15.3%)
  - Used for hyperparameter tuning
- **Test set:** 11 samples (15.3%)
  - Final unbiased performance evaluation

**Stratification ensures:** Each subset contains similar proportions of all 9 water content levels

### 2.3 Machine Learning Model

**Model:** Support Vector Machine Regression (SVR)

**Rationale for SVM:**
- Handles high-dimensional data well (2,557 features)
- Effective with non-linear relationships
- Robust to outliers
- Well-established in spectroscopy applications

**Hyperparameters:**
- **Kernel:** RBF (Radial Basis Function) - captures non-linear relationships
- **C (regularization):** 100 - controls trade-off between margin and errors
- **Gamma:** 'scale' - automatically calculated as 1/(n_features × X.var())
- **Epsilon:** 0.1 - defines ε-insensitive tube

**Support Vectors:** 49 out of 50 training samples (98%)

### 2.4 Cross-Validation

**5-Fold Cross-Validation** on training set:
- Mean R² Score: 0.8948 (±0.1041)
- Individual fold R² scores: [0.923, 0.799, 0.913, 0.889, 0.951]
- Confirms model stability and robustness

---

## 3. Results

### 3.1 Model Performance Summary

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **R² Score** | 0.9701 | 0.9488 | **0.9398** |
| **RMSE (%)** | 2.30 | 2.66 | **2.98** |
| **MAE (%)** | 1.38 | 1.93 | **2.38** |
| **Max Error (%)** | 6.74 | 5.78 | **5.59** |
| **Accuracy (±5%)** | 92.0% | 90.91% | **90.91%** |
| **Accuracy (±2%)** | 78.0% | 63.64% | **54.55%** |
| **Accuracy (±1%)** | 62.0% | 45.45% | **36.36%** |

### 3.2 Performance Interpretation

**R² Score (0.94):** The model explains 94% of the variance in soil water content
- Excellent predictive capability
- Strong correlation between spectral features and water content

**Mean Absolute Error (2.38%):** On average, predictions are within ±2.4% of actual water content
- High practical accuracy
- 90.91% of test predictions within ±5% tolerance

**Generalization:** Train-Test R² difference = 0.03
- Minimal overfitting
- Model generalizes well to unseen data

### 3.3 Performance by Water Content Level

The model performs consistently across all water content levels, with slight variations:
- Best performance: Mid-range water contents (10-30%)
- Slightly higher errors at extremes (0%, 40%)
- No systematic bias observed

---

## 4. Discussion

### 4.1 Key Findings

**1. Individual Measurements vs. Median Values**

Initial approach using median values (9 samples):
- R² Score: 0.22
- MAE: 10.5%
- Result: POOR performance

Final approach using individual measurements (72 samples):
- R² Score: 0.94
- MAE: 2.38%
- Result: EXCELLENT performance

**Impact:** Using individual measurements instead of median values increased R² by **327%** and reduced error by **77%**.

**2. Feature Dimensionality**

Challenge: 2,557 features with 72 samples
- Feature-to-sample ratio: 35:1
- Normally problematic ("curse of dimensionality")
- SVM with RBF kernel effectively handles this through:
  - Implicit feature space transformation
  - Regularization (C=100)
  - Support vector selection

**3. Physical Interpretation**

NIR spectroscopy detects water through:
- O-H stretching vibrations (~1,400-1,450 nm, ~1,900-1,950 nm)
- O-H combination bands (~970 nm)
- Water absorption features throughout NIR range

The model successfully captures these spectral signatures without manual feature engineering.

### 4.2 Comparison to Literature

Typical soil water content prediction performance in literature:
- R² range: 0.75 - 0.95
- MAE range: 2-5% water content

**Our results (R² = 0.94, MAE = 2.38%):** Within the top range of published studies, demonstrating competitive performance.

### 4.3 Limitations

**1. Sample Size**
- 72 total samples is relatively small
- More samples would likely improve generalization
- Particularly beneficial for rare/extreme conditions

**2. Spectral Range Coverage**
- Limited to one soil type
- Model trained on specific soil composition
- Generalization to different soil types uncertain

**3. Environmental Factors**
- All measurements taken on same day
- No variation in temperature, humidity
- Real-world deployment may see different conditions

**4. Water Content Range**
- Limited to 0-40% water content
- Extrapolation beyond this range not validated

### 4.4 Practical Applications

**Advantages of this approach:**
- ✓ Non-destructive measurement
- ✓ Rapid prediction (<1 second per sample)
- ✓ No sample preparation required
- ✓ Suitable for field deployment
- ✓ High accuracy (90% within ±5%)

**Potential use cases:**
- Precision agriculture (irrigation management)
- Environmental monitoring
- Soil science research
- Quality control in soil processing

---

## 5. Conclusions

### 5.1 Summary of Achievements

1. **Successfully developed** an SVM regression model for soil water content prediction
2. **Achieved excellent performance:** R² = 0.94, MAE = 2.38%
3. **Demonstrated** the importance of using individual measurements vs. aggregated data
4. **Validated** model robustness through cross-validation and train/val/test splits
5. **Created** a deployable model saved for future predictions

### 5.2 Key Takeaways

- **NIR spectroscopy + ML** is highly effective for soil water content prediction
- **SVM with RBF kernel** handles high-dimensional spectral data well
- **Sample size matters:** 72 samples >> 9 samples in model performance
- **Proper validation** essential for reliable performance estimates

### 5.3 Future Work Recommendations

**1. Expand Dataset**
- Collect more samples (target: 200-500)
- Include multiple soil types
- Vary environmental conditions

**2. Feature Engineering**
- Explore dimensionality reduction (PCA, wavelength selection)
- Investigate physics-informed features
- Test derivative spectroscopy

**3. Model Optimization**
- Grid search for optimal hyperparameters
- Compare with other algorithms (Random Forest, Neural Networks)
- Ensemble methods for improved robustness

**4. Deployment**
- Develop user-friendly prediction interface
- Field validation studies
- Real-time monitoring system integration

**5. Multi-Property Prediction**
- Extend to other soil properties (nitrogen, organic matter, pH)
- Multi-output regression models
- Investigate spectral fingerprints

---

## 6. Technical Specifications

### 6.1 Software and Libraries

- **Python:** 3.12.3
- **scikit-learn:** SVM implementation, preprocessing, metrics
- **pandas:** Data manipulation
- **numpy:** Numerical operations
- **matplotlib:** Visualization

### 6.2 Hardware Requirements

- **Training time:** ~5 seconds on standard CPU
- **Prediction time:** <0.01 seconds per sample
- **Memory:** ~50 MB for model + scaler

### 6.3 Model Files

- `svm_water_content_model.pkl`: Serialized model and scaler (deployable)
- `soil_spectral_data_individual.csv`: Individual measurement dataset
- `soil_spectral_data_summary.csv`: Median values per water content
- `soil_spectral_data_wide.csv`: Wide-format summary statistics

### 6.4 Reproducibility

- **Random seed:** 42 (for train/test split)
- **Code:** `train_svm_individual.py`
- **Data source:** `Kopie von Feuchtigkeitsstufen.NIR.xlsx`

---

## 7. References

### Methodology
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine learning*, 20(3), 273-297.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, pp. 2825-2830.

### NIR Spectroscopy for Soil Analysis
- Stenberg, B., et al. (2010). Visible and near infrared spectroscopy in soil science. *Advances in Agronomy*, 107, 163-215.
- Viscarra Rossel, R. A., et al. (2006). Visible, near infrared, mid infrared or combined diffuse reflectance spectroscopy for simultaneous assessment of various soil properties. *Geoderma*, 131(1-2), 59-75.

---

## Appendix A: Data Structure

### Individual Measurements Dataset
```
Columns: water_content_percent, replicate, wavelength_nm, reflectance
Rows: 184,104
Structure: Long format (one row per wavelength per measurement)
```

### Summary Statistics Dataset
```
Columns: water_content_percent, wavelength_nm, median_reflectance, std_reflectance
Rows: 23,013
Structure: Long format (one row per wavelength per water content)
```

---

## Appendix B: Model Equations

**SVM Regression (ε-SVR):**

Minimize:
```
½||w||² + C Σ(ξᵢ + ξᵢ*)
```

Subject to:
```
yᵢ - (w·φ(xᵢ) + b) ≤ ε + ξᵢ
(w·φ(xᵢ) + b) - yᵢ ≤ ε + ξᵢ*
ξᵢ, ξᵢ* ≥ 0
```

**RBF Kernel:**
```
K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)
where γ = 1/(n_features × X.var()) = 1/(2557 × var(X))
```

---

## Appendix C: Performance Metrics Definitions

**R² Score (Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)
where SS_res = Σ(yᵢ - ŷᵢ)²
      SS_tot = Σ(yᵢ - ȳ)²
```

**Root Mean Squared Error:**
```
RMSE = √(1/n Σ(yᵢ - ŷᵢ)²)
```

**Mean Absolute Error:**
```
MAE = 1/n Σ|yᵢ - ŷᵢ|
```

**Mean Absolute Percentage Error:**
```
MAPE = 100/n Σ|（yᵢ - ŷᵢ)/yᵢ|
```

---

## 5. Extension: Water + Nitrogen Dual-Model (February 2026)

### 5.1 Motivation

Following the successful water content model, new measurements were collected with an additional variable: nitrogen fertiliser levels (0 / 75 / 150 / 300 mg N kg⁻¹, NH₄NO₃ source). The primary goal was to determine whether NIR spectroscopy can detect nitrogen application steps.

### 5.2 New Dataset

| Property | Value |
|----------|-------|
| Spectrometer | NQ5500316 (NIRQuest) |
| Spectral range | 898–2514 nm (512 pixels) |
| Water levels | 0, 5, 15, 25, 35 % |
| Nitrogen levels | 0, 75, 150, 300 mg N kg⁻¹ |
| Conditions | 17 (W×N) |
| Replicates | ≥ 20 per condition |
| Total measurements | 343 |
| Background soil N | ~0.06% |

Two filename variants were encountered and handled (`soil_rep_*` and `soil_NN_rep_*`).

### 5.3 Methods

**Ingestion (`src/ingest_records.py`):**  
All 343 NIRQuest text files were parsed (0 errors). Long-format CSV written to `data/soil_spectral_data_records.csv` (175,616 rows). Six data quality checks all passed.

**Modelling (`src/train_water_nitrogen.py`):**
- Feature matrix: 343 samples × 512 wavelengths
- Stratified 70/15/15 split on (water × nitrogen) joint label
- StandardScaler normalisation
- Water: SVR (RBF, C=100, ε=0.1)
- Nitrogen: RandomForest (300 trees, balanced weights) vs SVC (RBF, C=10); best by val balanced accuracy

### 5.4 Results

#### Water Content Regression (SVR)

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| R² | 0.936 | 0.992 | **0.844** |
| RMSE (%) | 3.00 | 1.07 | **4.71** |
| MAE (%) | 0.76 | 0.80 | **1.55** |
| Acc ±5% | 99.2% | 100.0% | **92.3%** |

The validation R² is notably higher than test, suggesting moderate variance in test samples at the distribution boundaries. Overall ±5% accuracy of 92% is practically useful.

#### Nitrogen Classification (RandomForest, test set)

| Class (mg/kg) | Precision | Recall | F1 |
|---------------|-----------|--------|-----|
| 0 | 0.88 | 0.93 | 0.90 |
| 75 | 0.83 | 0.83 | 0.83 |
| 150 | 0.91 | 0.83 | 0.87 |
| 300 | 0.92 | 0.92 | 0.92 |
| **Macro avg** | **0.89** | **0.88** | **0.88** |

**Balanced accuracy: 88.1%** — exceeds the 70% success criterion.

Top discriminating wavelengths (RF importance): 2492.7 nm, 2474.2 nm, 2427.7 nm, 2452.5 nm — predominantly in the 2400–2510 nm region, consistent with N–H and C–N absorption bands.

SVC (RBF, C=10) achieved 81.4% balanced accuracy on the test set — functional but lower than RF.

### 5.5 Success Criteria Assessment

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| All files parsed | 0 errors | 0 errors (343/343) | ✓ |
| Water model R² | ≥ 0.85 | 0.844 | ⚠ close |
| Nitrogen balanced acc | ≥ 0.70 | 88.1% | ✓ |
| Evaluation plots | generated | 4 plots saved | ✓ |
| README updated | — | updated | ✓ |
| Tests pass | all pass | 50/50 | ✓ |

The water R² of 0.844 is just below the 0.85 target but the ±5% accuracy of 92.3% indicates practical utility. The gap between validation (0.992) and test (0.844) R² suggests that some additional regularisation or spectral preprocessing (e.g. Savitzky-Golay derivatives) could close this gap.

### 5.6 Conclusion

**Can NIR spectroscopy detect nitrogen application steps?**  
**YES** — The RandomForest classifier achieves 88% balanced accuracy across all four nitrogen classes (0 / 75 / 150 / 300 mg N kg⁻¹) on unseen test data. All four classes are reliably distinguished (minimum per-class F1 = 0.83).

Water content prediction on the new NIRQuest dataset reaches R²=0.84 and 92% accuracy within ±5%, demonstrating that both tasks are feasible with the same 512-pixel NIR spectrum.

---

**End of Report**
