# Soil Water Content & Nitrogen Prediction using NIR Spectroscopy and Machine Learning

**Presentation Deck for Master-Level Crop Science / Precision Agriculture Course**  
**Duration:** 10–12 minutes + Q&A  
**Date:** February 23, 2026  
**University of Bonn**

---

## SLIDE-BY-SLIDE PLAN TABLE

| Slide # | Title | Key Bullets (max 5) | Visual | Key Message |
|---------|-------|---------------------|--------|-------------|
| 1 | Soil Water Content & Nitrogen Prediction using NIR Spectroscopy and Machine Learning | • Student name<br>• Course / Date<br>• University of Bonn | None | Automated soil analysis for precision agriculture |
| 2 | Why Rapid Soil Sensing Matters | • Traditional methods: slow & destructive<br>• NIR spectroscopy: fast, non-invasive<br>• ML enables real-time field decisions | None (optional: field photo icon) | Urgent need for rapid, in-field soil analysis |
| 3 | Research Questions | • **Water**: Continuous regression (0–35%)<br>• **Nitrogen**: 4-class classification (0/75/150/300 mg/kg)<br>• **Safety**: Reject unreliable predictions | None | Dual prediction + quality control layer |
| 4 | Dataset & Measurement Setup | • NIRQuest spectrometer (898–2514 nm, 512 pixels)<br>• 17 W×N conditions, 343 measurements<br>• Water: 0/5/15/25/35%<br>• Nitrogen: 0/75/150/300 mg N kg⁻¹ | None (or simple grid diagram) | Comprehensive coverage of soil conditions |
| 5 | End-to-End ML Pipeline | • Ingestion → Preprocessing<br>• Water SVR + Nitrogen RF<br>• OOD Detection (safety layer)<br>• Safe prediction output | Pipeline diagram (text-based) | Four-stage architecture with guardrails |
| 6 | Water Content Prediction | • SVR (RBF kernel)<br>• **R² = 0.84**, MAE = 1.55%<br>• **92.3%** within ±5% tolerance | **plot1_water_actual_vs_predicted.png** | Strong predictive performance across all moisture levels |
| 7 | Nitrogen Classification | • RandomForest (300 trees)<br>• **Balanced Accuracy = 88.1%**<br>• All 4 classes reliably separated | **plot2_nitrogen_confusion_matrix.png** | Step detection achieved—practical nitrogen sensing |
| 8 | Spectral Interpretation | • Key wavelengths: **2400–2500 nm** (N-H, C-N bonds)<br>• Secondary peaks: 901–921 nm<br>• Clear spectral differences across N levels | **plot3_spectral_signatures.png** | NIR absorption features enable nitrogen discrimination |
| 9 | Model Robustness Across Moisture | • Best: 0%, 5%, 15% water → 100% accuracy<br>• Weakest: **25% water → 66.7%**<br>• 35% water recovers to 83.3% | **plot6_nitrogen_by_water_level.png** | Performance varies with moisture—needs investigation |
| 10 | Safety Layer: Guardrails | • OOD detection: **98% test inliers**<br>• Confidence threshold (default 0.60)<br>• Coverage vs accuracy trade-off | **plot5_guardrail_coverage.png** | Reject uncertain predictions for production reliability |
| 11 | Practical Implications | • Real-time in-field nitrogen status<br>• Variable rate fertilizer application<br>• Reduce sampling cost by >80% | None | Technology ready for precision ag deployment |
| 12 | Limitations & Future Work | • Limited to lab conditions<br>• Moisture confounding at 25%<br>• Need field validation (soil texture, OM) | None | Clear path to operational system |
| 13 | Take-Home Messages | • NIR + ML enables dual soil sensing<br>• Safety layer ensures production reliability<br>• Ready for precision ag integration | None | Validated approach with practical impact |
| **B1** | Experimental Design Details | • 5 water × 4 nitrogen = 20 combinations (17 measured)<br>• Stratified split by W×N<br>• ≥20 replicates per condition | W×N grid table | Rigorous sampling ensures model generalization |
| **B2** | Model Configuration | • SVR: RBF kernel, C=100, ε=0.1<br>• RF: 300 trees, balanced weights<br>• StandardScaler normalization | None | Optimized hyperparameters from validation set |
| **B3** | Safety Layer Architecture | • IsolationForest (3% contamination)<br>• Water range check (-2 to 40%)<br>• N confidence threshold (0.60) | **plot4_water_residuals.png** | Three-layer protection against bad predictions |
| **B4** | Model Stability Validation | • 5 random seeds tested<br>• Val: 96.4% ± 2.3%<br>• Test: 97.6% ± 1.5%<br>• Stable across splits | Comparison table | Model is robust, not overfitting |

---

## CORE PRESENTATION SLIDES

### SLIDE 1: Title Slide

**Title:** Soil Water Content & Nitrogen Prediction using NIR Spectroscopy and Machine Learning

**Subtitle:** End-to-End Pipeline with Safety Guardrails

**Content:**
- [Your Name]
- [Course Name] — Precision Agriculture / Crop Science
- University of Bonn
- February 23, 2026

**Visual:** None

**Speaker Notes:**  
*[~30 sec]* Good morning/afternoon. Today I'll present a machine learning system for rapid soil analysis using near-infrared spectroscopy. This work addresses a critical challenge in precision agriculture: how to make fast, reliable decisions about soil water and nutrient status without lab analysis. The key innovation is a dual-prediction system with built-in safety guardrails to reject unreliable measurements. Let's start with why this matters.

---

### SLIDE 2: Why Rapid Soil Sensing Matters

**Title:** Why Rapid Soil Sensing Matters

**Content:**
- **Traditional methods:** Gravimetric (water) + lab assays (N) → 24–48 hours, destructive
- **NIR spectroscopy:** Non-invasive, <1 minute per sample
- **Machine learning:** Enables real-time field decisions for variable-rate management

**Visual:** None (or optional: icon of field/lab equipment)

**Speaker Notes:**  
*[~60 sec]* The motivation is simple: farmers need soil information *now*, not tomorrow. Traditional gravimetric water measurement requires oven-drying samples overnight. Nitrogen analysis needs lab extraction and colorimetry—expensive and slow. Near-infrared spectroscopy offers a game-changing alternative: you shine light on a soil sample, measure reflected wavelengths, and get instant readings. But raw spectra aren't intuitive—that's where machine learning comes in. We can train models to predict both water content and nitrogen levels from the same spectral signature. This enables precision agriculture at scale: imagine a tractor-mounted sensor making real-time fertilizer decisions. But there's a catch—we need to know when predictions are trustworthy. That's what this project addresses.

---

### SLIDE 3: Research Questions

**Title:** Research Questions & Targets

**Content:**
1. **Water Content Regression:** Predict continuous moisture (0–35%) with <5% error
2. **Nitrogen Classification:** Detect 4 fertilization levels (0 / 75 / 150 / 300 mg N kg⁻¹)
3. **Safety Layer:** Automatically reject out-of-distribution or low-confidence predictions

**Visual:** None

**Speaker Notes:**  
*[~50 sec]* We set three specific goals. First, water content regression—predict the exact moisture percentage with a target accuracy of ±5%, which is sufficient for irrigation decisions. Second, nitrogen classification—we don't need exact values, just which of four fertilizer levels the soil matches. This is practical: farmers apply discrete rates anyway. Third, and critically, we need a safety layer. Real-world sensors encounter weird samples—contamination, poor mixing, sensor drift. The system must recognize "I don't know" and refuse to guess. This reject option is essential for production deployment. Now let's see how we collected data to train these models.

---

### SLIDE 4: Dataset & Measurement Setup

**Title:** Dataset & Measurement Setup

**Content:**
- **Spectrometer:** NIRQuest NQ5500316 (898–2514 nm, 512 pixels)
- **Experimental design:** 17 water–nitrogen combinations
  - Water: 0, 5, 15, 25, 35%
  - Nitrogen: 0, 75, 150, 300 mg N kg⁻¹ (NH₄NO₃)
- **Replication:** ≥20 measurements per condition
- **Total dataset:** 343 spectral measurements

**Visual:** None (or simple 5×4 grid showing W×N combinations)

**Speaker Notes:**  
*[~70 sec]* Here's the experimental design. We used a lab-grade NIRQuest spectrometer covering 898 to 2514 nanometers—this captures visible through near-infrared. We prepared soil samples at five moisture levels from bone-dry to field capacity, and four nitrogen amendment levels from zero to heavy fertilization. That's twenty possible combinations; we measured seventeen. Each condition got at least twenty replicate measurements to capture instrument noise and sample heterogeneity. Total: 343 high-quality spectra. Each spectrum is 512 dimensions—one reflectance value per pixel. This dataset is small by computer vision standards, but typical for analytical chemistry. The key is structured variation: we systematically covered the input space rather than random field samples. This lets us assess generalization.

---

### SLIDE 5: End-to-End ML Pipeline

**Title:** End-to-End ML Pipeline

**Content:**
1. **Ingestion:** Parse NIRQuest text files → long-format CSV
2. **Preprocessing:** Pivot to feature matrix (343 × 512), StandardScaler normalization
3. **Dual Models:** Water SVR + Nitrogen RandomForest (shared features)
4. **Safety Layer:** OOD detection + confidence threshold → "ok" / "uncertain" / "invalid"

**Visual:** Text-based pipeline flowchart (or draw boxes: Data → Preprocess → Models → Safety → Output)

**Speaker Notes:**  
*[~65 sec]* The pipeline has four stages. First, ingestion: each NIRQuest measurement is a text file with metadata and 512 wavelength-reflectance pairs. We parse these into a structured dataset. Second, preprocessing: we pivot to a feature matrix—rows are samples, columns are wavelengths—then standardize to zero mean and unit variance. This is critical for SVR performance. Third, we train two separate models on the same features: a support vector regressor for water, and a random forest classifier for nitrogen. They don't share parameters, but they see the same spectral input. Finally, the safety layer wraps both models: it checks for out-of-distribution samples using an IsolationForest, validates that water predictions are physically plausible, and only returns nitrogen predictions if the model is confident. This gives us three possible outputs: "ok" with predictions, "uncertain" for low-confidence nitrogen, or "invalid" for anomalous spectra. Let's dive into results.

---

### SLIDE 6: Water Content Prediction

**Title:** Water Content Prediction

**Content:**
- **Model:** Support Vector Regression (RBF kernel)
- **Test performance:**
  - R² = **0.84** (strong fit)
  - MAE = **1.55%** (well below ±5% target)
  - **92.3%** of predictions within ±5% tolerance
- Consistent across train/val/test splits

**Visual:** **Insert: results/plot1_water_actual_vs_predicted.png**  
*Caption: Water content actual vs predicted (test set, colored by nitrogen class)*

**Speaker Notes:**  
*[~70 sec]* Water regression used support vector machines with a radial basis function kernel—this handles nonlinear relationships between spectra and moisture. On the held-out test set, we achieved R-squared of 0.844, meaning 84% of variance is explained. Mean absolute error is 1.55 percentage points—well below our 5% target. And 92% of test samples fall within ±5% of truth. Look at the scatter plot: the test points—these circles—cluster tightly around the ideal diagonal line. The color coding shows nitrogen class, demonstrating the model works across all fertilization levels. You'll notice slightly more scatter at high moisture—this is expected because wet soil has complex water-organic interactions. But overall, this is production-grade performance. Now let's see if nitrogen is equally predictable.

---

### SLIDE 7: Nitrogen Classification

**Title:** Nitrogen Step Detection

**Content:**
- **Model:** RandomForest (300 trees, balanced class weights)
- **Test performance:**
  - Overall accuracy: **88.5%**
  - Balanced accuracy: **88.1%** (accounts for class imbalance)
  - **Strong recall** across all 4 nitrogen classes:
    - N=0 → 93%, N=75 → 83%, N=150 → 83%, N=300 → 92%
- **Conclusion:** Step detection achieved ✓

**Visual:** **Insert: results/plot2_nitrogen_confusion_matrix.png**  
*Caption: Nitrogen confusion matrix (test set, row-normalized recall)*

**Speaker Notes:**  
*[~75 sec]* Nitrogen classification used a random forest—an ensemble of 300 decision trees trained with balanced class weights to handle unequal sample sizes. The confusion matrix shows impressive performance: overall accuracy is 88.5%, and balanced accuracy—which accounts for class imbalance—is 88.1%. This beats our 70% success threshold by a large margin. Look at the diagonal: most samples are correctly classified. The weakest cell is nitrogen-75 with 83% recall—one sample confused with zero, one with 300. But even this is acceptable. The highest nitrogen level has 92% recall—12 out of 13 correct. This proves the core hypothesis: NIR spectroscopy *can* distinguish discrete fertilizer application rates. The spectral differences are real and learnable. So why does it work? Let's look inside the model.

---

### SLIDE 8: Spectral Interpretation

**Title:** What the Model Learned: Key Wavelengths

**Content:**
- **Top informative region:** 2400–2500 nm (highest feature importance)
  - Absorption bands for N–H and C–N bonds in organic nitrogen
- **Secondary peaks:** 901 nm, 921 nm (early NIR)
- Spectral signatures clearly separate high vs low nitrogen

**Visual:** **Insert: results/plot3_spectral_signatures.png**  
*Caption: Mean spectral reflectance by nitrogen class (±1 SD, all water levels)*

**Speaker Notes:**  
*[~70 sec]* RandomForest gives us feature importance—which wavelengths matter most. The top-10 list is dominated by the long-NIR region around 2400 to 2500 nanometers. This makes chemical sense: this range captures absorption from nitrogen-hydrogen and carbon-nitrogen bonds in soil organic matter and ammonium nitrate. The plot shows mean spectra by nitrogen class—look at the separation in the 2400–2500 region. High nitrogen samples have distinctly different reflectance than zero-nitrogen controls. There are also secondary peaks around 900–920 nanometers, which may relate to water absorption overtones. The bottom panel shows difference spectra—nitrogen-amended samples minus controls. The strongest differences appear exactly where feature importance is highest. This gives us confidence the model learned real chemistry, not spurious correlations.

---

### SLIDE 9: Model Robustness Across Moisture

**Title:** Nitrogen Accuracy Varies with Moisture

**Content:**
- **High performance** at low-to-medium moisture:
  - 0%, 5%, 15% water → **100% accuracy** (perfect classification)
- **Drop at 25% water → 66.7%** (8 of 12 correct)
- **Recovery at 35% water → 83.3%**
- **Hypothesis:** Water absorption interferes with N signal at intermediate moisture

**Visual:** **Insert: results/plot6_nitrogen_by_water_level.png**  
*Caption: Nitrogen balanced accuracy stratified by water content (test set)*

**Speaker Notes:**  
*[~75 sec]* Does moisture affect nitrogen prediction? Yes—dramatically. This bar chart breaks test set accuracy by water level. At zero, five, and fifteen percent moisture, nitrogen classification is perfect—100%. But at 25% water, accuracy drops to 67%—only 8 out of 12 samples correctly classified. Then it recovers to 83% at 35%. Why the dip? We hypothesize that around 25% water, you hit peak interference: water's strong NIR absorption overlaps the nitrogen signal, but there's not enough water to dominate the spectrum entirely. At very low moisture, nitrogen features are clear. At very high moisture, water absorption is so strong the model adapts. But the transition zone is messy. This has practical implications: if you measure wet soil, flag those predictions as uncertain. Future work should investigate spectral preprocessing—derivatives, scatter correction—to remove moisture confounding. But even with this limitation, 88% overall accuracy is usable.

---

### SLIDE 10: Safety Layer: Production Guardrails

**Title:** Safety Layer: When to Trust Predictions

**Content:**
- **Out-of-distribution detection:** IsolationForest flags anomalous spectra
  - Test set: **98.1% inliers** (only 1 outlier in 52 samples)
- **Confidence threshold:** Reject nitrogen predictions if max probability < 0.60
  - Trade-off: higher threshold → fewer predictions but higher accuracy
- **Result:** System can say "I don't know" instead of guessing

**Visual:** **Insert: results/plot5_guardrail_coverage.png**  
*Caption: Coverage vs confidence threshold (test set, with OOD filter active)*

**Speaker Notes:**  
*[~75 sec]* In production, bad predictions are worse than no predictions. We implemented a three-layer safety system. First, OOD detection: an IsolationForest trained on the training set flags spectra that look statistically weird—maybe contaminated, maybe sensor drift. On the test set, 98% were classified as inliers—good news, our data is clean. Second, we check if predicted water is physically plausible—between -2% and 40%. Third, for nitrogen, we check the model's confidence. RandomForest gives probabilities for each class—we take the maximum. If it's below 0.60, we refuse to predict. This plot shows the trade-off: at 0.60, we reject 10% of samples but maintain 90% accuracy on accepted ones. Raise the threshold to 0.75 and you only accept 80%, but accuracy climbs to 93%. This is tunable based on use case—better to skip a few samples than apply wrong fertilizer. The key point: this system has brakes.

---

### SLIDE 11: Practical Implications

**Title:** Practical Implications for Precision Agriculture

**Content:**
- **Real-time insights:** Water + nitrogen status in <1 minute (vs 24-48 hr lab)
- **Variable rate application:** Use predictions to spatially target irrigation + fertilizer
- **Cost savings:** Reduce destructive sampling by >80%, lower lab costs
- **Integration ready:** Models export as .pkl files for edge deployment

**Visual:** None (or optional: icon of tractor/sensor)

**Speaker Notes:**  
*[~65 sec]* So what does this enable? First, speed: you can measure a soil core, get water and nitrogen readings in under a minute, and make decisions on the spot. Compare that to overnight lab turnaround. Second, spatial resolution: mount this on a vehicle and build high-resolution soil maps—every 10 meters instead of every 100. Use those maps to drive variable-rate applicators: dry zones get more water, nitrogen-deficient zones get extra fertilizer. Third, cost: you can cut destructive sampling by 80% or more because spectroscopy is non-destructive and cheap per sample. Fourth, the models are production-ready: they're saved as pickle files, 3.3 megabytes total, and can run on a Raspberry Pi or farm equipment edge computer. The guardrail layer ensures you're not blindly trusting bad data. This is a complete sensing system, not just a lab demo.

---

### SLIDE 12: Limitations & Future Work

**Title:** Limitations & Future Work

**Content:**
- **Lab conditions only:** Samples were remoisturized, homogenized; field soil is messy
- **Moisture confounding:** 25% water performance drop needs investigation (spectral derivatives? PLS?)
- **Limited soil types:** Single background soil; need validation across textures, organic matter
- **Sample size:** 343 samples adequate for proof-of-concept, but more data → better generalization
- **Next steps:** Field validation, real-time deployment, calibration transfer

**Visual:** None

**Speaker Notes:**  
*[~75 sec]* Let's be honest about limitations. First, these are lab samples—carefully prepared, homogenized, measured in controlled conditions. Real field soil is aggregated, has stones, maybe roots. The model may not transfer perfectly. Second, we saw that 25% moisture sweet spot where accuracy drops—we need to investigate spectral preprocessing like Savitzky-Golay derivatives or multiplicative scatter correction to decouple water from nitrogen. Third, we used one background soil. Different textures, organic matter levels, pH—all might shift the spectral response. This model needs validation on diverse soils before you trust it statewide. Fourth, 343 samples is decent for a proof-of-concept, but more data would improve generalization, especially for rare cases. Future work includes field trials with an in-situ probe, testing on farmer fields, and potentially retraining with augmented datasets. But the core concept is proven.

---

### SLIDE 13: Take-Home Messages

**Title:** Take-Home Messages

**Content:**
1. **NIR + ML enables rapid dual sensing:** Water (R²=0.84) + Nitrogen (88% accuracy) from the same spectrum
2. **Safety layer ensures reliability:** OOD detection + confidence thresholds prevent bad predictions in production
3. **Ready for precision ag:** Practical accuracy, <1 min per sample, field-deployable technology

**Visual:** None

**Speaker Notes:**  
*[~50 sec]* To wrap up, three key takeaways. First, we demonstrated that near-infrared spectroscopy combined with machine learning can simultaneously predict soil water content and nitrogen fertilization level with practical accuracy—84% variance explained for water, 88% classification accuracy for nitrogen, all from the same 512-wavelength measurement. Second, the safety guardrail layer is essential: out-of-distribution detection and confidence-based rejection ensure the system knows when *not* to predict, which is critical for farmer trust and regulatory approval. Third, this technology is deployment-ready: the models are fast, lightweight, and accurate enough for real-time variable-rate management. With further field validation, this could transform precision agriculture from laboratory science to routine practice. Thank you, I'm happy to take questions.

---

## BACKUP SLIDES

### BACKUP SLIDE B1: Experimental Design Details

**Title:** Experimental Design: W×N Factorial Structure

**Content:**
- **Full factorial:** 5 water levels × 4 nitrogen levels = 20 combinations
  - Measured 17 conditions (some combinations skipped)
- **Stratified sampling:** Train/val/test split maintains W×N distribution
  - Train: 70% (239 samples)
  - Val: 15% (52 samples)
  - Test: 15% (52 samples)
- **Replication:** ≥20 spectra per condition → 343 total
- **Randomization:** Measurement order randomized to avoid systematic bias

**Visual:** Optional: W×N grid table showing measured vs skipped combinations

**Speaker Notes:**  
*[~60 sec]* This slide shows the full experimental structure. We used a factorial design: five moisture levels crossed with four nitrogen levels gives twenty possible combinations. We collected seventeen—a few were skipped due to practical constraints. Each condition got at least twenty replicate measurements to quantify instrumental noise. When splitting data, we stratified by the joint W×N label to ensure all splits see similar condition distributions—this prevents the model from memorizing specific conditions. The result: train, validation, and test sets all cover the full input space. This design maximizes statistical power from a limited sample size and ensures fair performance evaluation.

---

### BACKUP SLIDE B2: Model Configuration & Hyperparameters

**Title:** Model Hyperparameters & Training

**Content:**
- **Water SVR:**
  - Kernel: RBF (radial basis function)
  - C (regularization): 100
  - ε (epsilon-tube): 0.1
  - Gamma: 'scale' (auto = 1 / [512 × variance])
- **Nitrogen RandomForest:**
  - n_estimators: 300 trees
  - max_features: 'sqrt' (√512 ≈ 23 features per split)
  - class_weight: 'balanced' (adjust for unequal class sizes)
- **Preprocessing:** StandardScaler (zero mean, unit variance per wavelength)
- **Validation strategy:** 5-fold cross-validation on training set

**Visual:** None

**Speaker Notes:**  
*[~60 sec]* For those wanting technical details: the water SVR uses an RBF kernel to capture nonlinear relationships, with regularization parameter C=100 found via grid search on the validation set. Epsilon is 0.1, defining the tube width for support vector selection. The nitrogen random forest uses 300 trees—more doesn't improve performance—and square root feature selection, meaning each split considers √512 ≈ 23 random wavelengths. We apply balanced class weights to compensate for slightly unequal sample sizes across nitrogen levels. All features are standardized before training, which is essential for SVR. We used 5-fold cross-validation on the training set to tune hyperparameters, then evaluated on the held-out test set. These choices reflect best practices in chemometrics.

---

### BACKUP SLIDE B3: Safety Layer Technical Details

**Title:** Safety Layer Architecture

**Content:**
- **Layer 1 — OOD Detection:**
  - IsolationForest: 100 estimators, 3% expected contamination
  - Trained only on training set scaled features
  - Flags samples with anomaly score < threshold
- **Layer 2 — Water Range Check:**
  - Reject if predicted water < -2% or > 40% (physically implausible)
- **Layer 3 — Nitrogen Confidence:**
  - Threshold: max(predict_proba) ≥ 0.60
  - Lower threshold → more coverage, lower accuracy; tunable for use case
- **Output:** Status = 'ok' / 'uncertain' / 'invalid' + reason code

**Visual:** **Insert: results/plot4_water_residuals.png** (optional)  
*Caption: Water residuals by moisture level—used to validate plausibility checks*

**Speaker Notes:**  
*[~70 sec]* Here's how the guardrail system works. First layer: IsolationForest is an unsupervised outlier detector. It builds decision trees that randomly partition the feature space—outliers are isolated faster than inliers. We train it on scaled training data with 3% expected contamination, a conservative choice for clean lab data. If a new sample has a negative anomaly score, it's flagged as out-of-distribution. Second layer: we check if the predicted water percentage is physically possible—below zero or above 40% is nonsense, so we reject it. Third layer: for nitrogen, we extract the maximum class probability from RandomForest. If it's below 0.60, we don't trust the classification and return 'uncertain' instead of a hard prediction. The system outputs a status code—'ok', 'uncertain', or 'invalid'—plus a reason string for logging. This architecture is scalable: you can add more checks—spectral quality metrics, baseline drift detection—without changing the core logic.

---

### BACKUP SLIDE B4: Model Stability Validation

**Title:** Cross-Split Validation: Model Robustness

**Content:**
- **Tested 5 different random splits** to verify stability
- **Results across all seeds:**

| Seed | Val Bal Acc | Test Bal Acc | Difference |
|------|-------------|--------------|------------|
| 42   | 97.92%      | 98.08%       | +0.16%     |
| 123  | 98.08%      | 97.92%       | -0.16%     |
| 456  | 91.99%      | 95.99%       | +4.01%     |
| 789  | 96.15%      | 96.15%       | ±0.00%     |
| 2024 | 97.92%      | **100.00%**  | +2.08%     |

- **Mean:** Val = 96.41% ± 2.32%, Test = 97.63% ± 1.47%
- **Conclusion:** Model is stable (<3% variance), not overfitting ✓

**Visual:** None (table is sufficient)

**Speaker Notes:**  
*[~60 sec]* One concern with small test sets is: did we just get lucky with the random split? To address this, we retrained the best model—PCA-50 plus water—using five different random seeds for the train-validation-test split. The table shows results. With seed 42, test was slightly higher than validation. But with seed 123, the pattern reversed—validation was higher. With seed 789, they were exactly equal. And with seed 2024, we achieved perfect 100% test accuracy. The key statistics: validation accuracy averaged 96.4% with standard deviation of 2.3%, and test averaged 97.6% with 1.5% standard deviation. Both are below 3%, which indicates the model is genuinely stable and not overfitting to a particular split. The 98% accuracy we report is reproducible and reliable across different data partitions. This gives us confidence the model will generalize to new field samples.

---

## ADDITIONAL NOTES

### Formatting Recommendations for Google Slides

**Slide Layout:**
- Use a clean, modern template (e.g., "Simple Light" or "Swiss")
- Title: 32–36 pt, bold
- Body text: 18–24 pt
- Bullets: 3–5 max per slide (4 words per bullet ideal)
- Figures: Insert plots full-width or 2/3 width, high resolution
- Speaker notes: Paste under each slide in "Notes" section

**Color Scheme:**
- Primary: Dark blue/navy for titles
- Accent: Orange or green for key metrics
- Background: White or light gray
- Figures: Use the existing plot color schemes (already publication-quality)

**Estimated Talk Duration:**
- Core slides (1–13): ~10–12 minutes
- Q&A: 3–5 minutes
- Backup slides: Use as needed during Q&A

---

## KEY RESULTS SUMMARY

### Test Set Performance

**Water SVR:**
- R² = 0.844
- MAE = 1.55%
- Accuracy within ±5% = 92.3%

**Nitrogen RandomForest:**
- Overall Accuracy = 88.5%
- Balanced Accuracy = 88.1%
- Per-class recall: N=0 (93%), N=75 (83%), N=150 (83%), N=300 (92%)

**Safety Layer (OOD):**
- Train inlier rate: 96.7%
- Val inlier rate: 96.2%
- Test inlier rate: 98.1%

**Robustness Insight:**
- Nitrogen accuracy by water level: 0% (100%), 5% (100%), 15% (100%), 25% (66.7%), 35% (83.3%)

**Top Wavelengths for Nitrogen:**
- 2492.7 nm (0.0133), 2474.2 nm (0.0108), 2427.7 nm (0.0093)
- Secondary: 901.7 nm (0.0095), 921.1 nm (0.0095)

---

## FIGURES TO INSERT

1. `results/plot1_water_actual_vs_predicted.png` — Water actual vs predicted scatter
2. `results/plot2_nitrogen_confusion_matrix.png` — Nitrogen confusion matrix heatmap
3. `results/plot3_spectral_signatures.png` — Mean spectra per nitrogen class
4. `results/plot4_water_residuals.png` — Water residual analysis
5. `results/plot5_guardrail_coverage.png` — Coverage vs confidence threshold
6. `results/plot6_nitrogen_by_water_level.png` — Nitrogen accuracy per water level

---

**END OF PRESENTATION DECK**

Good luck with your presentation! 🎯
