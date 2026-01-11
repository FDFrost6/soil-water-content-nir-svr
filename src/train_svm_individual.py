"""
SVM Model Training with Individual Measurements (44 samples)
This script uses ALL individual spectral measurements, not just median values.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SVM MODEL - USING ALL INDIVIDUAL MEASUREMENTS")
print("="*70)
print(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load the individual measurements data
print("Loading individual measurement data...")
df = pd.read_csv('data/soil_spectral_data_individual.csv')
print(f"✓ Data loaded: {df.shape[0]} rows")

# Prepare the data - pivot to get one row per measurement
print("\nPreparing feature matrix...")
unique_combinations = df.groupby(['water_content_percent', 'replicate']).size().reset_index()[['water_content_percent', 'replicate']]
print(f"✓ Found {len(unique_combinations)} unique measurements")

# Create feature matrix: rows = individual measurements, columns = wavelengths
X_list = []
y_list = []

for idx, row in unique_combinations.iterrows():
    wc = row['water_content_percent']
    rep = row['replicate']
    
    # Get spectral data for this measurement
    measurement_data = df[(df['water_content_percent'] == wc) & (df['replicate'] == rep)]
    
    if len(measurement_data) > 0:
        # Sort by wavelength to ensure consistent order
        measurement_data = measurement_data.sort_values('wavelength_nm')
        spectral_values = measurement_data['reflectance'].values
        
        X_list.append(spectral_values)
        y_list.append(wc)

X = np.array(X_list)
y = np.array(y_list)

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Number of samples: {len(y)}")
print(f"✓ Number of features (wavelengths): {X.shape[1]}")
print(f"\nSamples per water content:")
for wc in sorted(np.unique(y)):
    count = np.sum(y == wc)
    print(f"  {int(wc):2d}%: {count} measurements")

# Split data: 70% train, 15% validation, 15% test
print("\n" + "="*70)
print("SPLITTING DATA")
print("="*70)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, shuffle=True, stratify=y_temp)

print(f"Training set:   {len(y_train)} samples ({len(y_train)/len(y)*100:.1f}%)")
print(f"Validation set: {len(y_val)} samples ({len(y_val)/len(y)*100:.1f}%)")
print(f"Test set:       {len(y_test)} samples ({len(y_test)/len(y)*100:.1f}%)")

# Standardize features
print("\n" + "="*70)
print("FEATURE SCALING")
print("="*70)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✓ Features standardized (zero mean, unit variance)")

# Train SVM model
print("\n" + "="*70)
print("TRAINING SVM MODEL")
print("="*70)
print("Hyperparameters:")
print("  - Kernel: RBF (Radial Basis Function)")
print("  - C (regularization): 100")
print("  - Gamma: scale")
print("  - Epsilon: 0.1")

svm_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
print("\nTraining model...")
svm_model.fit(X_train_scaled, y_train)
print("✓ Model training complete!")
print(f"  Number of support vectors: {len(svm_model.support_)}")

# Make predictions
print("\n" + "="*70)
print("MAKING PREDICTIONS")
print("="*70)
y_train_pred = svm_model.predict(X_train_scaled)
y_val_pred = svm_model.predict(X_val_scaled)
y_test_pred = svm_model.predict(X_test_scaled)
print("✓ Predictions generated for all datasets")

# Calculate metrics
def calculate_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Avoid division by zero in MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')
    
    max_error = np.max(np.abs(y_true - y_pred))
    
    tolerance_5 = np.sum(np.abs(y_true - y_pred) <= 5.0) / len(y_true) * 100
    tolerance_2 = np.sum(np.abs(y_true - y_pred) <= 2.0) / len(y_true) * 100
    tolerance_1 = np.sum(np.abs(y_true - y_pred) <= 1.0) / len(y_true) * 100
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  R² Score:                {r2:.4f}")
    print(f"  Root Mean Squared Error: {rmse:.4f}%")
    print(f"  Mean Absolute Error:     {mae:.4f}%")
    print(f"  Mean Absolute % Error:   {mape:.2f}%")
    print(f"  Maximum Error:           {max_error:.4f}%")
    print(f"  Accuracy (±5%):          {tolerance_5:.2f}%")
    print(f"  Accuracy (±2%):          {tolerance_2:.2f}%")
    print(f"  Accuracy (±1%):          {tolerance_1:.2f}%")
    
    return {
        'dataset': dataset_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'max_error': max_error,
        'accuracy_5pct': tolerance_5,
        'accuracy_2pct': tolerance_2,
        'accuracy_1pct': tolerance_1
    }

print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)

train_metrics = calculate_metrics(y_train, y_train_pred, "TRAINING SET")
val_metrics = calculate_metrics(y_val, y_val_pred, "VALIDATION SET")
test_metrics = calculate_metrics(y_test, y_test_pred, "TEST SET")

# Cross-validation
print("\n" + "="*70)
print("5-FOLD CROSS-VALIDATION")
print("="*70)
cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Save the model
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)
with open('models/svm_water_content_model.pkl', 'wb') as f:
    pickle.dump({'model': svm_model, 'scaler': scaler}, f)
print("✓ Model saved as 'models/svm_water_content_model.pkl'")

# Create visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('SVM Model Performance - Individual Measurements (44 samples)', 
             fontsize=16, fontweight='bold')

# 1. Training Set
ax1 = axes[0, 0]
ax1.scatter(y_train, y_train_pred, alpha=0.6, s=80, color='blue')
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Water Content (%)', fontsize=11)
ax1.set_ylabel('Predicted Water Content (%)', fontsize=11)
ax1.set_title(f'Training Set (n={len(y_train)})\nR² = {train_metrics["r2"]:.4f}', fontsize=12)
ax1.grid(True, alpha=0.3)

# 2. Validation Set
ax2 = axes[0, 1]
ax2.scatter(y_val, y_val_pred, alpha=0.6, s=80, color='green')
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Water Content (%)', fontsize=11)
ax2.set_ylabel('Predicted Water Content (%)', fontsize=11)
ax2.set_title(f'Validation Set (n={len(y_val)})\nR² = {val_metrics["r2"]:.4f}', fontsize=12)
ax2.grid(True, alpha=0.3)

# 3. Test Set
ax3 = axes[0, 2]
ax3.scatter(y_test, y_test_pred, alpha=0.6, s=80, color='orange')
ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Water Content (%)', fontsize=11)
ax3.set_ylabel('Predicted Water Content (%)', fontsize=11)
ax3.set_title(f'Test Set (n={len(y_test)})\nR² = {test_metrics["r2"]:.4f}', fontsize=12)
ax3.grid(True, alpha=0.3)

# 4-6. Residuals
for idx, (y_true, y_pred, title, color) in enumerate([
    (y_train, y_train_pred, 'Training Set', 'blue'),
    (y_val, y_val_pred, 'Validation Set', 'green'),
    (y_test, y_test_pred, 'Test Set', 'orange')
]):
    ax = axes[1, idx]
    residuals = y_pred - y_true
    ax.scatter(y_pred, residuals, alpha=0.6, s=80, color=color)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.axhline(y=5, color='orange', linestyle=':', lw=1.5, alpha=0.5)
    ax.axhline(y=-5, color='orange', linestyle=':', lw=1.5, alpha=0.5)
    ax.set_xlabel('Predicted Water Content (%)', fontsize=11)
    ax.set_ylabel('Residuals (%)', fontsize=11)
    ax.set_title(f'{title} Residuals', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/svm_model_individual_measurements.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'results/svm_model_individual_measurements.png'")

# Summary report
print("\n" + "="*70)
print("FINAL EVALUATION SUMMARY")
print("="*70)
print(f"\n{'Metric':<30} {'Training':<12} {'Validation':<12} {'Test':<12}")
print("-" * 70)
print(f"{'R² Score':<30} {train_metrics['r2']:<12.4f} {val_metrics['r2']:<12.4f} {test_metrics['r2']:<12.4f}")
print(f"{'RMSE (%)':<30} {train_metrics['rmse']:<12.4f} {val_metrics['rmse']:<12.4f} {test_metrics['rmse']:<12.4f}")
print(f"{'MAE (%)':<30} {train_metrics['mae']:<12.4f} {val_metrics['mae']:<12.4f} {test_metrics['mae']:<12.4f}")
print(f"{'MAPE (%)':<30} {train_metrics['mape']:<12.2f} {val_metrics['mape']:<12.2f} {test_metrics['mape']:<12.2f}")
print(f"{'Max Error (%)':<30} {train_metrics['max_error']:<12.4f} {val_metrics['max_error']:<12.4f} {test_metrics['max_error']:<12.4f}")
print(f"{'Accuracy (±5%)':<30} {train_metrics['accuracy_5pct']:<12.2f} {val_metrics['accuracy_5pct']:<12.2f} {test_metrics['accuracy_5pct']:<12.2f}")
print(f"{'Accuracy (±2%)':<30} {train_metrics['accuracy_2pct']:<12.2f} {val_metrics['accuracy_2pct']:<12.2f} {test_metrics['accuracy_2pct']:<12.2f}")

# Model assessment
print("\n" + "="*70)
print("MODEL ASSESSMENT")
print("="*70)

if test_metrics['r2'] > 0.95:
    print("✓ EXCELLENT: Test R² > 0.95")
elif test_metrics['r2'] > 0.90:
    print("✓ VERY GOOD: Test R² > 0.90")
elif test_metrics['r2'] > 0.80:
    print("✓ GOOD: Test R² > 0.80")
elif test_metrics['r2'] > 0.70:
    print("⚠ MODERATE: Test R² > 0.70")
else:
    print("✗ NEEDS IMPROVEMENT: Test R² < 0.70")

r2_diff = train_metrics['r2'] - test_metrics['r2']
if abs(r2_diff) < 0.1:
    print(f"✓ Good generalization: Train-Test R² difference = {r2_diff:.4f}")
elif r2_diff > 0.1:
    print(f"⚠ Possible overfitting: Train R² - Test R² = {r2_diff:.4f}")
else:
    print(f"⚠ Unusual: Test R² > Train R² (difference = {r2_diff:.4f})")

print("\n" + "="*70)
print("COMPARISON: 9 vs 44 SAMPLES")
print("="*70)
print("\nWith 9 samples (median values):")
print("  - R² Score: ~0.22")
print("  - MAE: ~10.5%")
print("  - Result: POOR")
print("\nWith 44 samples (individual measurements):")
print(f"  - R² Score: {test_metrics['r2']:.4f}")
print(f"  - MAE: {test_metrics['mae']:.4f}%")
if test_metrics['r2'] > 0.8:
    print("  - Result: MUCH BETTER! ✓")
elif test_metrics['r2'] > 0.5:
    print("  - Result: IMPROVED ✓")
else:
    print("  - Result: Still needs work")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  - models/svm_water_content_model.pkl (trained model + scaler)")
print("  - results/svm_model_individual_measurements.png (performance plots)")
print("\n" + "="*70)
