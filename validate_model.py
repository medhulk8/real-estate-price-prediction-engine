"""
Model validation and sanity checks
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from model import load_model_artifact

print("=" * 80)
print("MODEL VALIDATION & SANITY CHECKS")
print("=" * 80)

# Load data
df = pd.read_csv('transactions-2025-03-21.csv')
print(f"\nTotal rows: {len(df):,}")

# Load model artifact
artifact = load_model_artifact('models/trained_model.pkl')
metadata = artifact['preprocessing_metadata']

# Recreate the same splits
from preprocessing import (
    parse_dates, filter_by_procedure, remove_data_errors,
    chronological_split, run_full_preprocessing_pipeline
)

df = parse_dates(df)
print(f"\nAfter parsing dates: {len(df):,}")

# Filter to sale procedures only
sale_procedures = [p for p in df['PROCEDURE_EN'].unique() if 'Sale' in p or 'sales' in p.lower()]
if sale_procedures:
    df = filter_by_procedure(df, sale_procedures)
    print(f"After filtering to sales: {len(df):,}")
    print(f"Procedures kept: {sale_procedures}")

df = remove_data_errors(df, verbose=False)
print(f"After removing errors: {len(df):,}")

# Split
train_df, val_df, test_df = chronological_split(df, train_ratio=0.70, val_ratio=0.15)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

print(f"\nDate ranges:")
print(f"  Train: {train_df['INSTANCE_DATE'].min()} to {train_df['INSTANCE_DATE'].max()}")
print(f"  Val:   {val_df['INSTANCE_DATE'].min()} to {val_df['INSTANCE_DATE'].max()}")
print(f"  Test:  {test_df['INSTANCE_DATE'].min()} to {test_df['INSTANCE_DATE'].max()}")

# CHECK 1: Baseline models
print("\n" + "=" * 80)
print("CHECK 1: BASELINE MODELS")
print("=" * 80)

y_test = test_df['TRANS_VALUE'].values

# Baseline 1: Global median
global_median = train_df['TRANS_VALUE'].median()
baseline_1 = np.full(len(y_test), global_median)
r2_1 = r2_score(y_test, baseline_1)
mae_1 = mean_absolute_error(y_test, baseline_1)
mape_1 = mean_absolute_percentage_error(y_test, baseline_1) * 100

print(f"\n1. Global Median Baseline:")
print(f"   Median: ${global_median:,.0f}")
print(f"   R²: {r2_1:.4f}")
print(f"   MAE: ${mae_1:,.0f}")
print(f"   MAPE: {mape_1:.2f}%")

# Baseline 2: Area-only median
area_medians = train_df.groupby('AREA_EN')['TRANS_VALUE'].median()
baseline_2 = test_df['AREA_EN'].map(area_medians).fillna(global_median).values
r2_2 = r2_score(y_test, baseline_2)
mae_2 = mean_absolute_error(y_test, baseline_2)
mape_2 = mean_absolute_percentage_error(y_test, baseline_2) * 100

print(f"\n2. Area Median Baseline:")
print(f"   R²: {r2_2:.4f}")
print(f"   MAE: ${mae_2:,.0f}")
print(f"   MAPE: {mape_2:.2f}%")

# Baseline 3: Area + Property Type median
area_type_medians = train_df.groupby(['AREA_EN', 'PROP_TYPE_EN'])['TRANS_VALUE'].median()
test_df['key'] = list(zip(test_df['AREA_EN'], test_df['PROP_TYPE_EN']))
baseline_3 = test_df['key'].map(area_type_medians).fillna(test_df['AREA_EN'].map(area_medians)).fillna(global_median).values
r2_3 = r2_score(y_test, baseline_3)
mae_3 = mean_absolute_error(y_test, baseline_3)
mape_3 = mean_absolute_percentage_error(y_test, baseline_3) * 100

print(f"\n3. Area + Property Type Median Baseline:")
print(f"   R²: {r2_3:.4f}")
print(f"   MAE: ${mae_3:,.0f}")
print(f"   MAPE: {mape_3:.2f}%")

# CHECK 2: Verify log space evaluation
print("\n" + "=" * 80)
print("CHECK 2: LOG VS REAL SPACE VERIFICATION")
print("=" * 80)

# Run preprocessing pipeline
results = run_full_preprocessing_pipeline(df, keep_procedures=sale_procedures)
X_test = results['X_test']
y_test_log = results['y_test']
test_areas = results['test_areas']  # For price reconstruction

# Get predictions (handle both v1.0 and v2.0 models)
if 'models' in artifact:
    # v2.0: Use median model
    model = artifact['models']['median']
else:
    # v1.0: Use single model
    model = artifact['model']

y_pred_log = model.predict(X_test)

# Convert back to real space with price reconstruction
# Model predicts price_per_sqm, we reconstruct total price
y_pred_ppsm = np.expm1(y_pred_log)
y_test_ppsm = np.expm1(y_test_log)

# Reconstruct prices: price = price_per_sqm × area
y_pred_real = y_pred_ppsm * test_areas
y_test_real = y_test_ppsm * test_areas

# Metrics in both spaces
r2_log = r2_score(y_test_log, y_pred_log)
r2_real = r2_score(y_test_real, y_pred_real)
mae_log = mean_absolute_error(y_test_log, y_pred_log)
mae_real = mean_absolute_error(y_test_real, y_pred_real)

print(f"\nLog space:")
print(f"   R²: {r2_log:.4f}")
print(f"   MAE: {mae_log:.4f}")

print(f"\nReal space (correct):")
print(f"   R²: {r2_real:.4f}")
print(f"   MAE: ${mae_real:,.0f}")

# Sample predictions
print(f"\nSample predictions (first 5):")
print(f"{'True Price':>15} {'Pred Price':>15} {'Error':>15} {'Error %':>10}")
for i in range(min(5, len(y_test_real))):
    true_val = y_test_real[i]
    pred_val = y_pred_real[i]
    error = pred_val - true_val
    error_pct = (error / true_val) * 100
    print(f"${true_val:>14,.0f} ${pred_val:>14,.0f} ${error:>14,.0f} {error_pct:>9.1f}%")

# CHECK 3: Model vs Best Baseline
print("\n" + "=" * 80)
print("CHECK 3: MODEL VS BEST BASELINE")
print("=" * 80)

print(f"\nBest Baseline (Area+Type): R² = {r2_3:.4f}, MAE = ${mae_3:,.0f}")
print(f"CatBoost Model:            R² = {r2_real:.4f}, MAE = ${mae_real:,.0f}")

if r2_real < r2_3:
    print("\n⚠️  WARNING: Model is WORSE than simple baseline!")
    print("   This suggests a fundamental problem with the model.")
else:
    improvement = ((mae_3 - mae_real) / mae_3) * 100
    print(f"\n✓ Model beats baseline by {improvement:.1f}% MAE reduction")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nBaseline Performance:")
print(f"  Global Median:      R²={r2_1:.4f}, MAE=${mae_1:,.0f}")
print(f"  Area Median:        R²={r2_2:.4f}, MAE=${mae_2:,.0f}")
print(f"  Area+Type Median:   R²={r2_3:.4f}, MAE=${mae_3:,.0f}")
print(f"\nModel Performance:")
print(f"  CatBoost:           R²={r2_real:.4f}, MAE=${mae_real:,.0f}")

if r2_real > 0.5:
    print("\n✓ Model performance is good")
elif r2_real > 0.3:
    print("\n⚠️  Model performance is mediocre - room for improvement")
else:
    print("\n❌ Model performance is poor - needs investigation")

