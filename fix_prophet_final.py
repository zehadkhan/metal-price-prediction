#!/usr/bin/env python3
"""
Final Prophet Fix - Proper Prediction Alignment
Ensures R² is calculated correctly
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def rmse(a, b): 
    return mean_squared_error(a, b)**0.5

print("="*80)
print("FINAL PROPHET FIX - PROPER ALIGNMENT")
print("="*80)

# Load data
DATA_PATH = 'global_commodity_economy_cleaned.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

mask = df['Commodity'].astype(str).str.contains('Gold', case=False, na=False)
df_gold = df[mask][['Date', 'Close']].copy()
df_gold = df_gold.rename(columns={'Close': 'Price'})
df_gold = df_gold.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)

df_gold = df_gold.set_index('Date').sort_index()
df_gold = df_gold.asfreq('D', method='ffill')
df_gold = df_gold.interpolate(method='time').bfill().ffill()
df_gold = df_gold.reset_index()

# Prepare Prophet data
prophet_df = df_gold[['Date', 'Price']].copy()
prophet_df = prophet_df.rename(columns={'Date': 'ds', 'Price': 'y'})
prophet_df = prophet_df.dropna().reset_index(drop=True)

# Use log transformation
prophet_df_log = prophet_df.copy()
prophet_df_log['y'] = np.log(prophet_df_log['y'])

# Split properly
test_days = 60
val_days = 30

train_prop = prophet_df_log.iloc[:-test_days].copy()
val_prop = prophet_df_log.iloc[-test_days:-val_days].copy() if test_days > val_days else prophet_df_log.iloc[-test_days:].copy()
test_prop = prophet_df_log.iloc[-val_days:].copy() if test_days > val_days else prophet_df_log.iloc[-test_days:].copy()

print(f"Train: {len(train_prop)}, Val: {len(val_prop)}, Test: {len(test_prop)}")

# Train model
m = Prophet(
    changepoint_prior_scale=0.05,
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive',
    seasonality_prior_scale=10.0
)
m.fit(train_prop)

# Forecast for test period
future = m.make_future_dataframe(periods=len(test_prop))
fc = m.predict(future)

# CRITICAL: Properly align predictions with actual values
# Get test dates
test_dates = test_prop['ds'].values
actual_log = test_prop['y'].values

# Get predictions for exact test dates
pred_log = []
actual_aligned = []

for date in test_dates:
    date_pd = pd.to_datetime(date)
    # Find matching prediction
    pred_row = fc[fc['ds'] == date_pd]
    if len(pred_row) > 0:
        pred_val_log = pred_row['yhat'].values[0]
        pred_log.append(pred_val_log)
        # Find matching actual
        actual_row = test_prop[test_prop['ds'] == date_pd]
        if len(actual_row) > 0:
            actual_aligned.append(actual_row['y'].values[0])
        else:
            actual_aligned.append(np.nan)

# Convert to arrays
pred_log = np.array(pred_log)
actual_aligned = np.array(actual_aligned)

# Remove NaN
valid_mask = np.isfinite(pred_log) & np.isfinite(actual_aligned)
pred_log = pred_log[valid_mask]
actual_aligned = actual_aligned[valid_mask]

# Inverse log transform
pred_actual = np.exp(pred_log)
actual_actual = np.exp(actual_aligned)

# Calculate metrics
if len(pred_actual) > 0 and len(actual_actual) > 0:
    rmse_val = rmse(actual_actual, pred_actual)
    mae_val = mean_absolute_error(actual_actual, pred_actual)
    r2_val = r2_score(actual_actual, pred_actual)
    
    print(f"\n✅ PROPHET FIXED - Proper Alignment:")
    print(f"   RMSE: ${rmse_val:.2f}")
    print(f"   MAE:  ${mae_val:.2f}")
    print(f"   R²:   {r2_val:.4f}")
    
    if r2_val > 0.7:
        print(f"   Status: ✅ EXCELLENT!")
    elif r2_val > 0.5:
        print(f"   Status: ✅ GOOD!")
    elif r2_val > 0:
        print(f"   Status: ⚠️ FAIR")
    else:
        print(f"   Status: ❌ POOR")
        
    # Compare
    print(f"\n📊 COMPARISON:")
    print(f"   Previous (broken): R² = -1863.67")
    print(f"   Fixed:            R² = {r2_val:.4f}")
    improvement = "HUGE" if r2_val > 0 else "Still negative"
    print(f"   Improvement: {improvement}")
    
    # Save
    import os
    os.makedirs('reports', exist_ok=True)
    results = pd.DataFrame([['Prophet (Properly Fixed)', rmse_val, mae_val, r2_val]], 
                          columns=['Model', 'RMSE', 'MAE', 'R2'])
    results.to_csv('reports/prophet_fixed_final.csv', index=False)
    print(f"\n✅ Results saved to: reports/prophet_fixed_final.csv")
else:
    print("\n❌ Could not align predictions properly")

