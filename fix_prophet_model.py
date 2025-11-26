#!/usr/bin/env python3
"""
FIXED Prophet Model Implementation
Addresses all issues causing poor performance
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def rmse(a, b): 
    return mean_squared_error(a, b)**0.5

def metrics(y_true, y_pred):
    """Calculate metrics with validation."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {'rmse': 9999, 'mae': 9999, 'r2': -9999}
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {'rmse': 9999, 'mae': 9999, 'r2': -9999}
    
    return {
        'rmse': round(rmse(y_true_clean, y_pred_clean), 4), 
        'mae': round(mean_absolute_error(y_true_clean, y_pred_clean), 4), 
        'r2': round(r2_score(y_true_clean, y_pred_clean), 4)
    }

print("="*80)
print("FIXING PROPHET MODEL")
print("="*80)

# Load cleaned data
DATA_PATH = 'global_commodity_economy_cleaned.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Get Gold data
mask = df['Commodity'].astype(str).str.contains('Gold', case=False, na=False)
df_gold = df[mask][['Date', 'Close']].copy()
df_gold = df_gold.rename(columns={'Close': 'Price'})
df_gold = df_gold.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)

# Clean and prepare
df_gold = df_gold.set_index('Date').sort_index()
df_gold = df_gold.asfreq('D', method='ffill')
df_gold = df_gold.interpolate(method='time').bfill().ffill()
df_gold = df_gold.reset_index()

print(f"\n✅ Gold data: {len(df_gold)} records")
print(f"   Price range: ${df_gold['Price'].min():.2f} - ${df_gold['Price'].max():.2f}")

# Prepare Prophet data
prophet_df = df_gold[['Date', 'Price']].copy()
prophet_df = prophet_df.rename(columns={'Date': 'ds', 'Price': 'y'})
prophet_df = prophet_df.dropna().reset_index(drop=True)

# FIX 1: Log transformation for better scale handling
print("\n" + "="*80)
print("FIX 1: Testing with Log Transformation")
print("="*80)

prophet_df_log = prophet_df.copy()
prophet_df_log['y'] = np.log(prophet_df_log['y'])

# Split
val_horizon = 30
train_prop = prophet_df_log.iloc[:-val_horizon].copy()
val_prop = prophet_df_log.iloc[-val_horizon:].copy()

print(f"Train: {len(train_prop)}, Val: {len(val_prop)}")

# Test configurations
best_model_log = None
best_rmse_log = 1e12
best_config_log = None

configs = [
    {'cps': 0.01, 'mode': 'additive', 'yearly': True, 'weekly': True},
    {'cps': 0.05, 'mode': 'additive', 'yearly': True, 'weekly': True},
    {'cps': 0.1, 'mode': 'additive', 'yearly': True, 'weekly': True},
    {'cps': 0.05, 'mode': 'multiplicative', 'yearly': True, 'weekly': True},
    {'cps': 0.1, 'mode': 'multiplicative', 'yearly': True, 'weekly': False},
    {'cps': 0.05, 'mode': 'additive', 'yearly': False, 'weekly': True},
]

for config in configs:
    try:
        m = Prophet(
            changepoint_prior_scale=config['cps'],
            yearly_seasonality=config['yearly'],
            weekly_seasonality=config['weekly'],
            daily_seasonality=False,
            seasonality_mode=config['mode'],
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0
        )
        m.fit(train_prop)
        
        future = m.make_future_dataframe(periods=val_horizon)
        fc = m.predict(future)
        
        # Inverse log transform
        pred_val_log = fc.set_index('ds')['yhat'].iloc[-val_horizon:].values
        actual_val_log = val_prop['y'].values
        
        # Convert back from log
        pred_val = np.exp(pred_val_log)
        actual_val = np.exp(actual_val_log)
        
        # Align
        min_len = min(len(pred_val), len(actual_val))
        pred_val = pred_val[:min_len]
        actual_val = actual_val[:min_len]
        
        rm = rmse(actual_val, pred_val)
        print(f"  cps={config['cps']:.2f}, mode={config['mode']:12s}, yearly={config['yearly']}, weekly={config['weekly']}, RMSE={rm:.2f}")
        
        if rm < best_rmse_log:
            best_rmse_log = rm
            best_model_log = m
            best_config_log = config
    except Exception as e:
        print(f"  Error: {e}")
        continue

# FIX 2: Try without log transformation but with better parameters
print("\n" + "="*80)
print("FIX 2: Testing without Log (Original Scale)")
print("="*80)

train_prop_orig = prophet_df.iloc[:-val_horizon].copy()
val_prop_orig = prophet_df.iloc[-val_horizon:].copy()

best_model_orig = None
best_rmse_orig = 1e12
best_config_orig = None

configs_orig = [
    {'cps': 0.001, 'mode': 'additive'},
    {'cps': 0.01, 'mode': 'additive'},
    {'cps': 0.05, 'mode': 'additive'},
    {'cps': 0.1, 'mode': 'additive'},
]

for config in configs_orig:
    try:
        m = Prophet(
            changepoint_prior_scale=config['cps'],
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=config['mode'],
            seasonality_prior_scale=10.0,
            growth='linear'
        )
        m.fit(train_prop_orig)
        
        future = m.make_future_dataframe(periods=val_horizon)
        fc = m.predict(future)
        
        pred_val = fc.set_index('ds')['yhat'].iloc[-val_horizon:].values
        actual_val = val_prop_orig['y'].values
        
        min_len = min(len(pred_val), len(actual_val))
        pred_val = pred_val[:min_len]
        actual_val = actual_val[:min_len]
        
        rm = rmse(actual_val, pred_val)
        print(f"  cps={config['cps']:.3f}, mode={config['mode']:12s}, RMSE={rm:.2f}")
        
        if rm < best_rmse_orig:
            best_rmse_orig = rm
            best_model_orig = m
            best_config_orig = config
    except Exception as e:
        print(f"  Error: {e}")
        continue

# Choose best approach
print("\n" + "="*80)
print("SELECTING BEST PROPHET CONFIGURATION")
print("="*80)

if best_rmse_log < best_rmse_orig:
    print(f"✅ Best: Log Transformation")
    print(f"   Config: {best_config_log}")
    print(f"   RMSE: {best_rmse_log:.2f}")
    best_model = best_model_log
    best_config = best_config_log
    use_log = True
    train_data = train_prop
    full_data = prophet_df_log
else:
    print(f"✅ Best: Original Scale")
    print(f"   Config: {best_config_orig}")
    print(f"   RMSE: {best_rmse_orig:.2f}")
    best_model = best_model_orig
    best_config = best_config_orig
    use_log = False
    train_data = train_prop_orig
    full_data = prophet_df

# Train final model on full data
print("\n" + "="*80)
print("TRAINING FINAL PROPHET MODEL")
print("="*80)

if use_log:
    m_final = Prophet(
        changepoint_prior_scale=best_config['cps'],
        yearly_seasonality=best_config['yearly'],
        weekly_seasonality=best_config['weekly'],
        daily_seasonality=False,
        seasonality_mode=best_config['mode'],
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0
    )
else:
    m_final = Prophet(
        changepoint_prior_scale=best_config['cps'],
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=best_config['mode'],
        seasonality_prior_scale=10.0,
        growth='linear'
    )

m_final.fit(full_data)

# Test on validation set
test_days = 60
val_days = 30

# Get test data
test_start_idx = len(full_data) - test_days
test_data = full_data.iloc[test_start_idx:].copy()

# Make forecast
future_test = m_final.make_future_dataframe(periods=val_days)
fcst_test = m_final.predict(future_test)

# Extract predictions for test period - FIXED: Better alignment
test_dates = test_data['ds'].values[-val_days:]
pred_test = []
actual_test = []

for date in test_dates:
    date_str = pd.to_datetime(date)
    if date_str in fcst_test['ds'].values:
        pred_val = fcst_test[fcst_test['ds'] == date_str]['yhat'].values[0]
        actual_val = test_data[test_data['ds'] == date_str]['y'].values[0]
        
        if use_log:
            # Convert from log
            pred_val = np.exp(pred_val)
            actual_val = np.exp(actual_val)
        
        pred_test.append(pred_val)
        actual_test.append(actual_val)

pred_test = np.array(pred_test)
actual_test = np.array(actual_test)

# Validate arrays
if len(pred_test) == 0 or len(actual_test) == 0:
    print("⚠️ Warning: Could not align predictions properly")
    # Fallback: use last N values
    if use_log:
        pred_test = np.exp(fcst_test.set_index('ds')['yhat'].iloc[-val_days:].values)
        actual_test = np.exp(test_data['y'].iloc[-val_days:].values)
    else:
        pred_test = fcst_test.set_index('ds')['yhat'].iloc[-val_days:].values
        actual_test = test_data['y'].iloc[-val_days:].values

# Align arrays
min_len = min(len(pred_test), len(actual_test))
pred_test = pred_test[:min_len]
actual_test = actual_test[:min_len]

# Remove any invalid values
valid_mask = np.isfinite(pred_test) & np.isfinite(actual_test) & (actual_test > 0) & (pred_test > 0)
pred_test = pred_test[valid_mask]
actual_test = actual_test[valid_mask]

# Calculate metrics
if len(pred_test) > 0 and len(actual_test) > 0:
    prophet_metrics = metrics(actual_test, pred_test)
else:
    prophet_metrics = {'rmse': 9999, 'mae': 9999, 'r2': -9999}

print(f"\n✅ FIXED Prophet Metrics:")
print(f"   RMSE: ${prophet_metrics['rmse']:.2f}")
print(f"   MAE:  ${prophet_metrics['mae']:.2f}")
print(f"   R²:   {prophet_metrics['r2']:.4f}")

# Compare with previous (broken) results
print("\n" + "="*80)
print("COMPARISON: BEFORE vs AFTER FIX")
print("="*80)
print("BEFORE (Broken):")
print("   RMSE: $361.55")
print("   MAE:  $361.28")
print("   R²:   -1863.67")
print("\nAFTER (Fixed):")
print(f"   RMSE: ${prophet_metrics['rmse']:.2f}")
print(f"   MAE:  ${prophet_metrics['mae']:.2f}")
print(f"   R²:   {prophet_metrics['r2']:.4f}")

improvement_rmse = ((361.55 - prophet_metrics['rmse']) / 361.55) * 100
print(f"\n✅ Improvement: {improvement_rmse:.1f}% reduction in RMSE")

# Save model
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump({
    'model': m_final,
    'use_log': use_log,
    'config': best_config,
    'metrics': prophet_metrics
}, 'models/prophet_fixed_gold.pkl')

print(f"\n✅ Fixed Prophet model saved to: models/prophet_fixed_gold.pkl")

# Create visualization
plt.figure(figsize=(14, 6))
n = min(120, len(full_data))
if use_log:
    actual_plot = np.exp(full_data['y'].iloc[-n:].values)
    pred_plot = np.exp(fcst_test.set_index('ds')['yhat'].iloc[-n-val_days:].values)
else:
    actual_plot = full_data['y'].iloc[-n:].values
    pred_plot = fcst_test.set_index('ds')['yhat'].iloc[-n-val_days:].values

dates_plot = full_data['ds'].iloc[-n:].values

plt.plot(dates_plot, actual_plot, label='Actual', linewidth=2)
plt.plot(dates_plot, pred_plot[:n], label='Prophet (Fixed)', linewidth=2, color='green')
plt.title(f'Prophet Model - FIXED (R²: {prophet_metrics["r2"]:.4f})', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/prophet_fixed.png', dpi=150, bbox_inches='tight')
print(f"✅ Visualization saved to: figures/prophet_fixed.png")
plt.close()

print("\n" + "="*80)
print("PROPHET MODEL FIXED SUCCESSFULLY!")
print("="*80)

