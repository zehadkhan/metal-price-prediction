#!/usr/bin/env python3
"""
FIXED Test with Cleaned Data
Properly aligned and validated
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def rmse(a, b): 
    return mean_squared_error(a, b)**0.5

def metrics(y_true, y_pred):
    """Calculate metrics with validation."""
    # Remove any NaN or inf values
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
print("FIXED TEST WITH CLEANED DATA")
print("="*80)

# Load CLEANED data
DATA_PATH = 'global_commodity_economy_cleaned.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n✅ Loaded cleaned data: {len(df)} records")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

# Get Gold data - FIXED: Proper filtering
mask = df['Commodity'].astype(str).str.contains('Gold', case=False, na=False)
df_gold = df[mask][['Date', 'Close']].copy()
df_gold = df_gold.rename(columns={'Close': 'Price'})
df_gold = df_gold.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)

print(f"\n✅ Gold data: {len(df_gold)} records")
print(f"   Price range: ${df_gold['Price'].min():.2f} - ${df_gold['Price'].max():.2f}")

# Merge macro features
for col in ['Oil_Price', 'USD_Index', 'CPI']:
    if col in df.columns:
        df_gold = df_gold.merge(
            df[['Date', col]].drop_duplicates('Date'), 
            on='Date', 
            how='left'
        )

# Set index and fill
df_gold = df_gold.set_index('Date').sort_index()
df_gold = df_gold.asfreq('D', method='ffill')
df_gold = df_gold.interpolate(method='time').bfill().ffill()
df_gold = df_gold.reset_index()

# Feature engineering
df_feat = df_gold.copy()
price_col = 'Price'

print(f"\n🔧 Feature engineering...")

# Lags
for l in [1, 2, 7, 14, 30]:
    df_feat[f'lag_{l}'] = df_feat[price_col].shift(l)

# Rolling stats
df_feat['ma7'] = df_feat[price_col].rolling(7, min_periods=1).mean()
df_feat['ma30'] = df_feat[price_col].rolling(30, min_periods=1).mean()
df_feat['std7'] = df_feat[price_col].rolling(7, min_periods=1).std()

# Returns
df_feat['return_1'] = df_feat[price_col].pct_change(1)
df_feat['return_7'] = df_feat[price_col].pct_change(7)

# Date features
df_feat['dayofweek'] = df_feat['Date'].dt.dayofweek
df_feat['month'] = df_feat['Date'].dt.month

# Macro lags
if 'Oil_Price' in df_feat.columns:
    df_feat['Oil_lag1'] = df_feat['Oil_Price'].shift(1)
if 'USD_Index' in df_feat.columns:
    df_feat['USD_lag1'] = df_feat['USD_Index'].shift(1)

# Drop rows with NaN in critical features
df_feat = df_feat.dropna(subset=[price_col]).reset_index(drop=True)

# Ensure we have enough data
min_required = 100
if len(df_feat) < min_required:
    raise ValueError(f"Not enough data: {len(df_feat)} records (need at least {min_required})")

print(f"✅ Features ready: {len(df_feat)} records")

# Train/Test split - FIXED: Proper split
test_days = 60
val_days = 30

if len(df_feat) < test_days + val_days + 100:
    test_days = min(30, len(df_feat) // 4)
    val_days = min(15, len(df_feat) // 8)

train_df = df_feat.iloc[:-test_days].copy()
test_df = df_feat.iloc[-test_days:].copy()

print(f"\n📊 Data Split:")
print(f"   Train: {len(train_df)} records")
print(f"   Test:  {len(test_df)} records")

# Feature selection
feature_cols = ['lag_1', 'lag_2', 'lag_7', 'ma7', 'ma30', 'std7', 'return_1']
for col in ['Oil_lag1', 'USD_lag1', 'CPI']:
    if col in df_feat.columns:
        feature_cols.append(col)
feature_cols = [c for c in feature_cols if c in df_feat.columns and c in train_df.columns]

# Remove any columns with all NaN
feature_cols = [c for c in feature_cols if not train_df[c].isna().all()]

X_train = train_df[feature_cols].fillna(0)
y_train = train_df[price_col]
X_test = test_df[feature_cols].fillna(0)
y_test = test_df[price_col]

print(f"   Features: {len(feature_cols)}")
print(f"   Feature names: {feature_cols}")

# Validate data
print(f"\n📊 Data Validation:")
print(f"   y_train range: ${y_train.min():.2f} - ${y_train.max():.2f}")
print(f"   y_test range:  ${y_test.min():.2f} - ${y_test.max():.2f}")

# Test Ridge Regression
print("\n" + "="*80)
print("TESTING RIDGE REGRESSION")
print("="*80)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

lr = Ridge(alpha=1.0).fit(X_train_s, y_train)
pred_lr = lr.predict(X_test_s)

# Validate predictions
print(f"\n📊 Prediction Validation:")
print(f"   pred_lr range: ${pred_lr.min():.2f} - ${pred_lr.max():.2f}")
print(f"   y_test range:  ${y_test.min():.2f} - ${y_test.max():.2f}")

lr_metrics = metrics(y_test.values, pred_lr)
print(f"\n✅ Ridge Regression Metrics:")
print(f"   RMSE: ${lr_metrics['rmse']:.2f}")
print(f"   MAE:  ${lr_metrics['mae']:.2f}")
print(f"   R²:   {lr_metrics['r2']:.4f}")

# Test Prophet
print("\n" + "="*80)
print("TESTING PROPHET")
print("="*80)

prophet_df = df_feat[['Date', price_col]].copy()
prophet_df = prophet_df.rename(columns={'Date': 'ds', price_col: 'y'})
prophet_df = prophet_df.dropna().reset_index(drop=True)

val_horizon = min(30, len(prophet_df) // 10)
train_prop = prophet_df.iloc[:-val_horizon].copy()
val_prop = prophet_df.iloc[-val_horizon:].copy()

print(f"   Train: {len(train_prop)}, Val: {len(val_prop)}")

# Simple Prophet (no regressors for stability)
best_model = None
best_rmse = 1e12
best_config = None

configs = [
    {'cps': 0.05, 'mode': 'additive'},
    {'cps': 0.1, 'mode': 'additive'},
    {'cps': 0.05, 'mode': 'multiplicative'},
]

for config in configs:
    try:
        m = Prophet(
            changepoint_prior_scale=config['cps'],
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=config['mode']
        )
        m.fit(train_prop)
        
        future = m.make_future_dataframe(periods=val_horizon)
        fc = m.predict(future)
        
        pred_val = fc.set_index('ds')['yhat'].iloc[-val_horizon:].values
        actual_val = val_prop['y'].values
        
        # Align arrays
        min_len = min(len(pred_val), len(actual_val))
        pred_val = pred_val[:min_len]
        actual_val = actual_val[:min_len]
        
        rm = rmse(actual_val, pred_val)
        print(f"   cps={config['cps']:.2f}, mode={config['mode']:12s}, RMSE={rm:.2f}")
        
        if rm < best_rmse:
            best_rmse = rm
            best_model = m
            best_config = config
    except Exception as e:
        print(f"   Error: {e}")
        continue

if best_model:
    print(f"\n✅ Best Prophet: cps={best_config['cps']}, mode={best_config['mode']}, RMSE={best_rmse:.2f}")
    
    # Final forecast
    future_30 = best_model.make_future_dataframe(periods=min(30, len(test_df)))
    fcst_30 = best_model.predict(future_30)
    
    # Evaluate on test set
    test_dates = test_df['Date'].values
    prophet_pred = []
    prophet_actual = []
    
    for date in test_dates:
        date_str = pd.to_datetime(date)
        if date_str in fcst_30['ds'].values:
            pred_val = fcst_30[fcst_30['ds'] == date_str]['yhat'].values[0]
            actual_val = test_df[test_df['Date'] == date][price_col].values[0]
            prophet_pred.append(pred_val)
            prophet_actual.append(actual_val)
    
    if len(prophet_pred) > 0:
        prophet_pred = np.array(prophet_pred)
        prophet_actual = np.array(prophet_actual)
        
        prop_metrics = metrics(prophet_actual, prophet_pred)
        print(f"\n✅ Prophet Metrics:")
        print(f"   RMSE: ${prop_metrics['rmse']:.2f}")
        print(f"   MAE:  ${prop_metrics['mae']:.2f}")
        print(f"   R²:   {prop_metrics['r2']:.4f}")
    else:
        print("\n❌ Could not align Prophet predictions")
        prop_metrics = {'rmse': 9999, 'mae': 9999, 'r2': -9999}
        prophet_pred = None
else:
    print("\n❌ Prophet failed")
    prop_metrics = {'rmse': 9999, 'mae': 9999, 'r2': -9999}
    prophet_pred = None

# Final comparison
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

results = []
results.append(['Ridge Regression', lr_metrics['rmse'], lr_metrics['mae'], lr_metrics['r2']])

if prophet_pred is not None:
    results.append(['Prophet', prop_metrics['rmse'], prop_metrics['mae'], prop_metrics['r2']])
    
    # Ensemble
    min_len = min(len(pred_lr), len(prophet_pred), len(y_test))
    pred_lr_align = pred_lr[:min_len]
    prophet_align = prophet_pred[:min_len]
    actual_align = y_test.values[:min_len]
    
    w_lr = 1.0 / (rmse(actual_align, pred_lr_align) + 1e-9)
    w_prop = 1.0 / (rmse(actual_align, prophet_align) + 1e-9)
    weights = np.array([w_lr, w_prop])
    preds = np.array([pred_lr_align, prophet_align])
    ensemble = (weights.reshape(-1, 1) * preds).sum(axis=0) / weights.sum()
    
    ens_metrics = metrics(actual_align, ensemble)
    results.append(['Ensemble', ens_metrics['rmse'], ens_metrics['mae'], ens_metrics['r2']])

results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MAE', 'R2'])

os.makedirs('reports', exist_ok=True)
results_df.to_csv('reports/test_results_cleaned_data_FIXED.csv', index=False)

print("\n" + results_df.to_string(index=False))
print("\n✅ Results saved to reports/test_results_cleaned_data_FIXED.csv")

