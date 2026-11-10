#!/usr/bin/env python3
"""
Test Models with Cleaned Data
Compare performance and fix issues
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def rmse(a, b): 
    return mean_squared_error(a, b)**0.5

def metrics(y_true, y_pred):
    return {
        'rmse': round(rmse(y_true, y_pred), 4), 
        'mae': round(mean_absolute_error(y_true, y_pred), 4), 
        'r2': round(r2_score(y_true, y_pred), 4)
    }

print("="*80)
print("TESTING MODELS WITH CLEANED DATA")
print("="*80)

# Load CLEANED data
DATA_PATH = 'global_commodity_economy_cleaned.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n✅ Loaded cleaned data: {len(df)} records")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

# Get Gold data
mask = df['Commodity'].astype(str).str.contains('Gold', case=False, na=False)
df_gold = df[mask][['Date', 'Close']].rename(columns={'Close': 'Price'})
df_gold = df_gold.sort_values('Date').drop_duplicates('Date')

# Merge macro features
for col in ['Oil_Price', 'USD_Index', 'CPI']:
    if col in df.columns:
        df_gold = df_gold.merge(df[['Date', col]].drop_duplicates('Date'), on='Date', how='left')

# Clean and prepare
df_gold = df_gold.set_index('Date').asfreq('D')
df_gold = df_gold.interpolate(method='time').bfill().ffill()
df_gold = df_gold.reset_index()

# Feature engineering
df_feat = df_gold.copy()
price_col = 'Price'

# Lags
for l in [1, 2, 7, 14, 30]:
    df_feat[f'lag_{l}'] = df_feat[price_col].shift(l)

# Rolling stats
df_feat['ma7'] = df_feat[price_col].rolling(7).mean()
df_feat['ma30'] = df_feat[price_col].rolling(30).mean()
df_feat['std7'] = df_feat[price_col].rolling(7).std()

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

df_feat = df_feat.dropna().reset_index(drop=True)

print(f"✅ Feature engineering complete: {len(df_feat)} records")

# Train/Test split
test_days = 60
val_days = 30

train_df = df_feat.iloc[:-test_days]
val_df = df_feat.iloc[-test_days:-val_days] if test_days > val_days else df_feat.iloc[-test_days:]
test_df = df_feat.iloc[-val_days:] if test_days > val_days else df_feat.iloc[-test_days:]

# Feature selection
feature_cols = ['lag_1', 'lag_2', 'lag_7', 'ma7', 'ma30', 'std7', 'return_1']
for col in ['Oil_lag1', 'USD_lag1', 'CPI']:
    if col in df_feat.columns:
        feature_cols.append(col)
feature_cols = [c for c in feature_cols if c in df_feat.columns]

X_train = train_df[feature_cols]
y_train = train_df[price_col]
X_test = test_df[feature_cols]
y_test = test_df[price_col]

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# Test Ridge Regression
print("\n" + "="*80)
print("TESTING RIDGE REGRESSION")
print("="*80)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

lr = Ridge(alpha=1.0).fit(X_train_s, y_train)
pred_lr = lr.predict(X_test_s)

lr_metrics = metrics(y_test, pred_lr)
print(f"Ridge Regression Metrics:")
print(f"  RMSE: ${lr_metrics['rmse']:.2f}")
print(f"  MAE:  ${lr_metrics['mae']:.2f}")
print(f"  R²:   {lr_metrics['r2']:.4f}")

# Test Prophet
print("\n" + "="*80)
print("TESTING PROPHET (IMPROVED)")
print("="*80)

prophet_df = df_feat[['Date', price_col]].rename(columns={'Date': 'ds', price_col: 'y'})

# Split for validation
val_horizon = 30
train_prop = prophet_df.iloc[:-val_horizon].copy()
val_prop = prophet_df.iloc[-val_horizon:].copy()

print(f"Prophet: Train={len(train_prop)}, Val={len(val_prop)}")

# Try multiple Prophet configurations
best_model = None
best_rmse = 1e12
best_config = None

configs = [
    {'cps': 0.01, 'mode': 'additive'},
    {'cps': 0.05, 'mode': 'additive'},
    {'cps': 0.1, 'mode': 'additive'},
    {'cps': 0.01, 'mode': 'multiplicative'},
    {'cps': 0.05, 'mode': 'multiplicative'},
    {'cps': 0.1, 'mode': 'multiplicative'},
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
        
        rm = rmse(actual_val, pred_val)
        print(f"  cps={config['cps']:.2f}, mode={config['mode']:12s}, RMSE={rm:.2f}")
        
        if rm < best_rmse:
            best_rmse = rm
            best_model = m
            best_config = config
    except Exception as e:
        print(f"  Error: {e}")
        continue

if best_model:
    print(f"\n✅ Best Prophet: cps={best_config['cps']}, mode={best_config['mode']}, RMSE={best_rmse:.2f}")
    
    # Final Prophet forecast
    future_30 = best_model.make_future_dataframe(periods=30)
    fcst_30 = best_model.predict(future_30)
    
    # Evaluate on test set
    fcst_indexed = fcst_30.set_index('ds')
    val_ds = val_prop['ds'].values
    
    prophet_val_pred = []
    prophet_val_actual = []
    for ds in val_ds:
        if ds in fcst_indexed.index:
            prophet_val_pred.append(fcst_indexed.loc[ds]['yhat'])
            prophet_val_actual.append(val_prop[val_prop['ds'] == ds]['y'].values[0])
    
    prophet_val_pred = np.array(prophet_val_pred)
    prophet_val_actual = np.array(prophet_val_actual)
    
    prop_metrics = metrics(prophet_val_actual, prophet_val_pred)
    print(f"\nProphet Metrics (on validation):")
    print(f"  RMSE: ${prop_metrics['rmse']:.2f}")
    print(f"  MAE:  ${prop_metrics['mae']:.2f}")
    print(f"  R²:   {prop_metrics['r2']:.4f}")
else:
    print("\n❌ Prophet failed to train")
    prop_metrics = {'rmse': 9999, 'mae': 9999, 'r2': -9999}

# Compare with last 30 days
pred_lr_last30 = pred_lr[-30:] if len(pred_lr) >= 30 else pred_lr
actual_last30 = y_test[-30:].values if len(y_test) >= 30 else y_test.values
prophet_last30 = fcst_30.set_index('ds')['yhat'].iloc[-30:].values if best_model else None

print("\n" + "="*80)
print("FINAL COMPARISON (Last 30 days)")
print("="*80)

lr_final = metrics(actual_last30, pred_lr_last30)
print(f"Ridge Regression:")
print(f"  RMSE: ${lr_final['rmse']:.2f}")
print(f"  MAE:  ${lr_final['mae']:.2f}")
print(f"  R²:   {lr_final['r2']:.4f}")

if prophet_last30 is not None:
    prop_final = metrics(actual_last30, prophet_last30)
    print(f"\nProphet:")
    print(f"  RMSE: ${prop_final['rmse']:.2f}")
    print(f"  MAE:  ${prop_final['mae']:.2f}")
    print(f"  R²:   {prop_final['r2']:.4f}")

# Ensemble
if prophet_last30 is not None:
    w_lr = 1.0 / (rmse(actual_last30, pred_lr_last30) + 1e-9)
    w_prop = 1.0 / (rmse(actual_last30, prophet_last30) + 1e-9)
    weights = np.array([w_lr, w_prop])
    preds = np.array([pred_lr_last30, prophet_last30])
    ensemble = (weights.reshape(-1, 1) * preds).sum(axis=0) / weights.sum()
    
    ens_metrics = metrics(actual_last30, ensemble)
    print(f"\nEnsemble:")
    print(f"  RMSE: ${ens_metrics['rmse']:.2f}")
    print(f"  MAE:  ${ens_metrics['mae']:.2f}")
    print(f"  R²:   {ens_metrics['r2']:.4f}")

# Save results
results_df = pd.DataFrame([
    ['Ridge Regression', lr_final['rmse'], lr_final['mae'], lr_final['r2']],
], columns=['Model', 'RMSE', 'MAE', 'R2'])

if prophet_last30 is not None:
    results_df = pd.concat([results_df, pd.DataFrame([[
        'Prophet', prop_final['rmse'], prop_final['mae'], prop_final['r2']
    ]], columns=results_df.columns)], ignore_index=True)
    
    results_df = pd.concat([results_df, pd.DataFrame([[
        'Ensemble', ens_metrics['rmse'], ens_metrics['mae'], ens_metrics['r2']
    ]], columns=results_df.columns)], ignore_index=True)

os.makedirs('reports', exist_ok=True)
results_df.to_csv('reports/test_results_cleaned_data.csv', index=False)

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print(results_df.to_string(index=False))
print("\n✅ Test complete! Results saved to reports/test_results_cleaned_data.csv")

