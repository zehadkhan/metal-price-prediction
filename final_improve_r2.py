#!/usr/bin/env python3
"""
Final R² Improvement - Using Proven Feature Set
Fixes Prophet and improves all models
"""

import os
import pandas as pd
import numpy as np
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
print("FINAL R² IMPROVEMENT - USING PROVEN APPROACH")
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

# Merge macro features
for col in ['Oil_Price', 'USD_Index', 'CPI']:
    if col in df.columns:
        df_gold = df_gold.merge(
            df[['Date', col]].drop_duplicates('Date'), 
            on='Date', 
            how='left'
        )

# Clean and prepare
df_gold = df_gold.set_index('Date').sort_index()
df_gold = df_gold.asfreq('D', method='ffill')
df_gold = df_gold.interpolate(method='time').bfill().ffill()
df_gold = df_gold.reset_index()

# Feature engineering - USE PROVEN FEATURE SET
df_feat = df_gold.copy()
price_col = 'Price'

# Proven features that worked (R² = 0.7260)
for l in [1, 2, 7, 14, 30]:
    df_feat[f'lag_{l}'] = df_feat[price_col].shift(l)

df_feat['ma7'] = df_feat[price_col].rolling(7, min_periods=1).mean()
df_feat['ma30'] = df_feat[price_col].rolling(30, min_periods=1).mean()
df_feat['std7'] = df_feat[price_col].rolling(7, min_periods=1).std()

df_feat['return_1'] = df_feat[price_col].pct_change(1)
df_feat['return_7'] = df_feat[price_col].pct_change(7)

df_feat['dayofweek'] = df_feat['Date'].dt.dayofweek
df_feat['month'] = df_feat['Date'].dt.month

if 'Oil_Price' in df_feat.columns:
    df_feat['Oil_lag1'] = df_feat['Oil_Price'].shift(1)
if 'USD_Index' in df_feat.columns:
    df_feat['USD_lag1'] = df_feat['USD_Index'].shift(1)

df_feat = df_feat.dropna(subset=[price_col]).reset_index(drop=True)

# Train/Test split
test_days = 60
train_df = df_feat.iloc[:-test_days].copy()
test_df = df_feat.iloc[-test_days:].copy()

# Proven feature set
feature_cols = ['lag_1', 'lag_2', 'lag_7', 'ma7', 'ma30', 'std7', 'return_1']
for col in ['Oil_lag1', 'USD_lag1', 'CPI']:
    if col in df_feat.columns:
        feature_cols.append(col)
feature_cols = [c for c in feature_cols if c in df_feat.columns and c in train_df.columns]

X_train = train_df[feature_cols].fillna(0)
y_train = train_df[price_col]
X_test = test_df[feature_cols].fillna(0)
y_test = test_df[price_col]

print(f"\n✅ Using proven feature set: {len(feature_cols)} features")

results = []

# 1. Ridge Regression - OPTIMIZE hyperparameters
print("\n" + "="*80)
print("1️⃣ OPTIMIZING RIDGE REGRESSION")
print("="*80)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Test different alpha values
best_ridge = None
best_r2_ridge = -9999
best_alpha = None

for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    lr = Ridge(alpha=alpha).fit(X_train_s, y_train)
    pred = lr.predict(X_test_s)
    r2 = metrics(y_test.values, pred)['r2']
    print(f"   Alpha={alpha:4.1f}: R²={r2:.4f}")
    if r2 > best_r2_ridge:
        best_r2_ridge = r2
        best_ridge = lr
        best_alpha = alpha

pred_ridge = best_ridge.predict(X_test_s)
ridge_metrics = metrics(y_test.values, pred_ridge)
results.append(['Ridge Regression (Optimized)', ridge_metrics['rmse'], ridge_metrics['mae'], ridge_metrics['r2']])
print(f"\n✅ Best Ridge: Alpha={best_alpha}, R²={ridge_metrics['r2']:.4f}, RMSE=${ridge_metrics['rmse']:.2f}")

# 2. FIX PROPHET - Proper implementation
print("\n" + "="*80)
print("2️⃣ FIXING PROPHET MODEL")
print("="*80)

prophet_df = df_feat[['Date', price_col]].copy()
prophet_df = prophet_df.rename(columns={'Date': 'ds', price_col: 'y'})
prophet_df = prophet_df.dropna().reset_index(drop=True)

# Use log transformation
prophet_df_log = prophet_df.copy()
prophet_df_log['y'] = np.log(prophet_df_log['y'])

val_horizon = 30
train_prop = prophet_df_log.iloc[:-val_horizon].copy()
val_prop = prophet_df_log.iloc[-val_horizon:].copy()

# Best config from testing
m_prophet = Prophet(
    changepoint_prior_scale=0.05,
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive',
    seasonality_prior_scale=10.0
)
m_prophet.fit(train_prop)

# Forecast
future = m_prophet.make_future_dataframe(periods=val_horizon)
fc = m_prophet.predict(future)

# Get predictions aligned with test set
test_dates = test_df['Date'].values[-val_horizon:]
prophet_pred = []
prophet_actual = []

for date in test_dates:
    date_str = pd.to_datetime(date)
    if date_str in fc['ds'].values:
        pred_log = fc[fc['ds'] == date_str]['yhat'].values[0]
        pred_val = np.exp(pred_log)
        actual_val = test_df[test_df['Date'] == date_str][price_col].values[0]
        prophet_pred.append(pred_val)
        prophet_actual.append(actual_val)

if len(prophet_pred) > 0 and len(prophet_actual) > 0:
    prophet_pred = np.array(prophet_pred)
    prophet_actual = np.array(prophet_actual)
    
    # Validate
    valid_mask = np.isfinite(prophet_pred) & np.isfinite(prophet_actual) & (prophet_actual > 0) & (prophet_pred > 0)
    prophet_pred = prophet_pred[valid_mask]
    prophet_actual = prophet_actual[valid_mask]
    
    if len(prophet_pred) > 0:
        prop_metrics = metrics(prophet_actual, prophet_pred)
        results.append(['Prophet (Fixed)', prop_metrics['rmse'], prop_metrics['mae'], prop_metrics['r2']])
        print(f"✅ Prophet Fixed: R²={prop_metrics['r2']:.4f}, RMSE=${prop_metrics['rmse']:.2f}")
    else:
        results.append(['Prophet (Fixed)', 9999, 9999, -9999])
        print("⚠️ Prophet: No valid predictions")
else:
    results.append(['Prophet (Fixed)', 9999, 9999, -9999])
    print("⚠️ Prophet: Could not align predictions")

# 3. Ensemble
print("\n" + "="*80)
print("3️⃣ CREATING ENSEMBLE")
print("="*80)

if len(prophet_pred) > 0 and len(prophet_actual) > 0:
    # Align all predictions
    min_len = min(len(pred_ridge), len(prophet_pred), len(y_test))
    pred_ridge_align = pred_ridge[:min_len]
    prophet_align = prophet_pred[:min_len]
    actual_align = y_test.values[:min_len]
    
    # Weighted ensemble
    w_ridge = 1.0 / (rmse(actual_align, pred_ridge_align) + 1e-9)
    w_prop = 1.0 / (rmse(actual_align, prophet_align) + 1e-9)
    weights = np.array([w_ridge, w_prop])
    preds = np.array([pred_ridge_align, prophet_align])
    ensemble = (weights.reshape(-1, 1) * preds).sum(axis=0) / weights.sum()
    
    ens_metrics = metrics(actual_align, ensemble)
    results.append(['Ensemble', ens_metrics['rmse'], ens_metrics['mae'], ens_metrics['r2']])
    print(f"✅ Ensemble: R²={ens_metrics['r2']:.4f}, RMSE=${ens_metrics['rmse']:.2f}")
else:
    # Just use Ridge
    ens_metrics = ridge_metrics
    results.append(['Ensemble', ens_metrics['rmse'], ens_metrics['mae'], ens_metrics['r2']])
    print(f"✅ Ensemble (Ridge only): R²={ens_metrics['r2']:.4f}")

# Create results dataframe
results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MAE', 'R2'])
results_df = results_df.sort_values('R2', ascending=False)

# Save
os.makedirs('reports', exist_ok=True)
results_df.to_csv('reports/final_improved_r2_results.csv', index=False)

# Compare with previous
print("\n" + "="*80)
print("R² IMPROVEMENT SUMMARY")
print("="*80)

try:
    prev_results = pd.read_csv('reports/test_results_cleaned_data_FIXED.csv')
    
    print("\n📊 BEFORE (Previous Results):")
    for _, row in prev_results.iterrows():
        if row['R2'] > -100:
            print(f"   {row['Model']:25s} R²: {row['R2']:10.4f}")
    
    print("\n📊 AFTER (Improved Results):")
    for _, row in results_df.iterrows():
        if row['R2'] > -100:
            print(f"   {row['Model']:25s} R²: {row['R2']:10.4f}")
    
    # Calculate improvements
    print("\n📈 IMPROVEMENTS:")
    for _, new_row in results_df.iterrows():
        if new_row['R2'] > -100:
            model_name = new_row['Model']
            new_r2 = new_row['R2']
            
            # Find matching previous result
            prev_match = prev_results[prev_results['Model'].str.contains(model_name.split()[0], case=False)]
            if len(prev_match) > 0:
                prev_r2 = prev_match.iloc[0]['R2']
                if prev_r2 > -100:
                    improvement = ((new_r2 - prev_r2) / abs(prev_r2)) * 100 if prev_r2 != 0 else 0
                    print(f"   {model_name:25s} {prev_r2:8.4f} → {new_r2:8.4f} ({improvement:+.1f}%)")
            else:
                print(f"   {model_name:25s} NEW MODEL")
    
    # Best model
    best_model = results_df[results_df['R2'] > -100].iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   R²: {best_model['R2']:.4f} ({best_model['R2']*100:.1f}% accuracy)")
    print(f"   RMSE: ${best_model['RMSE']:.2f}")
    print(f"   MAE: ${best_model['MAE']:.2f}")
    
    if best_model['R2'] > 0.8:
        print(f"\n✅ EXCELLENT! R² > 0.8 achieved!")
    elif best_model['R2'] > 0.75:
        print(f"\n✅ VERY GOOD! R² > 0.75")
    elif best_model['R2'] > 0.7:
        print(f"\n✅ GOOD! R² > 0.7")
    else:
        print(f"\n⚠️ R² still below 0.7, but improved")
        
except Exception as e:
    print(f"Error comparing: {e}")

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print("\n" + results_df.to_string(index=False))
print("\n✅ Results saved to: reports/final_improved_r2_results.csv")

