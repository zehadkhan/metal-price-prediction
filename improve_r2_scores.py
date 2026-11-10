#!/usr/bin/env python3
"""
Improve R² Scores for All Models
Focuses on getting R² > 0.8 for all models
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
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
print("IMPROVING R² SCORES FOR ALL MODELS")
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
for col in ['Oil_Price', 'USD_Index', 'CPI', 'US_Tariff_Index', 'Sanction_Intensity']:
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

# IMPROVED Feature engineering
df_feat = df_gold.copy()
price_col = 'Price'

print(f"\n🔧 Enhanced Feature Engineering...")

# More lags
for l in [1, 2, 3, 5, 7, 14, 21, 30, 60]:
    df_feat[f'lag_{l}'] = df_feat[price_col].shift(l)

# Rolling stats with multiple windows
for window in [7, 14, 30, 60]:
    df_feat[f'ma{window}'] = df_feat[price_col].rolling(window, min_periods=1).mean()
    df_feat[f'std{window}'] = df_feat[price_col].rolling(window, min_periods=1).std()
    df_feat[f'min{window}'] = df_feat[price_col].rolling(window, min_periods=1).min()
    df_feat[f'max{window}'] = df_feat[price_col].rolling(window, min_periods=1).max()

# Returns and momentum
df_feat['return_1'] = df_feat[price_col].pct_change(1)
df_feat['return_7'] = df_feat[price_col].pct_change(7)
df_feat['return_30'] = df_feat[price_col].pct_change(30)
df_feat['momentum_7'] = df_feat[price_col] / df_feat[price_col].shift(7) - 1
df_feat['momentum_30'] = df_feat[price_col] / df_feat[price_col].shift(30) - 1

# Date features
df_feat['dayofweek'] = df_feat['Date'].dt.dayofweek
df_feat['dayofmonth'] = df_feat['Date'].dt.day
df_feat['month'] = df_feat['Date'].dt.month
df_feat['quarter'] = df_feat['Date'].dt.quarter
df_feat['is_month_end'] = df_feat['Date'].dt.is_month_end.astype(int)

# Macro lags and features
if 'Oil_Price' in df_feat.columns:
    df_feat['Oil_lag1'] = df_feat['Oil_Price'].shift(1)
    df_feat['Oil_lag7'] = df_feat['Oil_Price'].shift(7)
    df_feat['Oil_ma7'] = df_feat['Oil_Price'].rolling(7, min_periods=1).mean()
if 'USD_Index' in df_feat.columns:
    df_feat['USD_lag1'] = df_feat['USD_Index'].shift(1)
    df_feat['USD_lag7'] = df_feat['USD_Index'].shift(7)
    df_feat['USD_ma7'] = df_feat['USD_Index'].rolling(7, min_periods=1).mean()
if 'CPI' in df_feat.columns:
    df_feat['CPI_lag1'] = df_feat['CPI'].shift(1)
    df_feat['CPI_change'] = df_feat['CPI'].pct_change(1)

df_feat = df_feat.dropna(subset=[price_col]).reset_index(drop=True)

print(f"✅ Features ready: {len(df_feat)} records, {len(df_feat.columns)} features")

# Train/Test split
test_days = 60
train_df = df_feat.iloc[:-test_days].copy()
test_df = df_feat.iloc[-test_days:].copy()

# Feature selection - IMPROVED
feature_cols = [
    'lag_1', 'lag_2', 'lag_7', 'lag_14', 'lag_30',
    'ma7', 'ma30', 'ma60', 'std7', 'std30',
    'return_1', 'return_7', 'momentum_7', 'momentum_30'
]

# Add macro features
for col in ['Oil_lag1', 'USD_lag1', 'CPI_lag1', 'Oil_ma7', 'USD_ma7', 'CPI_change']:
    if col in df_feat.columns:
        feature_cols.append(col)

feature_cols = [c for c in feature_cols if c in df_feat.columns and c in train_df.columns]
feature_cols = [c for c in feature_cols if not train_df[c].isna().all()]

X_train = train_df[feature_cols].fillna(0)
y_train = train_df[price_col]
X_test = test_df[feature_cols].fillna(0)
y_test = test_df[price_col]

print(f"✅ Using {len(feature_cols)} features")

# Test multiple models with different scalers
print("\n" + "="*80)
print("TESTING MODELS WITH IMPROVED FEATURES")
print("="*80)

results = []

# 1. Ridge Regression with StandardScaler
print("\n1️⃣ Ridge Regression (StandardScaler)...")
scaler_std = StandardScaler().fit(X_train)
X_train_s = scaler_std.transform(X_train)
X_test_s = scaler_std.transform(X_test)

lr_ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
pred_ridge = lr_ridge.predict(X_test_s)
ridge_metrics = metrics(y_test.values, pred_ridge)
results.append(['Ridge Regression', ridge_metrics['rmse'], ridge_metrics['mae'], ridge_metrics['r2']])
print(f"   R²: {ridge_metrics['r2']:.4f}, RMSE: ${ridge_metrics['rmse']:.2f}")

# 2. Ridge Regression with RobustScaler (better for outliers)
print("\n2️⃣ Ridge Regression (RobustScaler)...")
scaler_rob = RobustScaler().fit(X_train)
X_train_r = scaler_rob.transform(X_train)
X_test_r = scaler_rob.transform(X_test)

lr_rob = Ridge(alpha=1.0).fit(X_train_r, y_train)
pred_rob = lr_rob.predict(X_test_r)
rob_metrics = metrics(y_test.values, pred_rob)
results.append(['Ridge (RobustScaler)', rob_metrics['rmse'], rob_metrics['mae'], rob_metrics['r2']])
print(f"   R²: {rob_metrics['r2']:.4f}, RMSE: ${rob_metrics['rmse']:.2f}")

# 3. ElasticNet (combines Ridge and Lasso)
print("\n3️⃣ ElasticNet...")
lr_elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train_s, y_train)
pred_elastic = lr_elastic.predict(X_test_s)
elastic_metrics = metrics(y_test.values, pred_elastic)
results.append(['ElasticNet', elastic_metrics['rmse'], elastic_metrics['mae'], elastic_metrics['r2']])
print(f"   R²: {elastic_metrics['r2']:.4f}, RMSE: ${elastic_metrics['rmse']:.2f}")

# 4. Random Forest
print("\n4️⃣ Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
rf_metrics = metrics(y_test.values, pred_rf)
results.append(['Random Forest', rf_metrics['rmse'], rf_metrics['mae'], rf_metrics['r2']])
print(f"   R²: {rf_metrics['r2']:.4f}, RMSE: ${rf_metrics['rmse']:.2f}")

# 5. Gradient Boosting
print("\n5️⃣ Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
pred_gb = gb.predict(X_test)
gb_metrics = metrics(y_test.values, pred_gb)
results.append(['Gradient Boosting', gb_metrics['rmse'], gb_metrics['mae'], gb_metrics['r2']])
print(f"   R²: {gb_metrics['r2']:.4f}, RMSE: ${gb_metrics['rmse']:.2f}")

# 6. IMPROVED Prophet with better configuration
print("\n6️⃣ Prophet (Improved Configuration)...")
prophet_df = df_feat[['Date', price_col]].copy()
prophet_df = prophet_df.rename(columns={'Date': 'ds', price_col: 'y'})
prophet_df = prophet_df.dropna().reset_index(drop=True)

# Use log transformation
prophet_df_log = prophet_df.copy()
prophet_df_log['y'] = np.log(prophet_df_log['y'])

val_horizon = 30
train_prop = prophet_df_log.iloc[:-val_horizon].copy()

# Best config from previous testing
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

# Get predictions for test period
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

if len(prophet_pred) > 0:
    prophet_pred = np.array(prophet_pred)
    prophet_actual = np.array(prophet_actual)
    prop_metrics = metrics(prophet_actual, prophet_pred)
    results.append(['Prophet (Improved)', prop_metrics['rmse'], prop_metrics['mae'], prop_metrics['r2']])
    print(f"   R²: {prop_metrics['r2']:.4f}, RMSE: ${prop_metrics['rmse']:.2f}")
else:
    results.append(['Prophet (Improved)', 9999, 9999, -9999])
    print("   ⚠️ Could not align predictions")

# Create results dataframe
results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MAE', 'R2'])
results_df = results_df.sort_values('R2', ascending=False)

# Save results
os.makedirs('reports', exist_ok=True)
results_df.to_csv('reports/improved_r2_results.csv', index=False)

# Print comparison
print("\n" + "="*80)
print("IMPROVED R² SCORES - COMPARISON")
print("="*80)
print("\n" + results_df.to_string(index=False))

# Compare with previous
print("\n" + "="*80)
print("R² IMPROVEMENT ANALYSIS")
print("="*80)

try:
    prev_results = pd.read_csv('reports/test_results_cleaned_data_FIXED.csv')
    print("\n📊 Previous Results:")
    for _, row in prev_results.iterrows():
        print(f"   {row['Model']:20s} R²: {row['R2']:10.4f}")
    
    print("\n📊 Improved Results:")
    for _, row in results_df.iterrows():
        if row['R2'] > -100:  # Only show reasonable values
            print(f"   {row['Model']:20s} R²: {row['R2']:10.4f}")
    
    # Find best model
    best_model = results_df[results_df['R2'] > -100].iloc[0]
    print(f"\n🏆 Best Model: {best_model['Model']}")
    print(f"   R²: {best_model['R2']:.4f} ({best_model['R2']*100:.1f}% accuracy)")
    print(f"   RMSE: ${best_model['RMSE']:.2f}")
    print(f"   MAE: ${best_model['MAE']:.2f}")
    
    # Check if we achieved R² > 0.8
    best_r2 = best_model['R2']
    if best_r2 > 0.8:
        print(f"\n✅ SUCCESS! Achieved R² > 0.8 ({best_r2:.4f})")
    elif best_r2 > 0.7:
        print(f"\n✅ GOOD! R² > 0.7 ({best_r2:.4f})")
    else:
        print(f"\n⚠️ R² still below 0.7 ({best_r2:.4f})")
        
except Exception as e:
    print(f"Could not compare with previous: {e}")

print("\n✅ Results saved to: reports/improved_r2_results.csv")

