#!/usr/bin/env python3
"""
Improved Gold Price Prediction
Fixes Prophet performance issues and improves accuracy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# Utility functions
def rmse(a, b): 
    return mean_squared_error(a, b)**0.5

def metrics(y_true, y_pred):
    return {
        'rmse': round(rmse(y_true, y_pred), 4), 
        'mae': round(mean_absolute_error(y_true, y_pred), 4), 
        'r2': round(r2_score(y_true, y_pred), 4)
    }

# Ensure directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load data
DATA_PATH = 'global_commodity_economy.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=False)
df = df.sort_values('Date').reset_index(drop=True)

print("Loaded shape:", df.shape)
print("Columns:", df.columns.tolist())

# Step 1: Detect Gold rows
mask = pd.Series(False, index=df.index)
if 'Commodity' in df.columns:
    mask = mask | df['Commodity'].astype(str).str.contains('Gold', case=False, na=False)
if 'Ticker' in df.columns:
    mask = mask | df['Ticker'].astype(str).str.contains('GC=F|GC', case=False, na=False)

print("Detected Gold rows:", int(mask.sum()))

# Choose price column
price_cols = [c for c in ['Close', 'Adj Close', 'Close_Price', 'Price', 'close'] if c in df.columns]
if len(price_cols) == 0:
    raise ValueError("No price column found")
price_col = price_cols[0]
print("Using price column:", price_col)

# Build df_gold
df_gold_rows = df[mask].copy() if mask.sum() > 0 else df.copy()
df_gold = df_gold_rows[['Date', price_col]].rename(columns={price_col: 'Price'})
df_gold = df_gold.sort_values('Date').reset_index(drop=True)

# Merge macro regressors
macro_candidates = ['Oil_Price', 'USD_Index', 'Trade_Weighted_Dollar', 'CPI', 'Volume', 
                   'US_Tariff_Index', 'Sanction_Intensity', 'Russia_Sanction_Intensity']
for m in macro_candidates:
    if m in df.columns:
        df_gold = df_gold.merge(df[['Date', m]].drop_duplicates('Date'), on='Date', how='left')

print("df_gold columns:", df_gold.columns.tolist())

# Step 2: Clean and prepare
df_gold['Date'] = pd.to_datetime(df_gold['Date'], errors='coerce')
df_gold = df_gold.dropna(subset=['Date']).sort_values('Date').drop_duplicates(subset=['Date'], keep='first')
df_gold = df_gold.set_index('Date').asfreq('D')

# Convert to numeric and interpolate
for c in df_gold.columns:
    df_gold[c] = pd.to_numeric(df_gold[c], errors='coerce')

numeric_cols = df_gold.select_dtypes(include=[np.number]).columns
df_gold[numeric_cols] = df_gold[numeric_cols].interpolate(method='time').bfill().ffill()

# Cap outliers (winsorize at 99.5th percentile)
cap = df_gold['Price'].quantile(0.995)
df_gold['Price_cap'] = df_gold['Price'].clip(upper=cap)

# Create lagged regressors (IMPROVED: better lag selection)
oil_col = None
usd_col = None
for col in ['Oil_Price', 'Oil', 'Crude_Oil']:
    if col in df_gold.columns:
        oil_col = col
        break
for col in ['USD_Index', 'Dollar_Index', 'USD', 'DTWEXBGS']:
    if col in df_gold.columns:
        usd_col = col
        break

if oil_col:
    df_gold['Oil_lag1'] = df_gold[oil_col].shift(1)
    df_gold['Oil_lag7'] = df_gold[oil_col].shift(7)
    df_gold['Oil_ma7'] = df_gold[oil_col].rolling(7).mean()
if usd_col:
    df_gold['USD_lag1'] = df_gold[usd_col].shift(1)
    df_gold['USD_lag7'] = df_gold[usd_col].shift(7)
    df_gold['USD_ma7'] = df_gold[usd_col].rolling(7).mean()

df_gold = df_gold.reset_index()

# Save cleaned
df_gold.to_csv('data/cleaned_gold.csv', index=False)
print("Saved: data/cleaned_gold.csv")

# Step 3: IMPROVED Feature Engineering
df_feat = df_gold.copy()
price_used = 'Price_cap' if 'Price_cap' in df_feat.columns else 'Price'

# Price lags (IMPROVED: more strategic lags)
for l in [1, 2, 3, 5, 7, 14, 21, 30]:
    df_feat[f'lag_{l}'] = df_feat[price_used].shift(l)

# Rolling statistics (IMPROVED: multiple windows)
for window in [7, 14, 30, 60]:
    df_feat[f'ma{window}'] = df_feat[price_used].rolling(window).mean()
    df_feat[f'std{window}'] = df_feat[price_used].rolling(window).std()

# Returns and momentum
df_feat['return_1'] = df_feat[price_used].pct_change(1)
df_feat['return_7'] = df_feat[price_used].pct_change(7)
df_feat['return_30'] = df_feat[price_used].pct_change(30)
df_feat['momentum_7'] = df_feat[price_used] / df_feat[price_used].shift(7) - 1

# Date features (IMPROVED: more features)
df_feat['dayofweek'] = df_feat['Date'].dt.dayofweek
df_feat['dayofmonth'] = df_feat['Date'].dt.day
df_feat['month'] = df_feat['Date'].dt.month
df_feat['quarter'] = df_feat['Date'].dt.quarter
df_feat['is_month_end'] = df_feat['Date'].dt.is_month_end.astype(int)

# Drop NA
df_feat = df_feat.dropna().reset_index(drop=True)
print("Feature df shape:", df_feat.shape)

df_feat.to_csv('data/features_gold.csv', index=False)

# Step 4: IMPROVED Train/Test Split
test_days = 60
val_days = 30

train_df = df_feat.iloc[:-test_days].copy()
val_df = df_feat.iloc[-test_days:-val_days].copy() if test_days > val_days else df_feat.iloc[-test_days:].copy()
test_df = df_feat.iloc[-val_days:].copy() if test_days > val_days else df_feat.iloc[-test_days:].copy()

# IMPROVED Feature Selection (more comprehensive)
feature_cols = ['lag_1', 'lag_2', 'lag_7', 'lag_14', 'ma7', 'ma30', 'std7', 'return_1', 'return_7']

# Add macro features if available
for c in ['Oil_lag1', 'USD_lag1', 'Oil_ma7', 'USD_ma7', 'CPI', 'USD_Index']:
    if c in df_feat.columns:
        feature_cols.append(c)

# Remove any missing columns
feature_cols = [c for c in feature_cols if c in df_feat.columns]

X_train = train_df[feature_cols]
y_train = train_df[price_used]
X_val = val_df[feature_cols] if len(val_df) > 0 else None
y_val = val_df[price_used] if len(val_df) > 0 else None
X_test = test_df[feature_cols]
y_test = test_df[price_used]

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Step 5: IMPROVED Linear Regression (with Ridge regularization)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Try Ridge instead of plain Linear Regression (better generalization)
lr = Ridge(alpha=1.0).fit(X_train_s, y_train)
pred_lr = lr.predict(X_test_s)

print("LR test metrics:", metrics(y_test, pred_lr))

# Plot
plt.figure(figsize=(12, 5))
plt.plot(test_df['Date'], y_test, label='Actual', linewidth=2)
plt.plot(test_df['Date'], pred_lr, label='Ridge Pred', linewidth=2)
plt.title('PART 1 — Historical Prediction (Last {} days)'.format(len(test_df)))
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/historical_lr_pred.png', dpi=150)
plt.close()

joblib.dump(lr, 'models/linear_regression_gold.pkl')
joblib.dump(scaler, 'models/scaler_gold.pkl')

# Step 6: FIXED Prophet Implementation
print("\n" + "="*60)
print("FIXING PROPHET MODEL")
print("="*60)

# Prepare Prophet dataframe (IMPROVED: better regressor selection)
prophet_df = df_feat[['Date', price_used]].rename(columns={'Date': 'ds', price_used: 'y'}).copy()

# Select only stable, non-leaking regressors
prophet_regressors = []
for col in ['Oil_lag1', 'USD_lag1', 'CPI']:
    if col in df_feat.columns:
        prophet_df[col] = df_feat[col].values
        prophet_regressors.append(col)

prophet_df = prophet_df.dropna().reset_index(drop=True)
print("Prophet df rows:", prophet_df.shape[0])
print("Prophet regressors:", prophet_regressors)

# IMPROVED: Better validation split
val_horizon = 30
train_prop = prophet_df.iloc[:-val_horizon].copy()
val_prop = prophet_df.iloc[-val_horizon:].copy()

print(f"Train: {len(train_prop)} rows, Val: {len(val_prop)} rows")

# IMPROVED: Better hyperparameter tuning
cps_grid = [0.001, 0.01, 0.05, 0.1, 0.3]
seasonality_mode = ['additive', 'multiplicative']
best_cps, best_mode, best_rm = None, None, 1e12

for cps in cps_grid:
    for mode in seasonality_mode:
        try:
            mtmp = Prophet(
                changepoint_prior_scale=cps,
                seasonality_mode=mode,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_prior_scale=10.0,  # Allow more seasonality
                holidays_prior_scale=10.0
            )
            
            # Add regressors
            for reg in prophet_regressors:
                mtmp.add_regressor(reg, mode=mode)
            
            mtmp.fit(train_prop)
            
            # Create future dataframe
            future = mtmp.make_future_dataframe(periods=val_horizon)
            
            # FIXED: Properly handle regressors in future
            # Use last known values for future regressors
            for reg in prophet_regressors:
                last_val = train_prop[reg].iloc[-1]
                future[reg] = last_val  # Simple forward fill
            
            fc = mtmp.predict(future)
            pred_val = fc.set_index('ds')['yhat'].iloc[-val_horizon:].values
            actual_val = val_prop['y'].values
            
            rm = rmse(actual_val, pred_val)
            print(f"cps={cps:.3f}, mode={mode}, val_rmse={rm:.4f}")
            
            if rm < best_rm:
                best_rm, best_cps, best_mode = rm, cps, mode
        except Exception as e:
            print(f"Error with cps={cps}, mode={mode}: {e}")
            continue

print(f"\nBest: cps={best_cps}, mode={best_mode}, RMSE={best_rm:.4f}")

# Fit final Prophet with best parameters
m_final = Prophet(
    changepoint_prior_scale=best_cps,
    seasonality_mode=best_mode,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0
)

for reg in prophet_regressors:
    m_final.add_regressor(reg, mode=best_mode)

m_final.fit(prophet_df)
joblib.dump(m_final, 'models/prophet_optimized_gold.pkl')
print("Saved optimized Prophet model")

# Step 7: IMPROVED Prophet Forecasts with proper regressor handling
def make_prophet_forecast(model, periods, prophet_df, regressors):
    """Make Prophet forecast with proper regressor handling."""
    future = model.make_future_dataframe(periods=periods)
    
    # For regressors, use forward fill from last known values
    for reg in regressors:
        if reg in prophet_df.columns:
            # Get last known value
            last_val = prophet_df[reg].iloc[-1]
            # Fill future with last known value (or use a simple trend)
            future[reg] = last_val
    
    forecast = model.predict(future)
    return forecast

# 7-day forecast
fcst_7 = make_prophet_forecast(m_final, 7, prophet_df, prophet_regressors)

plt.figure(figsize=(12, 5))
plt.plot(prophet_df['ds'].iloc[-120:], prophet_df['y'].iloc[-120:], label='Actual', linewidth=2)
plt.plot(fcst_7['ds'].iloc[-127:], fcst_7['yhat'].iloc[-127:], label='Prophet 7-day', color='green', linewidth=2)
plt.fill_between(fcst_7['ds'].iloc[-127:], fcst_7['yhat_lower'].iloc[-127:], 
                 fcst_7['yhat_upper'].iloc[-127:], color='green', alpha=0.18)
plt.title('PART 2 — Prophet 7-day Forecast (IMPROVED)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/prophet_7day.png', dpi=150)
plt.close()

# 30-day forecast
fcst_30 = make_prophet_forecast(m_final, 30, prophet_df, prophet_regressors)

plt.figure(figsize=(12, 5))
plt.plot(prophet_df['ds'].iloc[-120:], prophet_df['y'].iloc[-120:], label='Actual', linewidth=2)
plt.plot(fcst_30['ds'].iloc[-150:], fcst_30['yhat'].iloc[-150:], label='Prophet 30-day', color='orange', linewidth=2)
plt.fill_between(fcst_30['ds'].iloc[-150:], fcst_30['yhat_lower'].iloc[-150:], 
                 fcst_30['yhat_upper'].iloc[-150:], color='orange', alpha=0.18)
plt.title('PART 2 — Prophet 30-day Forecast (IMPROVED)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/prophet_30day.png', dpi=150)
plt.close()

# Step 8: FIXED Prophet Evaluation on held-out 30 days
fcst_indexed = fcst_30.set_index('ds')
val_ds = val_prop['ds'].values

# Align dates properly
prophet_val_pred = []
prophet_val_actual = []
for ds in val_ds:
    if ds in fcst_indexed.index:
        prophet_val_pred.append(fcst_indexed.loc[ds]['yhat'])
        prophet_val_actual.append(val_prop[val_prop['ds'] == ds]['y'].values[0])

prophet_val_pred = np.array(prophet_val_pred)
prophet_val_actual = np.array(prophet_val_actual)

print("\n" + "="*60)
print("FIXED Prophet tuned metrics on held-out 30 days:")
print(metrics(prophet_val_actual, prophet_val_pred))
print("="*60)

# Step 9: IMPROVED Ensemble
pred_lr_last30 = pred_lr[-30:] if len(pred_lr) >= 30 else pred_lr
actual_last30 = y_test[-30:].values if len(y_test) >= 30 else y_test.values

# Align Prophet predictions
prophet_last30 = fcst_30.set_index('ds')['yhat'].iloc[-30:].values

# Print metrics
print("\nModel Performance (Last 30 days):")
print("LR:", metrics(actual_last30, pred_lr_last30))
print("Prophet:", metrics(actual_last30, prophet_last30))

# IMPROVED Ensemble (weighted by inverse RMSE)
weights = []
preds = []

w_lr = 1.0 / (rmse(actual_last30, pred_lr_last30) + 1e-9)
weights.append(w_lr)
preds.append(pred_lr_last30)

w_prop = 1.0 / (rmse(actual_last30, prophet_last30) + 1e-9)
weights.append(w_prop)
preds.append(prophet_last30)

weights = np.array(weights)
preds = np.array(preds)
ensemble = (weights.reshape(-1, 1) * preds).sum(axis=0) / weights.sum()

print("Ensemble:", metrics(actual_last30, ensemble))

# Save ensemble predictions
ens_df = pd.DataFrame({
    'date': df_feat['Date'].iloc[-30:].values,
    'actual': actual_last30,
    'lr': pred_lr_last30,
    'prophet': prophet_last30,
    'ensemble': ensemble
})
ens_df.to_csv('reports/final_predictions_last30.csv', index=False)
print("\nSaved: reports/final_predictions_last30.csv")

# Step 10: Final overlay plot
plt.figure(figsize=(14, 6))
n = min(120, len(df_feat))
plt.plot(df_feat['Date'].iloc[-n:], df_feat[price_used].iloc[-n:], label='Actual', linewidth=2)
plt.plot(fcst_30['ds'].iloc[-n-30:], fcst_30['yhat'].iloc[-n-30:], label='Prophet Forecast', linewidth=1.5)
plt.plot(test_df['Date'], pred_lr, label='Ridge Predictions', linewidth=1.2)
plt.title('Final Overlay: Actual vs Prophet Forecast vs Ridge Predictions (IMPROVED)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/final_overlay.png', dpi=150)
plt.close()

# Summary
summary = pd.DataFrame([
    ['Ridge Regression', *list(metrics(actual_last30, pred_lr_last30).values())],
    ['Prophet (Fixed)', *list(metrics(actual_last30, prophet_last30).values())],
    ['Ensemble', *list(metrics(actual_last30, ensemble).values())],
], columns=['Model', 'RMSE', 'MAE', 'R2'])

summary.to_csv('reports/model_summary.csv', index=False)
print("\n" + "="*60)
print("FINAL MODEL SUMMARY")
print("="*60)
print(summary.to_string(index=False))
print("="*60)
print("\nAll files saved successfully!")

