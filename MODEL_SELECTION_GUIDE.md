# Model Selection Guide for Metal Price Prediction

This guide explains which machine learning models are best suited for different situations when predicting metal prices (Gold, Silver, Oil).

## 📊 Available Models

### 1. **XGBoost** ⭐ (Recommended for Most Cases)
**Best For:**
- Tabular data with many features (economic indicators, tariffs, sanctions)
- Non-linear relationships between features and price
- Handling missing values and outliers
- Feature importance analysis
- When you have 15+ years of historical data

**When to Use:**
- ✅ You have multiple features (Oil_Price, USD_Index, CPI, Tariffs, Sanctions)
- ✅ You want to understand which features matter most
- ✅ You need high accuracy with complex relationships
- ✅ You have sufficient data (1000+ samples)

**Performance:**
- Usually achieves the best RMSE and R² scores
- Fast training and prediction
- Good generalization

**Example Use Case:**
Predicting Gold price using economic indicators, tariffs, and sanctions data.

---

### 2. **Random Forest** 🌲
**Best For:**
- Feature importance analysis
- Robustness to outliers
- Handling non-linear relationships
- When you want interpretability

**When to Use:**
- ✅ You want to know which features are most important
- ✅ Your data has outliers or noise
- ✅ You need a robust model that won't overfit easily
- ✅ You want feature importance rankings

**Performance:**
- Good accuracy, slightly lower than XGBoost
- Very robust to outliers
- Provides feature importance scores

**Example Use Case:**
Understanding which economic factors (USD Index, CPI, Sanctions) most impact Silver prices.

---

### 3. **ARIMA** 📈
**Best For:**
- Univariate time series (price only, no features)
- Capturing trends and seasonality
- When you only have historical price data
- Short-term forecasting (1-30 days)

**When to Use:**
- ✅ You only have price history (no economic indicators)
- ✅ You want to capture trends and patterns in price alone
- ✅ Short-term predictions (next day, next week)
- ✅ You need a simple, interpretable time series model

**Performance:**
- Good for trend-based predictions
- May miss complex feature relationships
- Best for univariate scenarios

**Example Use Case:**
Predicting tomorrow's Oil price based only on historical Oil prices.

---

### 4. **Prophet** 📅
**Best For:**
- Seasonality and holiday effects
- Missing data handling
- Trend changes over time
- Long-term forecasting (months/years)
- When you have irregular data

**When to Use:**
- ✅ You want to capture seasonal patterns (e.g., gold prices in December)
- ✅ Your data has missing dates
- ✅ You need long-term forecasts (3+ months)
- ✅ You want to account for holidays and events
- ✅ You need automatic trend change detection

**Performance:**
- Excellent for seasonal patterns
- Handles missing data well
- Good for long-term forecasts
- May be slower than other models

**Example Use Case:**
Forecasting Gold prices 6 months ahead, accounting for seasonal demand patterns.

---

### 5. **Linear Regression** 📉
**Best For:**
- Baseline model
- Interpretability
- Simple linear relationships
- When features have linear relationships with price

**When to Use:**
- ✅ You need a simple baseline to compare other models
- ✅ You want to understand linear relationships
- ✅ Your data shows clear linear patterns
- ✅ You need fast, interpretable predictions

**Performance:**
- Usually lower accuracy than tree-based models
- Very fast and interpretable
- Good baseline for comparison

**Example Use Case:**
Quick baseline prediction when you suspect linear relationships between CPI and metal prices.

---

### 6. **LSTM (Long Short-Term Memory)** 🧠
**Best For:**
- Long-term dependencies in time series
- Complex sequential patterns
- When you have very long time series (years of data)
- Deep learning approach

**When to Use:**
- ✅ You have 10+ years of daily data
- ✅ You want to capture complex long-term patterns
- ✅ You need deep learning capabilities
- ✅ You have computational resources

**Performance:**
- Can capture complex patterns
- Requires more data and computation
- May overfit with small datasets

**Note:** Full LSTM requires TensorFlow/Keras. The current implementation uses XGBoost as a proxy for time series features.

---

## 🎯 Model Selection Decision Tree

```
Do you have multiple features (economic indicators, tariffs, sanctions)?
│
├─ YES → Use XGBoost or Random Forest
│   │
│   ├─ Want feature importance? → Random Forest
│   └─ Want best accuracy? → XGBoost
│
└─ NO (only price history) → Use ARIMA or Prophet
    │
    ├─ Need seasonality/holidays? → Prophet
    └─ Simple trend? → ARIMA
```

## 📈 Expected Performance by Commodity

### Gold (GC=F)
- **Best Model:** XGBoost (uses economic indicators well)
- **Why:** Gold prices correlate strongly with USD Index, CPI, and sanctions
- **Alternative:** Random Forest (for feature importance)

### Silver (SI=F)
- **Best Model:** XGBoost or Random Forest
- **Why:** Similar to Gold, but more volatile - benefits from feature engineering
- **Alternative:** Prophet (if you want to capture seasonal patterns)

### Crude Oil (CL=F)
- **Best Model:** XGBoost or Prophet
- **Why:** Oil prices are affected by many factors (sanctions, tariffs, economic indicators)
- **Alternative:** ARIMA (if you only have price history)

## 🔧 Model Comparison Metrics

When comparing models, look at:

1. **RMSE (Root Mean Squared Error)**: Lower is better
   - Measures average prediction error
   - Penalizes large errors more

2. **MAE (Mean Absolute Error)**: Lower is better
   - Average absolute difference
   - Easier to interpret (e.g., "$5 off on average")

3. **R² (R-squared)**: Higher is better (0-1)
   - Proportion of variance explained
   - 0.9+ is excellent, 0.7+ is good

## 💡 Recommendations

### For Production Use:
1. **Start with XGBoost** - Usually best overall performance
2. **Use Random Forest** - If you need feature importance
3. **Try Prophet** - For long-term forecasts with seasonality

### For Research/Analysis:
1. **Compare all models** - See which works best for your data
2. **Use Linear Regression** - As a baseline
3. **Analyze feature importance** - Understand what drives prices

### For Different Time Horizons:
- **1-7 days:** XGBoost, ARIMA
- **1-4 weeks:** XGBoost, Random Forest
- **1-3 months:** Prophet, XGBoost
- **6+ months:** Prophet

## 🚀 Quick Start

```python
# Train all models
python train_models.py

# This will:
# 1. Train 6 models for each metal (Gold, Silver, Oil)
# 2. Compare performance
# 3. Save best models
# 4. Generate prediction plots
```

## 📝 Notes

- **More data = Better models**: With 15 years of data, all models perform better
- **Feature engineering matters**: Economic indicators, tariffs, and sanctions improve predictions
- **No single best model**: Different models excel in different situations
- **Ensemble approach**: Consider combining predictions from multiple models

