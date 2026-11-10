# Quick Start Guide: Metal Price Prediction

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- scikit-learn (for ML models)
- xgboost (gradient boosting)
- prophet (Facebook's time series model)
- statsmodels (for ARIMA)

### Step 2: Train Models

Train all models for Gold, Silver, and Oil:

```bash
python train_models.py
```

This will:
- ✅ Load your 15 years of historical data
- ✅ Train 6 different models for each metal
- ✅ Compare model performance
- ✅ Save trained models to `models/` directory
- ✅ Generate prediction plots to `model_plots/` directory

**Expected time:** 5-15 minutes (depending on your data size)

### Step 3: Make Predictions

Predict prices using trained models:

```bash
# Predict all metals for tomorrow
python predict_prices.py --commodity all --days 1

# Predict Gold price 7 days ahead
python predict_prices.py --commodity Gold_Futures --days 7

# Predict Silver price 30 days ahead
python predict_prices.py --commodity Silver_Futures --days 30
```

## 📊 What Models Are Trained?

For each metal (Gold, Silver, Oil), the following models are trained:

1. **XGBoost** - Best overall performance
2. **Random Forest** - Feature importance
3. **Linear Regression** - Baseline
4. **ARIMA** - Time series trends
5. **Prophet** - Seasonality and holidays
6. **LSTM Proxy** - Deep learning approach

## 📈 Understanding Results

After training, you'll see:

### Model Comparison Table
```
Model              RMSE      MAE       R²
XGBoost           12.45     8.32      0.95
Random Forest     13.21     9.01      0.94
Prophet           15.67     11.23     0.91
ARIMA             18.34     13.45     0.88
Linear Regression 22.10     16.78     0.82
```

**Lower RMSE/MAE = Better**  
**Higher R² = Better** (closer to 1.0 is best)

### Prediction Output
```
Price Predictions for Gold_Futures
============================================================
Current Price: $2,045.50

Predictions for 1 day(s) ahead:
------------------------------------------------------------
XGBoost          : $  2,048.23 (+0.13%)
Random Forest    : $  2,047.89 (+0.12%)
Prophet          : $  2,046.12 (+0.03%)
Average          : $  2,047.41 (+0.09%)
```

## 🎯 Which Model to Use?

**For Best Accuracy:** Use XGBoost (usually best)

**For Feature Importance:** Use Random Forest

**For Long-term Forecasts:** Use Prophet

**For Simple Trends:** Use ARIMA

See `MODEL_SELECTION_GUIDE.md` for detailed explanations.

## 📁 Output Files

After training, you'll have:

```
models/
├── Gold_Futures/
│   ├── XGBoost.pkl
│   ├── RandomForest.pkl
│   ├── LinearRegression.pkl
│   ├── ARIMA.pkl
│   ├── Prophet.pkl
│   └── ..._metrics.json files
├── Silver_Futures/
│   └── ...
└── Crude_Oil_Futures/
    └── ...

model_plots/
├── Gold_Futures_predictions.png
├── Silver_Futures_predictions.png
└── Crude_Oil_Futures_predictions.png
```

## 🔧 Troubleshooting

### "No module named 'xgboost'"
```bash
pip install xgboost
```

### "No module named 'prophet'"
```bash
pip install prophet
```

### "No data found for [commodity]"
- Make sure you've run `data_pipeline.py` first to collect data
- Check that `global_commodity_economy.csv` exists

### "Model not found" when predicting
- Make sure you've run `train_models.py` first
- Check that `models/` directory exists with trained models

## 💡 Tips

1. **More data = Better models**: With 15 years of data, models perform much better
2. **Compare models**: Don't just use one - compare multiple models
3. **Check feature importance**: Understand what drives prices
4. **Update regularly**: Retrain models as new data comes in
5. **Use ensemble**: Average predictions from multiple models for better accuracy

## 📚 Next Steps

1. Read `MODEL_SELECTION_GUIDE.md` for detailed model explanations
2. Experiment with different models
3. Analyze feature importance to understand price drivers
4. Set up automated retraining schedule
5. Create a web API for predictions (optional)

## 🎓 Example Workflow

```bash
# 1. Collect/update data (if needed)
python data_pipeline.py

# 2. Train models
python train_models.py

# 3. Make predictions
python predict_prices.py --commodity all --days 7

# 4. Check results
# - View model_plots/ for visualizations
# - Check models/ for saved models
# - Review console output for metrics
```

Happy predicting! 🚀

