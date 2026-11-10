#!/usr/bin/env python3
"""
Metal Price Prediction Model Training Script

This script trains multiple ML models for predicting Gold, Silver, and Oil prices.
It includes various models suitable for different situations and provides
comprehensive evaluation and comparison.

Models included:
1. LSTM - Best for capturing long-term dependencies and patterns
2. XGBoost - Best for tabular data with many features
3. Random Forest - Best for feature importance and robustness
4. Linear Regression - Simple baseline model
5. ARIMA - Best for univariate time series with trends
6. Prophet - Best for handling seasonality and holidays

Author: Senior Data Engineer
Date: 2024
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
import pickle
import json
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetalPricePredictor:
    """
    Comprehensive metal price prediction system with multiple models.
    """
    
    def __init__(self, data_file: str = 'global_commodity_economy.csv'):
        """
        Initialize the predictor.
        
        Args:
            data_file: Path to the CSV file with historical data
        """
        self.data_file = data_file
        self.data = None
        self.models = {}
        self.scalers = {}
        self.model_results = {}
        self.commodities = ['Gold_Futures', 'Silver_Futures', 'Crude_Oil_Futures']
        self.ticker_map = {
            'Gold_Futures': 'GC=F',
            'Silver_Futures': 'SI=F',
            'Crude_Oil_Futures': 'CL=F'
        }
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the dataset."""
        logger.info(f"Loading data from {self.data_file}...")
        self.data = pd.read_csv(self.data_file)
        
        # Convert Date to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Sort by date
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Remove rows with missing Close prices
        self.data = self.data.dropna(subset=['Close'])
        
        logger.info(f"Loaded {len(self.data)} records from {self.data['Date'].min()} to {self.data['Date'].max()}")
        return self.data
    
    def prepare_features(self, commodity: str, lookback_days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training.
        
        Args:
            commodity: Name of the commodity (Gold_Futures, Silver_Futures, Crude_Oil_Futures)
            lookback_days: Number of days to look back for features
            
        Returns:
            X, y arrays for training
        """
        logger.info(f"Preparing features for {commodity}...")
        
        # Filter data for this commodity
        commodity_data = self.data[self.data['Commodity'] == commodity].copy()
        
        if commodity_data.empty:
            logger.warning(f"No data found for {commodity}")
            return None, None
        
        # Select features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Oil_Price', 'USD_Index', 'CPI', 'Trade_Balance_US',
            'US_Tariff_Index', 'Sanction_Intensity', 'Russia_Sanction_Intensity',
            'US_Sanction_Index'
        ]
        
        # Only use columns that exist
        available_features = [col for col in feature_columns if col in commodity_data.columns]
        
        # Create feature matrix
        features = commodity_data[available_features].values
        
        # Create target (next day's Close price)
        target = commodity_data['Close'].shift(-1).values[:-1]
        features = features[:-1]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        features = features[valid_mask]
        target = target[valid_mask]
        
        # Create time series features (lookback)
        if len(features) > lookback_days:
            X_sequences = []
            y_sequences = []
            
            for i in range(lookback_days, len(features)):
                X_sequences.append(features[i-lookback_days:i].flatten())
                y_sequences.append(target[i])
            
            X = np.array(X_sequences)
            y = np.array(y_sequences)
        else:
            X = features
            y = target
        
        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features for {commodity}")
        return X, y
    
    def train_lstm(self, X: np.ndarray, y: np.ndarray, commodity: str) -> Dict:
        """
        Train LSTM model (simplified version using sequential features).
        Note: Full LSTM requires tensorflow/keras. This is a simplified version.
        
        Args:
            X: Feature matrix
            y: Target values
            commodity: Commodity name
            
        Returns:
            Model results dictionary
        """
        logger.info(f"Training LSTM model for {commodity}...")
        
        try:
            # For now, use a simpler approach since we don't have tensorflow
            # In production, you'd use Keras LSTM layers
            logger.warning("Full LSTM requires TensorFlow/Keras. Using XGBoost with time features instead.")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Scale features
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            
            # Use XGBoost as LSTM proxy (better for tabular time series)
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train_scaled)
            
            # Predictions
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results = {
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            logger.info(f"LSTM (XGBoost proxy) - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training LSTM for {commodity}: {str(e)}")
            return None
    
    def train_xgboost(self, X: np.ndarray, y: np.ndarray, commodity: str) -> Dict:
        """
        Train XGBoost model.
        
        Best for: Tabular data with many features, non-linear relationships
        
        Args:
            X: Feature matrix
            y: Target values
            commodity: Commodity name
            
        Returns:
            Model results dictionary
        """
        logger.info(f"Training XGBoost model for {commodity}...")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Scale features
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results = {
                'model': model,
                'scaler_X': scaler_X,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred,
                'feature_importance': dict(zip(range(len(X[0])), model.feature_importances_))
            }
            
            logger.info(f"XGBoost - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training XGBoost for {commodity}: {str(e)}")
            return None
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, commodity: str) -> Dict:
        """
        Train Random Forest model.
        
        Best for: Feature importance analysis, robustness to outliers
        
        Args:
            X: Feature matrix
            y: Target values
            commodity: Commodity name
            
        Returns:
            Model results dictionary
        """
        logger.info(f"Training Random Forest model for {commodity}...")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred,
                'feature_importance': dict(zip(range(len(X[0])), model.feature_importances_))
            }
            
            logger.info(f"Random Forest - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training Random Forest for {commodity}: {str(e)}")
            return None
    
    def train_linear_regression(self, X: np.ndarray, y: np.ndarray, commodity: str) -> Dict:
        """
        Train Linear Regression model.
        
        Best for: Baseline model, interpretability, linear relationships
        
        Args:
            X: Feature matrix
            y: Target values
            commodity: Commodity name
            
        Returns:
            Model results dictionary
        """
        logger.info(f"Training Linear Regression model for {commodity}...")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Scale features
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results = {
                'model': model,
                'scaler_X': scaler_X,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred,
                'coefficients': model.coef_
            }
            
            logger.info(f"Linear Regression - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training Linear Regression for {commodity}: {str(e)}")
            return None
    
    def train_arima(self, commodity: str) -> Dict:
        """
        Train ARIMA model.
        
        Best for: Univariate time series, trend and seasonality
        
        Args:
            commodity: Commodity name
            
        Returns:
            Model results dictionary
        """
        logger.info(f"Training ARIMA model for {commodity}...")
        
        try:
            # Get commodity data
            commodity_data = self.data[self.data['Commodity'] == commodity].copy()
            
            if commodity_data.empty:
                logger.warning(f"No data found for {commodity}")
                return None
            
            # Use Close price as time series
            ts = commodity_data['Close'].dropna().values
            
            if len(ts) < 100:
                logger.warning(f"Not enough data for ARIMA: {len(ts)} points")
                return None
            
            # Split data (80/20)
            split_idx = int(len(ts) * 0.8)
            train_ts = ts[:split_idx]
            test_ts = ts[split_idx:]
            
            # Fit ARIMA model (auto-select order)
            # Using (5,1,0) as default - can be optimized
            model = ARIMA(train_ts, order=(5, 1, 0))
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(test_ts))
            
            # Metrics
            mse = mean_squared_error(test_ts, forecast)
            mae = mean_absolute_error(test_ts, forecast)
            rmse = np.sqrt(mse)
            r2 = r2_score(test_ts, forecast)
            
            results = {
                'model': fitted_model,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'y_test': test_ts,
                'y_pred': forecast
            }
            
            logger.info(f"ARIMA - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training ARIMA for {commodity}: {str(e)}")
            return None
    
    def train_prophet(self, commodity: str) -> Dict:
        """
        Train Prophet model.
        
        Best for: Seasonality, holidays, missing data, trend changes
        
        Args:
            commodity: Commodity name
            
        Returns:
            Model results dictionary
        """
        logger.info(f"Training Prophet model for {commodity}...")
        
        try:
            # Get commodity data
            commodity_data = self.data[self.data['Commodity'] == commodity].copy()
            
            if commodity_data.empty:
                logger.warning(f"No data found for {commodity}")
                return None
            
            # Prepare data for Prophet (needs 'ds' and 'y' columns)
            prophet_data = commodity_data[['Date', 'Close']].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data = prophet_data.dropna()
            
            if len(prophet_data) < 100:
                logger.warning(f"Not enough data for Prophet: {len(prophet_data)} points")
                return None
            
            # Split data (80/20)
            split_idx = int(len(prophet_data) * 0.8)
            train_data = prophet_data[:split_idx]
            test_data = prophet_data[split_idx:]
            
            # Train model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(train_data)
            
            # Make future dataframe for test period
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)
            
            # Get predictions for test period
            test_forecast = forecast.tail(len(test_data))['yhat'].values
            y_test = test_data['y'].values
            
            # Metrics
            mse = mean_squared_error(y_test, test_forecast)
            mae = mean_absolute_error(y_test, test_forecast)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, test_forecast)
            
            results = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'y_test': y_test,
                'y_pred': test_forecast
            }
            
            logger.info(f"Prophet - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training Prophet for {commodity}: {str(e)}")
            return None
    
    def train_all_models(self, commodity: str):
        """
        Train all models for a specific commodity.
        
        Args:
            commodity: Commodity name
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training models for {commodity}")
        logger.info(f"{'='*60}")
        
        # Prepare features for ML models
        X, y = self.prepare_features(commodity)
        
        if X is None or y is None:
            logger.error(f"Cannot train models for {commodity} - no data")
            return
        
        results = {}
        
        # Train ML models (need features)
        if len(X) > 0:
            results['XGBoost'] = self.train_xgboost(X, y, commodity)
            results['RandomForest'] = self.train_random_forest(X, y, commodity)
            results['LinearRegression'] = self.train_linear_regression(X, y, commodity)
            results['LSTM_Proxy'] = self.train_lstm(X, y, commodity)
        
        # Train time series models (univariate)
        results['ARIMA'] = self.train_arima(commodity)
        results['Prophet'] = self.train_prophet(commodity)
        
        # Store results
        self.model_results[commodity] = results
        self.models[commodity] = results
    
    def compare_models(self, commodity: str) -> pd.DataFrame:
        """
        Compare all models for a commodity.
        
        Args:
            commodity: Commodity name
            
        Returns:
            DataFrame with model comparison
        """
        if commodity not in self.model_results:
            logger.error(f"No models trained for {commodity}")
            return None
        
        results = self.model_results[commodity]
        comparison = []
        
        for model_name, model_result in results.items():
            if model_result is not None:
                comparison.append({
                    'Model': model_name,
                    'RMSE': model_result['rmse'],
                    'MAE': model_result['mae'],
                    'R²': model_result['r2']
                })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('RMSE')
        
        return df
    
    def save_models(self, output_dir: str = 'models'):
        """Save all trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        for commodity, models in self.models.items():
            commodity_dir = os.path.join(output_dir, commodity.replace(' ', '_'))
            os.makedirs(commodity_dir, exist_ok=True)
            
            for model_name, model_result in models.items():
                if model_result is not None and 'model' in model_result:
                    # Save model
                    model_path = os.path.join(commodity_dir, f'{model_name}.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_result, f)
                    
                    # Save metrics
                    metrics = {
                        'rmse': model_result['rmse'],
                        'mae': model_result['mae'],
                        'r2': model_result['r2']
                    }
                    metrics_path = os.path.join(commodity_dir, f'{model_name}_metrics.json')
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
        
        logger.info(f"Models saved to {output_dir}")
    
    def plot_predictions(self, commodity: str, output_dir: str = 'model_plots'):
        """Plot predictions for all models."""
        os.makedirs(output_dir, exist_ok=True)
        
        if commodity not in self.model_results:
            return
        
        results = self.model_results[commodity]
        n_models = len([r for r in results.values() if r is not None])
        
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(n_models, 1, figsize=(15, 4*n_models))
        if n_models == 1:
            axes = [axes]
        
        plot_idx = 0
        for model_name, model_result in results.items():
            if model_result is not None:
                ax = axes[plot_idx]
                y_test = model_result['y_test']
                y_pred = model_result['y_pred']
                
                # Plot
                ax.plot(y_test[:100], label='Actual', alpha=0.7)
                ax.plot(y_pred[:100], label='Predicted', alpha=0.7)
                ax.set_title(f'{model_name} - {commodity}\nRMSE: {model_result["rmse"]:.2f}, R²: {model_result["r2"]:.4f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{commodity.replace(" ", "_")}_predictions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {plot_path}")


def main():
    """Main function to train all models."""
    logger.info("Starting Metal Price Prediction Model Training...")
    
    # Initialize predictor
    predictor = MetalPricePredictor()
    
    # Load data
    predictor.load_data()
    
    # Train models for each commodity
    for commodity in predictor.commodities:
        predictor.train_all_models(commodity)
        
        # Compare models
        comparison = predictor.compare_models(commodity)
        if comparison is not None:
            print(f"\n{'='*60}")
            print(f"Model Comparison for {commodity}")
            print(f"{'='*60}")
            print(comparison.to_string(index=False))
        
        # Plot predictions
        predictor.plot_predictions(commodity)
    
    # Save models
    predictor.save_models()
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print("Models saved to 'models/' directory")
    print("Plots saved to 'model_plots/' directory")
    print("\nBest model selection guide:")
    print("- XGBoost: Best for tabular data with many features")
    print("- Random Forest: Best for feature importance and robustness")
    print("- ARIMA: Best for univariate time series with trends")
    print("- Prophet: Best for seasonality and holiday effects")
    print("- Linear Regression: Simple baseline model")


if __name__ == "__main__":
    main()

