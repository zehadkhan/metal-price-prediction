#!/usr/bin/env python3
"""
Metal Price Prediction Script

This script uses trained models to predict future metal prices.
It loads saved models and makes predictions for Gold, Silver, and Oil.

Usage:
    python predict_prices.py --commodity Gold_Futures --days 30
    python predict_prices.py --commodity all --days 7
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PricePredictor:
    """Load trained models and make predictions."""
    
    def __init__(self, models_dir: str = 'models', data_file: str = 'global_commodity_economy.csv'):
        """
        Initialize predictor.
        
        Args:
            models_dir: Directory containing saved models
            data_file: Path to historical data CSV
        """
        self.models_dir = models_dir
        self.data_file = data_file
        self.data = None
        self.models = {}
        self.commodities = ['Gold_Futures', 'Silver_Futures', 'Crude_Oil_Futures']
        
    def load_data(self):
        """Load historical data."""
        logger.info(f"Loading data from {self.data_file}...")
        self.data = pd.read_csv(self.data_file)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        logger.info(f"Loaded {len(self.data)} records")
    
    def load_model(self, commodity: str, model_name: str) -> Optional[Dict]:
        """
        Load a trained model.
        
        Args:
            commodity: Commodity name
            model_name: Name of the model
            
        Returns:
            Model dictionary or None
        """
        commodity_dir = os.path.join(self.models_dir, commodity.replace(' ', '_'))
        model_path = os.path.join(commodity_dir, f'{model_name}.pkl')
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            logger.info(f"Loaded {model_name} for {commodity}")
            return model_data
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {str(e)}")
            return None
    
    def get_latest_features(self, commodity: str, lookback_days: int = 30) -> Optional[np.ndarray]:
        """
        Get latest features for prediction.
        
        Args:
            commodity: Commodity name
            lookback_days: Number of days to look back
            
        Returns:
            Feature array or None
        """
        commodity_data = self.data[self.data['Commodity'] == commodity].copy()
        
        if commodity_data.empty:
            logger.warning(f"No data found for {commodity}")
            return None
        
        # Select features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Oil_Price', 'USD_Index', 'CPI', 'Trade_Balance_US',
            'US_Tariff_Index', 'Sanction_Intensity', 'Russia_Sanction_Intensity',
            'US_Sanction_Index'
        ]
        
        available_features = [col for col in feature_columns if col in commodity_data.columns]
        
        # Get last N days
        latest_data = commodity_data.tail(lookback_days)
        features = latest_data[available_features].values
        
        # Flatten for models that need sequences
        if len(features) > 0:
            return features.flatten().reshape(1, -1)
        
        return None
    
    def predict_with_model(self, commodity: str, model_name: str, days: int = 1) -> Optional[float]:
        """
        Predict price using a specific model.
        
        Args:
            commodity: Commodity name
            model_name: Name of the model
            days: Number of days ahead to predict
            
        Returns:
            Predicted price or None
        """
        model_data = self.load_model(commodity, model_name)
        
        if model_data is None:
            return None
        
        model = model_data.get('model')
        if model is None:
            return None
        
        try:
            # For ML models (XGBoost, Random Forest, Linear Regression)
            if model_name in ['XGBoost', 'RandomForest', 'LinearRegression', 'LSTM_Proxy']:
                X = self.get_latest_features(commodity)
                if X is None:
                    return None
                
                # Scale if scaler exists
                scaler_X = model_data.get('scaler_X')
                if scaler_X:
                    X = scaler_X.transform(X)
                
                # Predict
                prediction = model.predict(X)[0]
                
                # Inverse scale if scaler exists
                scaler_y = model_data.get('scaler_y')
                if scaler_y:
                    prediction = scaler_y.inverse_transform([[prediction]])[0][0]
                
                return prediction
            
            # For ARIMA
            elif model_name == 'ARIMA':
                commodity_data = self.data[self.data['Commodity'] == commodity].copy()
                ts = commodity_data['Close'].dropna().values
                
                # Forecast
                forecast = model.forecast(steps=days)
                return forecast[-1] if days > 1 else forecast[0]
            
            # For Prophet
            elif model_name == 'Prophet':
                future = model.make_future_dataframe(periods=days)
                forecast = model.predict(future)
                return forecast.tail(days)['yhat'].values[-1]
            
        except Exception as e:
            logger.error(f"Error predicting with {model_name}: {str(e)}")
            return None
    
    def predict_all_models(self, commodity: str, days: int = 1) -> Dict[str, float]:
        """
        Predict using all available models.
        
        Args:
            commodity: Commodity name
            days: Number of days ahead
            
        Returns:
            Dictionary of model predictions
        """
        predictions = {}
        
        # Try to load available models
        commodity_dir = os.path.join(self.models_dir, commodity.replace(' ', '_'))
        if not os.path.exists(commodity_dir):
            logger.warning(f"No models found for {commodity}")
            return predictions
        
        # Get list of available models
        model_files = [f.replace('.pkl', '') for f in os.listdir(commodity_dir) if f.endswith('.pkl')]
        
        for model_name in model_files:
            prediction = self.predict_with_model(commodity, model_name, days)
            if prediction is not None:
                predictions[model_name] = prediction
        
        return predictions
    
    def get_current_price(self, commodity: str) -> Optional[float]:
        """Get the most recent price for a commodity."""
        commodity_data = self.data[self.data['Commodity'] == commodity].copy()
        if commodity_data.empty:
            return None
        return commodity_data['Close'].iloc[-1]
    
    def print_predictions(self, commodity: str, days: int = 1):
        """Print formatted predictions."""
        print(f"\n{'='*60}")
        print(f"Price Predictions for {commodity}")
        print(f"{'='*60}")
        
        current_price = self.get_current_price(commodity)
        if current_price:
            print(f"Current Price: ${current_price:.2f}")
        
        predictions = self.predict_all_models(commodity, days)
        
        if not predictions:
            print("No models available for prediction")
            return
        
        print(f"\nPredictions for {days} day(s) ahead:")
        print("-" * 60)
        
        for model_name, price in sorted(predictions.items(), key=lambda x: x[1]):
            change = ((price - current_price) / current_price * 100) if current_price else 0
            print(f"{model_name:20s}: ${price:10.2f} ({change:+.2f}%)")
        
        # Average prediction
        if predictions:
            avg_prediction = np.mean(list(predictions.values()))
            avg_change = ((avg_prediction - current_price) / current_price * 100) if current_price else 0
            print("-" * 60)
            print(f"{'Average':20s}: ${avg_prediction:10.2f} ({avg_change:+.2f}%)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Predict metal prices using trained models')
    parser.add_argument('--commodity', type=str, default='all',
                       choices=['Gold_Futures', 'Silver_Futures', 'Crude_Oil_Futures', 'all'],
                       help='Commodity to predict (default: all)')
    parser.add_argument('--days', type=int, default=1,
                       help='Number of days ahead to predict (default: 1)')
    parser.add_argument('--model', type=str, default='all',
                       help='Specific model to use (default: all)')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing trained models (default: models)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PricePredictor(models_dir=args.models_dir)
    predictor.load_data()
    
    # Predict
    if args.commodity == 'all':
        for commodity in predictor.commodities:
            predictor.print_predictions(commodity, args.days)
    else:
        predictor.print_predictions(args.commodity, args.days)


if __name__ == "__main__":
    main()

