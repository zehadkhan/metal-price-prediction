#!/usr/bin/env python3
"""
Global Commodity Economy Data Pipeline

This script collects and combines macroeconomic and market data for commodities
using public APIs (FRED + Yahoo Finance) into a unified dataset.

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
from typing import Dict, List, Optional, Tuple
import logging

# Third-party imports
from fredapi import Fred
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class CommodityDataPipeline:
    """
    A comprehensive data pipeline for collecting and combining 
    macroeconomic and commodity market data.
    """
    
    def __init__(self, fred_api_key: str):
        """
        Initialize the data pipeline.
        
        Args:
            fred_api_key (str): FRED API key for accessing economic data
        """
        self.fred_api_key = fred_api_key
        self.fred = Fred(api_key=fred_api_key)
        
        # FRED series mappings (using working series)
        self.fred_series = {
            'Oil_Price': 'DCOILWTICO',
            'USD_Index': 'DTWEXBGS',
            'CPI': 'CPIAUCSL',
            'Trade_Balance_US': 'NETEXP'
        }
        
        # Yahoo Finance symbols
        self.yahoo_symbols = {
            'GC=F': 'Gold_Futures',
            'SI=F': 'Silver_Futures', 
            'CL=F': 'Crude_Oil_Futures'
        }
        
        self.combined_data = None
        
    def get_fred_data(self, start_date: str = '2020-01-01', end_date: str = None) -> pd.DataFrame:
        """
        Fetch economic data from FRED API.
        
        Args:
            start_date (str): Start date for data collection (YYYY-MM-DD)
            end_date (str): End date for data collection (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Combined FRED data with all economic indicators
        """
        logger.info("Fetching FRED economic data...")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        fred_data = {}
        
        for indicator, series_id in self.fred_series.items():
            try:
                logger.info(f"Fetching {indicator} ({series_id})...")
                data = self.fred.get_series(
                    series_id, 
                    start=start_date, 
                    end=end_date,
                    observation_start=start_date,
                    observation_end=end_date
                )
                
                if not data.empty:
                    fred_data[indicator] = data
                    logger.info(f"Successfully fetched {len(data)} records for {indicator}")
                else:
                    logger.warning(f"No data found for {indicator} ({series_id})")
                    
            except Exception as e:
                logger.error(f"Error fetching {indicator} ({series_id}): {str(e)}")
                continue
        
        if not fred_data:
            logger.error("No FRED data was successfully fetched")
            return pd.DataFrame()
            
        # Combine all FRED data
        combined_fred = pd.DataFrame(fred_data)
        combined_fred.index.name = 'Date'
        combined_fred = combined_fred.reset_index()
        
        logger.info(f"Successfully combined {len(combined_fred)} records from FRED")
        return combined_fred
    
    def get_yfinance_data(self, start_date: str = '2020-01-01', end_date: str = None) -> pd.DataFrame:
        """
        Fetch commodity futures data from Yahoo Finance.
        
        Args:
            start_date (str): Start date for data collection (YYYY-MM-DD)
            end_date (str): End date for data collection (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Combined Yahoo Finance data with OHLCV for all symbols
        """
        logger.info("Fetching Yahoo Finance commodity data...")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        yahoo_data = []
        
        for symbol, name in self.yahoo_symbols.items():
            try:
                logger.info(f"Fetching {symbol} ({name})...")
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Reset index to get Date as column
                    data = data.reset_index()
                    data['Ticker'] = symbol
                    data['Commodity'] = name
                    yahoo_data.append(data)
                    logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                continue
        
        if not yahoo_data:
            logger.error("No Yahoo Finance data was successfully fetched")
            return pd.DataFrame()
            
        # Combine all Yahoo Finance data
        combined_yahoo = pd.concat(yahoo_data, ignore_index=True)
        
        # Rename columns to match expected format
        column_mapping = {
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }
        
        # Keep only the columns we need
        combined_yahoo = combined_yahoo[list(column_mapping.keys()) + ['Ticker', 'Commodity']]
        combined_yahoo = combined_yahoo.rename(columns=column_mapping)
        
        logger.info(f"Successfully combined {len(combined_yahoo)} records from Yahoo Finance")
        return combined_yahoo
    
    def merge_data(self, fred_data: pd.DataFrame, yahoo_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge FRED and Yahoo Finance data by date.
        
        Args:
            fred_data (pd.DataFrame): FRED economic data
            yahoo_data (pd.DataFrame): Yahoo Finance commodity data
            
        Returns:
            pd.DataFrame: Merged dataset with all indicators
        """
        logger.info("Merging FRED and Yahoo Finance data...")
        
        if fred_data.empty and yahoo_data.empty:
            logger.error("No data to merge")
            return pd.DataFrame()
        
        # Convert date columns to datetime and normalize timezones
        if not fred_data.empty:
            fred_data['Date'] = pd.to_datetime(fred_data['Date']).dt.tz_localize(None)
        if not yahoo_data.empty:
            yahoo_data['Date'] = pd.to_datetime(yahoo_data['Date']).dt.tz_localize(None)
        
        # Start with the dataset that has more data
        if len(fred_data) > len(yahoo_data):
            base_data = fred_data.copy()
            merge_data = yahoo_data
        else:
            base_data = yahoo_data.copy()
            merge_data = fred_data
        
        # Perform the merge
        if not merge_data.empty:
            merged = pd.merge(
                base_data, 
                merge_data, 
                on='Date', 
                how='outer'
            )
        else:
            merged = base_data
        
        # Sort by date
        merged = merged.sort_values('Date').reset_index(drop=True)
        
        # Handle missing values with forward fill
        numeric_columns = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_columns] = merged[numeric_columns].fillna(method='ffill')
        
        # Drop rows where all key indicators are missing
        key_indicators = ['Gold_Price', 'Silver_Price', 'Oil_Price', 'USD_Index', 'CPI']
        available_indicators = [col for col in key_indicators if col in merged.columns]
        
        if available_indicators:
            merged = merged.dropna(subset=available_indicators, how='all')
        
        logger.info(f"Successfully merged data: {len(merged)} records")
        return merged
    
    def generate_correlation_plot(self, data: pd.DataFrame, save_path: str = 'correlation_plot.png'):
        """
        Generate correlation plot for key economic indicators.
        
        Args:
            data (pd.DataFrame): Combined dataset
            save_path (str): Path to save the correlation plot
        """
        logger.info("Generating correlation plot...")
        
        # Select numeric columns for correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Focus on key indicators
        key_indicators = ['Gold_Price', 'Silver_Price', 'Oil_Price', 'USD_Index', 'CPI', 'Close']
        available_indicators = [col for col in key_indicators if col in numeric_cols]
        
        if len(available_indicators) < 2:
            logger.warning("Not enough indicators for correlation plot")
            return
        
        # Calculate correlation matrix
        corr_data = data[available_indicators].corr()
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_data, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title('Correlation Matrix: Key Economic Indicators', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Correlation plot saved to {save_path}")
    
    def generate_trend_plot(self, data: pd.DataFrame, save_path: str = 'trend_plot.png'):
        """
        Generate trend plot for major economic indicators.
        
        Args:
            data (pd.DataFrame): Combined dataset
            save_path (str): Path to save the trend plot
        """
        logger.info("Generating trend plot...")
        
        # Select key indicators for plotting
        plot_indicators = ['Gold_Price', 'Silver_Price', 'Oil_Price', 'USD_Index', 'CPI']
        available_indicators = [col for col in plot_indicators if col in data.columns]
        
        if not available_indicators:
            logger.warning("No indicators available for trend plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(len(available_indicators), 1, figsize=(15, 3*len(available_indicators)))
        if len(available_indicators) == 1:
            axes = [axes]
        
        for i, indicator in enumerate(available_indicators):
            if indicator in data.columns:
                axes[i].plot(data['Date'], data[indicator], linewidth=2, label=indicator)
                axes[i].set_title(f'{indicator} Over Time', fontweight='bold')
                axes[i].set_ylabel(indicator)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Trend plot saved to {save_path}")
    
    def print_summary_stats(self, data: pd.DataFrame):
        """
        Print summary statistics for the merged dataset.
        
        Args:
            data (pd.DataFrame): Combined dataset
        """
        logger.info("Dataset Summary Statistics:")
        print("\n" + "="*60)
        print("GLOBAL COMMODITY ECONOMY DATASET SUMMARY")
        print("="*60)
        
        print(f"\nDataset Shape: {data.shape}")
        print(f"Date Range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"Total Records: {len(data)}")
        
        print(f"\nColumns: {list(data.columns)}")
        
        print("\nMissing Values:")
        missing_data = data.isnull().sum()
        for col, missing in missing_data.items():
            if missing > 0:
                print(f"  {col}: {missing} ({missing/len(data)*100:.1f}%)")
        
        print("\nNumeric Summary:")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(data[numeric_cols].describe())
        
        print("\nCommodity Distribution:")
        if 'Commodity' in data.columns:
            print(data['Commodity'].value_counts())
        
        print("\nTicker Distribution:")
        if 'Ticker' in data.columns:
            print(data['Ticker'].value_counts())
        
        print("="*60)
    
    def run_pipeline(self, start_date: str = '2020-01-01', end_date: str = None, 
                    output_file: str = 'global_commodity_economy.csv'):
        """
        Run the complete data pipeline.
        
        Args:
            start_date (str): Start date for data collection
            end_date (str): End date for data collection  
            output_file (str): Output CSV filename
        """
        logger.info("Starting Global Commodity Economy Data Pipeline...")
        
        try:
            # Fetch FRED data
            fred_data = self.get_fred_data(start_date, end_date)
            
            # Fetch Yahoo Finance data
            yahoo_data = self.get_yfinance_data(start_date, end_date)
            
            # Merge the data
            self.combined_data = self.merge_data(fred_data, yahoo_data)
            
            if self.combined_data.empty:
                logger.error("Pipeline failed: No data was successfully collected")
                return
            
            # Save to CSV
            self.combined_data.to_csv(output_file, index=False)
            logger.info(f"Data saved to {output_file}")
            
            # Generate plots
            self.generate_correlation_plot(self.combined_data)
            self.generate_trend_plot(self.combined_data)
            
            # Print summary statistics
            self.print_summary_stats(self.combined_data)
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise


def main():
    """
    Main function to run the data pipeline.
    """
    # Get FRED API key from environment
    fred_api_key = os.getenv('FRED_API_KEY')
    
    if not fred_api_key:
        logger.error("FRED_API_KEY not found in environment variables")
        return
    
    # Initialize pipeline
    pipeline = CommodityDataPipeline(fred_api_key)
    
    # Run the pipeline
    pipeline.run_pipeline(
        start_date='2020-01-01',
        end_date=None,  # Will use current date
        output_file='global_commodity_economy.csv'
    )


if __name__ == "__main__":
    main()
