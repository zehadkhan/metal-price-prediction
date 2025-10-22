#!/usr/bin/env python3
"""
Example usage of the Global Commodity Economy Data Pipeline

This script demonstrates how to use the CommodityDataPipeline class
for different scenarios and customizations.
"""

from data_pipeline import CommodityDataPipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """
    Example usage of the data pipeline with different configurations.
    """
    
    # Get FRED API key
    fred_api_key = os.getenv('FRED_API_KEY')
    
    if not fred_api_key:
        print("Error: FRED_API_KEY not found in environment variables")
        return
    
    # Initialize pipeline
    pipeline = CommodityDataPipeline(fred_api_key)
    
    print("=== Example 1: Basic Usage ===")
    print("Running pipeline with default settings...")
    pipeline.run_pipeline(
        start_date='2020-01-01',
        end_date='2024-01-01',
        output_file='example_dataset.csv'
    )
    
    print("\n=== Example 2: Custom Date Range ===")
    print("Running pipeline for 2023 data only...")
    pipeline.run_pipeline(
        start_date='2023-01-01',
        end_date='2023-12-31',
        output_file='2023_commodity_data.csv'
    )
    
    print("\n=== Example 3: Accessing Individual Data Sources ===")
    print("Fetching only FRED data...")
    fred_data = pipeline.get_fred_data('2023-01-01', '2023-12-31')
    print(f"FRED data shape: {fred_data.shape}")
    print(f"FRED columns: {list(fred_data.columns)}")
    
    print("\nFetching only Yahoo Finance data...")
    yahoo_data = pipeline.get_yfinance_data('2023-01-01', '2023-12-31')
    print(f"Yahoo Finance data shape: {yahoo_data.shape}")
    print(f"Yahoo Finance columns: {list(yahoo_data.columns)}")
    
    print("\n=== Example 4: Custom Analysis ===")
    if pipeline.combined_data is not None:
        print("Analyzing correlation between Gold and Oil prices...")
        
        # Filter for Gold futures data
        gold_data = pipeline.combined_data[
            pipeline.combined_data['Ticker'] == 'GC=F'
        ].copy()
        
        if not gold_data.empty and 'Oil_Price' in gold_data.columns:
            correlation = gold_data['Close'].corr(gold_data['Oil_Price'])
            print(f"Gold-Oil correlation: {correlation:.4f}")
            
            # Show some statistics
            print(f"\nGold Price Statistics:")
            print(f"Mean: ${gold_data['Close'].mean():.2f}")
            print(f"Min: ${gold_data['Close'].min():.2f}")
            print(f"Max: ${gold_data['Close'].max():.2f}")
            
            print(f"\nOil Price Statistics:")
            print(f"Mean: ${gold_data['Oil_Price'].mean():.2f}")
            print(f"Min: ${gold_data['Oil_Price'].min():.2f}")
            print(f"Max: ${gold_data['Oil_Price'].max():.2f}")

if __name__ == "__main__":
    main()
