#!/usr/bin/env python3
"""
Script to view and explore the global commodity economy dataset
"""

import pandas as pd
import numpy as np

def view_data():
    """
    Load and display the dataset with various viewing options
    """
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('global_commodity_economy.csv')
    
    print("="*60)
    print("GLOBAL COMMODITY ECONOMY DATASET")
    print("="*60)
    
    # Basic info
    print(f"\nDataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Show first few rows
    print("\n" + "="*60)
    print("FIRST 10 ROWS")
    print("="*60)
    print(df.head(10))
    
    # Show last few rows
    print("\n" + "="*60)
    print("LAST 10 ROWS")
    print("="*60)
    print(df.tail(10))
    
    # Show data by commodity
    print("\n" + "="*60)
    print("DATA BY COMMODITY")
    print("="*60)
    for commodity in df['Commodity'].unique():
        if pd.notna(commodity):
            print(f"\n{commodity}:")
            commodity_data = df[df['Commodity'] == commodity]
            print(f"  Records: {len(commodity_data)}")
            print(f"  Date Range: {commodity_data['Date'].min()} to {commodity_data['Date'].max()}")
            if 'Close' in commodity_data.columns:
                print(f"  Price Range: ${commodity_data['Close'].min():.2f} - ${commodity_data['Close'].max():.2f}")
    
    # Show economic indicators
    print("\n" + "="*60)
    print("ECONOMIC INDICATORS SUMMARY")
    print("="*60)
    economic_cols = ['Oil_Price', 'USD_Index', 'CPI', 'Trade_Balance_US']
    for col in economic_cols:
        if col in df.columns:
            non_null_data = df[col].dropna()
            if len(non_null_data) > 0:
                print(f"\n{col}:")
                print(f"  Records: {len(non_null_data)}")
                print(f"  Range: {non_null_data.min():.2f} - {non_null_data.max():.2f}")
                print(f"  Mean: {non_null_data.mean():.2f}")
    
    # Show missing data
    print("\n" + "="*60)
    print("MISSING DATA ANALYSIS")
    print("="*60)
    missing_data = df.isnull().sum()
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"{col}: {missing} missing ({missing/len(df)*100:.1f}%)")
    
    # Show recent data
    print("\n" + "="*60)
    print("RECENT DATA (Last 5 days)")
    print("="*60)
    recent_data = df.tail(15)  # Last 15 records (5 days * 3 commodities)
    print(recent_data[['Date', 'Ticker', 'Commodity', 'Close', 'Oil_Price', 'USD_Index']])
    
    return df

def view_specific_data(df, commodity=None, date_range=None):
    """
    View specific subsets of the data
    """
    filtered_df = df.copy()
    
    if commodity:
        filtered_df = filtered_df[filtered_df['Commodity'] == commodity]
        print(f"\nFiltered for {commodity}: {len(filtered_df)} records")
    
    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
        print(f"Filtered for date range {start_date} to {end_date}: {len(filtered_df)} records")
    
    return filtered_df

if __name__ == "__main__":
    # View all data
    df = view_data()
    
    # Example: View only Gold data
    print("\n" + "="*60)
    print("GOLD FUTURES DATA ONLY")
    print("="*60)
    gold_data = view_specific_data(df, commodity='Gold_Futures')
    print(gold_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Oil_Price', 'USD_Index']].head(10))
    
    # Example: View recent data (2024)
    print("\n" + "="*60)
    print("2024 DATA ONLY")
    print("="*60)
    recent_2024 = view_specific_data(df, date_range=('2024-01-01', '2024-12-31'))
    print(f"2024 records: {len(recent_2024)}")
    if len(recent_2024) > 0:
        print(recent_2024[['Date', 'Ticker', 'Commodity', 'Close', 'Oil_Price', 'USD_Index']].head(10))
