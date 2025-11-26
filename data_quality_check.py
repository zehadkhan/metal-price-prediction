#!/usr/bin/env python3
"""
Comprehensive Data Quality Check and Cleaning Script
Identifies and fixes data quality issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality(df):
    """Comprehensive data quality analysis."""
    
    print("="*80)
    print("COMPREHENSIVE DATA QUALITY ANALYSIS")
    print("="*80)
    
    # Basic info
    print(f"\n📊 BASIC INFORMATION:")
    print(f"  Shape: {df.shape}")
    print(f"  Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Total Records: {len(df)}")
    
    # Check duplicates
    print(f"\n🔍 DUPLICATE CHECK:")
    dup_date_commodity = df.duplicated(subset=['Date', 'Commodity']).sum()
    dup_date = df.duplicated(subset=['Date']).sum()
    print(f"  Duplicate (Date + Commodity): {dup_date_commodity}")
    print(f"  Duplicate Dates (any commodity): {dup_date}")
    
    if dup_date_commodity > 0:
        print(f"  ⚠️ Found {dup_date_commodity} duplicate records!")
        duplicates = df[df.duplicated(subset=['Date', 'Commodity'], keep=False)]
        print(f"  Sample duplicates:\n{duplicates[['Date', 'Commodity', 'Close']].head(10)}")
    
    # Missing values
    print(f"\n🔍 MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("  ✅ No missing values!")
    
    # Check each commodity
    print(f"\n📈 COMMODITY-SPECIFIC ANALYSIS:")
    for commodity in df['Commodity'].dropna().unique():
        print(f"\n  {commodity}:")
        comm_df = df[df['Commodity'] == commodity].copy()
        print(f"    Records: {len(comm_df)}")
        print(f"    Date range: {comm_df['Date'].min()} to {comm_df['Date'].max()}")
        print(f"    Unique dates: {comm_df['Date'].nunique()}")
        
        if 'Close' in comm_df.columns:
            price = comm_df['Close'].dropna()
            if len(price) > 0:
                print(f"    Price range: ${price.min():.2f} - ${price.max():.2f}")
                print(f"    Price mean: ${price.mean():.2f}")
                
                # Outliers
                z_scores = np.abs((price - price.mean()) / (price.std() + 1e-9))
                outliers = (z_scores > 3).sum()
                print(f"    Outliers (>3 std): {outliers} ({outliers/len(price)*100:.1f}%)")
                
                # Check for zero or negative prices
                invalid_prices = (price <= 0).sum()
                if invalid_prices > 0:
                    print(f"    ⚠️ Invalid prices (<=0): {invalid_prices}")
    
    # Date gaps analysis
    print(f"\n📅 DATE GAP ANALYSIS:")
    for commodity in df['Commodity'].dropna().unique():
        comm_df = df[df['Commodity'] == commodity].copy()
        comm_df = comm_df.sort_values('Date')
        comm_df['date_diff'] = comm_df['Date'].diff().dt.days
        
        # Expected daily data
        gaps = comm_df[comm_df['date_diff'] > 1]
        if len(gaps) > 0:
            print(f"  {commodity}: {len(gaps)} gaps found")
            print(f"    Largest gap: {gaps['date_diff'].max()} days")
        else:
            print(f"  {commodity}: ✅ No gaps (continuous data)")
    
    # Data type issues
    print(f"\n🔧 DATA TYPE CHECK:")
    numeric_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'Oil_Price', 'USD_Index', 'CPI']
    for col in numeric_cols:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            if non_numeric > 0:
                print(f"  ⚠️ {col}: {non_numeric} non-numeric values")
    
    return missing_df

def clean_data(df):
    """Clean and refine the data."""
    
    print("\n" + "="*80)
    print("DATA CLEANING PROCESS")
    print("="*80)
    
    df_clean = df.copy()
    original_len = len(df_clean)
    
    # Step 1: Remove duplicates
    print(f"\n1️⃣ Removing duplicates...")
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['Date', 'Commodity'], keep='last')
    removed = before - len(df_clean)
    print(f"   Removed {removed} duplicate records")
    
    # Step 2: Fix date issues
    print(f"\n2️⃣ Fixing date issues...")
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Date'])
    print(f"   Removed {original_len - len(df_clean) - removed} invalid dates")
    
    # Step 3: Remove invalid prices
    print(f"\n3️⃣ Removing invalid prices...")
    before = len(df_clean)
    if 'Close' in df_clean.columns:
        df_clean = df_clean[df_clean['Close'] > 0]
    if 'Open' in df_clean.columns:
        df_clean = df_clean[df_clean['Open'] > 0]
    removed = before - len(df_clean)
    print(f"   Removed {removed} records with invalid prices")
    
    # Step 4: Handle outliers (winsorize at 99th percentile)
    print(f"\n4️⃣ Handling outliers (winsorizing at 99th percentile)...")
    for commodity in df_clean['Commodity'].dropna().unique():
        comm_mask = df_clean['Commodity'] == commodity
        if 'Close' in df_clean.columns:
            price_col = df_clean.loc[comm_mask, 'Close']
            upper_limit = price_col.quantile(0.99)
            lower_limit = price_col.quantile(0.01)
            
            # Cap outliers
            df_clean.loc[comm_mask & (df_clean['Close'] > upper_limit), 'Close'] = upper_limit
            df_clean.loc[comm_mask & (df_clean['Close'] < lower_limit), 'Close'] = lower_limit
    
    # Step 5: Sort by date
    print(f"\n5️⃣ Sorting by date...")
    df_clean = df_clean.sort_values(['Date', 'Commodity']).reset_index(drop=True)
    
    # Step 6: Forward fill missing values for each commodity separately
    print(f"\n6️⃣ Forward filling missing values (by commodity)...")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for commodity in df_clean['Commodity'].dropna().unique():
        comm_mask = df_clean['Commodity'] == commodity
        df_clean.loc[comm_mask, numeric_cols] = df_clean.loc[comm_mask, numeric_cols].ffill().bfill()
    
    # Step 7: Remove rows where key columns are still missing
    print(f"\n7️⃣ Removing rows with critical missing data...")
    before = len(df_clean)
    key_cols = ['Date', 'Commodity']
    if 'Close' in df_clean.columns:
        key_cols.append('Close')
    df_clean = df_clean.dropna(subset=key_cols)
    removed = before - len(df_clean)
    print(f"   Removed {removed} records with critical missing data")
    
    print(f"\n✅ Cleaning complete!")
    print(f"   Original records: {original_len}")
    print(f"   Cleaned records: {len(df_clean)}")
    print(f"   Removed: {original_len - len(df_clean)} ({((original_len - len(df_clean))/original_len*100):.1f}%)")
    
    return df_clean

def create_quality_report(df_original, df_cleaned):
    """Create visual quality report."""
    
    print("\n" + "="*80)
    print("CREATING QUALITY REPORT")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Missing values comparison
    ax1 = axes[0, 0]
    missing_orig = df_original.isnull().sum()
    missing_clean = df_cleaned.isnull().sum()
    
    comparison = pd.DataFrame({
        'Original': missing_orig,
        'Cleaned': missing_clean
    }).head(10)
    
    comparison.plot(kind='bar', ax=ax1)
    ax1.set_title('Missing Values: Before vs After Cleaning')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Price distribution for Gold
    ax2 = axes[0, 1]
    if 'Commodity' in df_cleaned.columns and 'Close' in df_cleaned.columns:
        gold_data = df_cleaned[df_cleaned['Commodity'].str.contains('Gold', case=False, na=False)]
        if len(gold_data) > 0:
            gold_data['Close'].hist(bins=50, ax=ax2, alpha=0.7)
            ax2.set_title('Gold Price Distribution (Cleaned)')
            ax2.set_xlabel('Price ($)')
            ax2.set_ylabel('Frequency')
    
    # 3. Records by commodity
    ax3 = axes[1, 0]
    if 'Commodity' in df_cleaned.columns:
        comm_counts = df_cleaned['Commodity'].value_counts()
        comm_counts.plot(kind='bar', ax=ax3)
        ax3.set_title('Records by Commodity (Cleaned)')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Date coverage
    ax4 = axes[1, 1]
    if 'Date' in df_cleaned.columns:
        df_cleaned['Year'] = pd.to_datetime(df_cleaned['Date']).dt.year
        year_counts = df_cleaned['Year'].value_counts().sort_index()
        year_counts.plot(kind='line', ax=ax4, marker='o')
        ax4.set_title('Data Coverage by Year')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Records')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_quality_report.png', dpi=150, bbox_inches='tight')
    print("✅ Quality report saved to 'data_quality_report.png'")
    plt.close()

def main():
    """Main function."""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('global_commodity_economy.csv', parse_dates=['Date'])
    
    # Analyze
    missing_df = analyze_data_quality(df)
    
    # Clean
    df_cleaned = clean_data(df)
    
    # Create report
    create_quality_report(df, df_cleaned)
    
    # Save cleaned data
    output_file = 'global_commodity_economy_cleaned.csv'
    df_cleaned.to_csv(output_file, index=False)
    print(f"\n✅ Cleaned data saved to '{output_file}'")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original: {len(df)} records")
    print(f"Cleaned: {len(df_cleaned)} records")
    print(f"Removed: {len(df) - len(df_cleaned)} records")
    print(f"\nUse '{output_file}' for model training!")

if __name__ == "__main__":
    main()

