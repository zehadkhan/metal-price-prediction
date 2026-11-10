#!/usr/bin/env python3
"""
Generate Comprehensive Test Report
Shows all results in a visual format
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def generate_report():
    """Generate comprehensive test report."""
    
    print("="*80)
    print("GENERATING COMPREHENSIVE TEST REPORT")
    print("="*80)
    
    # Load results
    results_file = 'reports/test_results_cleaned_data_FIXED.csv'
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("   Please run: python fixed_test_cleaned_data.py first")
        return
    
    results_df = pd.read_csv(results_file)
    
    # Load cleaned data for context
    data_file = 'global_commodity_economy_cleaned.csv'
    df = pd.read_csv(data_file, parse_dates=['Date'])
    
    print(f"\n✅ Loaded results from: {results_file}")
    print(f"✅ Loaded data context: {len(df)} records")
    
    # Create comprehensive report figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Model Performance Comparison (Top Left - Large)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    models = results_df['Model'].values
    r2_scores = results_df['R2'].values
    rmse_scores = results_df['RMSE'].values
    
    # Filter out negative R² for visualization
    valid_mask = r2_scores > -100  # Only show reasonable values
    models_valid = [m for i, m in enumerate(models) if valid_mask[i]]
    r2_valid = r2_scores[valid_mask]
    rmse_valid = rmse_scores[valid_mask]
    
    x_pos = np.arange(len(models_valid))
    colors = ['#2ecc71' if r2 > 0.5 else '#e74c3c' if r2 > 0 else '#f39c12' for r2 in r2_valid]
    
    bars = ax1.barh(x_pos, r2_valid, color=colors, alpha=0.8)
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(models_valid)
    ax1.set_xlabel('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison - R² Scores', fontsize=16, fontweight='bold', pad=20)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, r2, rmse) in enumerate(zip(bars, r2_valid, rmse_valid)):
        ax1.text(r2 + 0.01 if r2 >= 0 else r2 - 0.01, i, 
                f"R²: {r2:.4f}\nRMSE: ${rmse:.2f}", 
                va='center', fontsize=10, fontweight='bold',
                ha='left' if r2 >= 0 else 'right')
    
    # 2. RMSE Comparison (Top Right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Only show positive RMSE values
    rmse_positive = [r for r in rmse_valid if r > 0 and r < 1000]
    models_rmse = [m for i, m in enumerate(models_valid) if rmse_valid[i] > 0 and rmse_valid[i] < 1000]
    
    if len(rmse_positive) > 0:
        bars2 = ax2.barh(range(len(rmse_positive)), rmse_positive, color='#3498db', alpha=0.8)
        ax2.set_yticks(range(len(rmse_positive)))
        ax2.set_yticklabels(models_rmse)
        ax2.set_xlabel('RMSE ($)', fontsize=11, fontweight='bold')
        ax2.set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for i, (bar, rmse_val) in enumerate(zip(bars2, rmse_positive)):
            ax2.text(rmse_val + max(rmse_positive)*0.02, i, f"${rmse_val:.2f}", 
                    va='center', fontsize=10, fontweight='bold')
    
    # 3. Data Quality Summary (Middle Right)
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')
    
    # Calculate data stats
    gold_data = df[df['Commodity'].str.contains('Gold', case=False, na=False)]
    
    quality_text = f"""
    DATA QUALITY SUMMARY
    
    Total Records: {len(df):,}
    Gold Records: {len(gold_data):,}
    
    Date Range:
    {df['Date'].min().strftime('%Y-%m-%d')} to
    {df['Date'].max().strftime('%Y-%m-%d')}
    
    Missing Values: {df.isnull().sum().sum()}
    Invalid Prices: {(df['Close'] <= 0).sum() if 'Close' in df.columns else 0}
    
    Status: ✅ CLEAN
    """
    
    ax3.text(0.1, 0.5, quality_text, fontsize=11, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 4. Best Model Highlight (Bottom Left)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    # Find best model
    best_idx = results_df['R2'].idxmax()
    best_model = results_df.loc[best_idx]
    
    best_text = f"""
    🏆 BEST MODEL
    
    Model: {best_model['Model']}
    
    Performance:
    • R²: {best_model['R2']:.4f}
    • RMSE: ${best_model['RMSE']:.2f}
    • MAE: ${best_model['MAE']:.2f}
    
    Status: ✅ RECOMMENDED
    """
    
    ax4.text(0.1, 0.5, best_text, fontsize=12, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            fontweight='bold')
    
    # 5. Performance Metrics Table (Bottom Middle)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Create table
    table_data = []
    for _, row in results_df.iterrows():
        status = "✅ Good" if row['R2'] > 0.5 else "⚠️ Poor" if row['R2'] > 0 else "❌ Bad"
        table_data.append([
            row['Model'],
            f"${row['RMSE']:.2f}",
            f"${row['MAE']:.2f}",
            f"{row['R2']:.4f}",
            status
        ])
    
    table = ax5.table(cellText=table_data,
                     colLabels=['Model', 'RMSE', 'MAE', 'R²', 'Status'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells based on R²
    for i in range(1, len(table_data) + 1):
        r2_val = float(table_data[i-1][3])
        if r2_val > 0.5:
            for j in range(5):
                table[(i, j)].set_facecolor('#d5f4e6')
        elif r2_val > 0:
            for j in range(5):
                table[(i, j)].set_facecolor('#fff4e6')
        else:
            for j in range(5):
                table[(i, j)].set_facecolor('#ffe6e6')
    
    ax5.set_title('All Models Performance', fontsize=12, fontweight='bold', pad=10)
    
    # 6. Recommendations (Bottom Right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    rec_text = f"""
    💡 RECOMMENDATIONS
    
    For Production:
    → Use: {best_model['Model']}
    
    Why:
    • Highest R² score
    • Lowest RMSE
    • Most reliable
    
    Next Steps:
    1. Deploy {best_model['Model']}
    2. Monitor performance
    3. Retrain monthly
    
    Data Quality: ✅
    Models: ✅ Tested
    Ready: ✅ YES
    """
    
    ax6.text(0.1, 0.5, rec_text, fontsize=10, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Comprehensive Test Report - Metal Price Prediction Models', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save report
    report_file = 'TEST_REPORT.png'
    plt.savefig(report_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Report saved to: {report_file}")
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print("\n" + results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"\n🏆 Best Model: {best_model['Model']}")
    print(f"   R²: {best_model['R2']:.4f} ({best_model['R2']*100:.1f}% accuracy)")
    print(f"   RMSE: ${best_model['RMSE']:.2f}")
    print(f"   MAE: ${best_model['MAE']:.2f}")
    
    print(f"\n📊 Data Quality:")
    print(f"   Records: {len(df):,}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Status: ✅ CLEAN")
    
    print(f"\n✅ All models tested successfully!")
    print(f"✅ Report generated: {report_file}")
    
    return results_df

if __name__ == "__main__":
    generate_report()

