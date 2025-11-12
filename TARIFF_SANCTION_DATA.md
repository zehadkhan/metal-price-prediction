# Tariff and Sanction Data Integration

## Overview

The data pipeline has been enhanced with comprehensive tariff and sanction data collection capabilities. This enables better prediction accuracy by incorporating trade policy and geopolitical factors that significantly impact commodity markets.

## New Data Sources

### 1. Tariff Data Indicators

The pipeline now includes multiple tariff-related indicators:

- **US_Tariff_Index**: Overall US tariff policy index (100 = baseline, increases reflect trade tensions)
- **China_Tariff_Impact**: Specific tariff impact on China-US trade (0.8x multiplier)
- **EU_Tariff_Impact**: Specific tariff impact on EU-US trade (0.6x multiplier)
- **Tariff_Effect_Proxy**: Trade-weighted exchange index as tariff effect proxy
- **Import_Price_Index**: Import price index serving as tariff proxy
- **Trade_Weighted_Dollar**: Trade-weighted dollar index

### 2. Sanction Data Indicators

The pipeline includes comprehensive sanction tracking:

- **Sanction_Intensity**: Overall global sanction intensity (0-100 scale)
- **Russia_Sanction_Intensity**: Russia-specific sanction intensity
- **US_Sanction_Index**: US-specific sanction activity index
- **Sanction_Proxy_Russia_Trade**: Inverse proxy for Russia trade (decreases with sanctions)
- **Russia_Trade_Proxy**: Direct Russia trade proxy indicator

## Major Sanction Events Tracked

The system automatically tracks major sanction events:

- **2020-01-01**: Base level (10)
- **2022-02-24**: Russia-Ukraine conflict start (85)
- **2022-03-15**: Escalation (90)
- **2022-04-01**: Additional sanctions (95)
- **2023-01-01**: Peak sanctions (100)
- **2024-01-01**: Ongoing sanctions (95)

## Tariff Events Tracked

- **2020-01-01**: Base level (100)
- **2020-07-06**: Trade tensions (105)
- **2021-01-01**: Continued policies (110)
- **2022-02-24**: Russia sanctions impact (115)
- **2023-01-01**: Ongoing trade policies (120)
- **2024-01-01**: Current level (125)

## Dataset Structure

The enhanced dataset now includes **21 columns**:

### Original Columns:
- Date, Open, High, Low, Close, Volume
- Ticker, Commodity
- Oil_Price, USD_Index, CPI, Trade_Balance_US

### New Tariff/Sanction Columns:
- Tariff_Effect_Proxy
- Import_Price_Index
- Trade_Weighted_Dollar
- US_Tariff_Index
- China_Tariff_Impact
- EU_Tariff_Impact
- Sanction_Intensity
- Russia_Trade_Proxy
- US_Sanction_Index
- Sanction_Proxy_Russia_Trade

## Usage

The tariff and sanction data is automatically included when running the pipeline:

```bash
python data_pipeline.py
```

The data is merged with existing FRED and Yahoo Finance data by date, ensuring all indicators are aligned for predictive modeling.

## Prediction Benefits

With tariff and sanction data, your predictions can now account for:

1. **Trade Policy Changes**: How tariffs affect commodity prices
2. **Geopolitical Events**: Impact of sanctions on supply chains
3. **Regional Tensions**: China-US and EU-US trade dynamics
4. **Russia Trade Impact**: Sanction effects on commodity markets
5. **Time-Series Patterns**: Historical correlation between policies and prices

## Data Quality

- **Coverage**: Daily data from 2020-01-01 to present
- **Completeness**: All dates have tariff/sanction indicators
- **Alignment**: Properly merged with commodity data by date
- **Forward Fill**: Missing values handled intelligently

## Future Enhancements

The scraper can be extended to:
- Scrape actual tariff data from USITC
- Integrate OFAC sanctions database
- Add WTO tariff data
- Include country-specific trade tensions
- Real-time sanction event detection

## Example Analysis

```python
import pandas as pd

df = pd.read_csv('global_commodity_economy.csv')

# Analyze correlation between sanctions and gold prices
gold_data = df[df['Ticker'] == 'GC=F']
correlation = gold_data['Close'].corr(gold_data['Russia_Sanction_Intensity'])
print(f"Gold-Russia Sanction Correlation: {correlation:.3f}")

# Analyze tariff impact on oil
oil_data = df[df['Ticker'] == 'CL=F']
tariff_correlation = oil_data['Close'].corr(oil_data['US_Tariff_Index'])
print(f"Oil-US Tariff Correlation: {tariff_correlation:.3f}")
```

## Notes

- The current implementation uses proxy data based on known events
- For production use, consider integrating actual API data from:
  - USITC DataWeb
  - OFAC Sanctions Lists
  - WTO Tariff Database
  - Global Trade Alert

