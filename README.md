# Global Commodity Economy Data Pipeline

A comprehensive Python script that collects and combines macroeconomic and market data for commodities using public APIs (FRED + Yahoo Finance).

## 🎯 Features

- **FRED API Integration**: Fetches economic indicators from Federal Reserve Economic Data
- **Yahoo Finance Integration**: Collects commodity futures OHLCV data
- **Data Alignment**: Merges all data by date with intelligent handling of missing values
- **Visualization**: Generates correlation matrices and trend plots
- **Error Handling**: Robust error handling for API failures and rate limits
- **Modular Design**: Clean, maintainable code structure

## 📦 Data Sources

### FRED Economic Indicators
- **Gold Price**: GOLDAMGBD228NLBM
- **Silver Price**: SLVPRUSD
- **Oil Price**: DCOILWTICO
- **USD Index**: DTWEXBGS
- **CPI**: CPIAUCSL
- **Trade Balance US**: NETEXP
- **US Tariff Index**: AEXCA
- **Russia Trade Balance**: RUSNTBAL (sanction proxy)

### Yahoo Finance Commodities
- **GC=F**: Gold Futures
- **SI=F**: Silver Futures
- **CL=F**: Crude Oil Futures

## 🛠️ Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   The `.env` file is already configured with the FRED API key.

## 🚀 Usage

### Basic Usage
```bash
python data_pipeline.py
```

### Custom Date Range
```python
from data_pipeline import CommodityDataPipeline

# Initialize pipeline
pipeline = CommodityDataPipeline(fred_api_key="your_api_key")

# Run with custom date range
pipeline.run_pipeline(
    start_date='2020-01-01',
    end_date='2024-01-01',
    output_file='custom_dataset.csv'
)
```

## 📊 Output

The pipeline generates:

1. **CSV File**: `global_commodity_economy.csv` with combined dataset
2. **Correlation Plot**: `correlation_plot.png` showing relationships between indicators
3. **Trend Plot**: `trend_plot.png` showing time series of key indicators
4. **Summary Statistics**: Printed to console

## 📈 Dataset Structure

Each record includes:
- **Date**: YYYY-MM-DD format
- **OHLCV Data**: Open, High, Low, Close, Volume (from Yahoo Finance)
- **Ticker**: Symbol (GC=F, SI=F, CL=F)
- **Commodity**: Name (Gold_Futures, Silver_Futures, Crude_Oil_Futures)
- **Economic Indicators**: Gold_Price, Silver_Price, Oil_Price, USD_Index, CPI, Trade_Balance_US, US_Tariff_Index, Sanction_Proxy_Russia_Trade

## 🔧 Configuration

### Environment Variables
- `FRED_API_KEY`: Your FRED API key (already configured)

### Customization
You can modify the data sources by editing the `fred_series` and `yahoo_symbols` dictionaries in the `CommodityDataPipeline` class.

## 📋 Requirements

- Python 3.7+
- Internet connection for API calls
- FRED API key (included)

## 🚨 Error Handling

The pipeline includes comprehensive error handling for:
- API rate limits
- Network failures
- Missing data
- Invalid API responses

## 📊 Example Output

```
Dataset Shape: (1461, 15)
Date Range: 2020-01-01 to 2024-01-01
Total Records: 1461

Columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Commodity', 'Gold_Price', 'Silver_Price', 'Oil_Price', 'USD_Index', 'CPI', 'Trade_Balance_US', 'US_Tariff_Index', 'Sanction_Proxy_Russia_Trade']
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.
