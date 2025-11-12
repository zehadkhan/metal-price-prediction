# Real Tariff & Sanction Data Sources

## ⚠️ IMPORTANT: Current Implementation Status

**The current implementation uses PROXY/SIMULATED data**, not real tariff/sanction data. This is a placeholder framework that needs to be connected to real data sources.

## Where to Get REAL Data

### 1. Tariff Data Sources

#### A. USITC DataWeb (United States International Trade Commission)
- **URL**: https://dataweb.usitc.gov/
- **API**: Limited, but data can be downloaded
- **Format**: CSV/Excel downloads
- **Free**: Yes
- **How to Access**: 
  - Manual download from DataWeb interface
  - Or use their Harmonized Tariff Schedule data

#### B. World Trade Organization (WTO) Tariff Database
- **URL**: https://www.wto.org/english/res_e/statis_e/statis_e.htm
- **API**: No official API, but data files available
- **Format**: CSV/Excel
- **Free**: Yes
- **How to Access**: Download from WTO statistics portal

#### C. UN Comtrade (UN Trade Statistics)
- **URL**: https://comtradeplus.un.org/
- **API**: Yes! UN Comtrade API available
- **Format**: JSON/CSV
- **Free**: Yes (with registration)
- **How to Access**: 
  - Register for API key at https://comtradeplus.un.org/
  - Use their REST API

#### D. World Bank WITS (World Integrated Trade Solution)
- **URL**: https://wits.worldbank.org/
- **API**: Limited, but data export available
- **Format**: CSV/Excel
- **Free**: Yes
- **How to Access**: Query and export from WITS interface

### 2. Sanction Data Sources

#### A. OFAC (Office of Foreign Assets Control)
- **URL**: https://ofac.treasury.gov/
- **API**: No official API, but lists are downloadable
- **Format**: PDF/CSV/XML
- **Free**: Yes
- **How to Access**: 
  - Download Sanctions Lists: https://ofac.treasury.gov/specially-designated-nationals-list-sdn-list
  - SDN List: XML/CSV format
  - Sanctions Programs: https://ofac.treasury.gov/sanctions-programs-and-country-information

#### B. UN Sanctions Lists
- **URL**: https://www.un.org/securitycouncil/sanctions/information
- **API**: No official API
- **Format**: PDF/HTML
- **Free**: Yes
- **How to Access**: Manual download from UN Security Council website

#### C. EU Sanctions List
- **URL**: https://ec.europa.eu/info/business-economy-euro/banking-and-finance/international-relations/restrictive-measures-sanctions_en
- **API**: No official API
- **Format**: PDF/CSV
- **Free**: Yes
- **How to Access**: Download from EU website

#### D. Global Trade Alert (Academic/Research)
- **URL**: https://www.globaltradealert.org/
- **API**: Limited access, but data export available
- **Format**: CSV/Excel
- **Free**: Yes (with registration)
- **How to Access**: Register and download datasets

## Recommended Implementation Approach

### Option 1: UN Comtrade API (Best for Tariffs)
```python
# Requires registration at https://comtradeplus.un.org/
import requests

def get_tariff_data_from_comtrade(api_key, country='USA', year=2024):
    url = f"https://comtradeplus.un.org/api/get?"
    params = {
        'type': 'C',
        'freq': 'A',
        'px': 'HS',
        'ps': year,
        'r': '842',  # USA
        'p': 'all',
        'rg': '2',  # Import
        'cc': 'TOTAL',
        'fmt': 'json',
        'token': api_key
    }
    response = requests.get(url, params=params)
    return response.json()
```

### Option 2: OFAC SDN List (Best for Sanctions)
```python
import requests
import xml.etree.ElementTree as ET

def get_ofac_sanctions_list():
    """Download OFAC SDN List"""
    url = "https://ofac.treasury.gov/specially-designated-nationals-list-sdn-list"
    # Note: OFAC provides XML/CSV downloads
    # You'll need to parse their XML format
    pass
```

### Option 3: Manual Download + Processing
1. Download CSV files from USITC/OFAC manually
2. Process and update regularly
3. Store in your database

## Current Implementation Details

The current `tariff_sanction_scraper.py` creates:
- **Proxy data** based on known dates (e.g., Russia-Ukraine 2022-02-24)
- **Simulated trends** using mathematical models
- **Not real API data**

## Next Steps to Get Real Data

1. **Register for UN Comtrade API** (for tariff data)
   - Go to: https://comtradeplus.un.org/
   - Get API key
   - Integrate into scraper

2. **Download OFAC SDN Lists** (for sanctions)
   - Download XML/CSV from OFAC
   - Parse and count sanctions by date
   - Create time series

3. **Use USITC DataWeb** (for US tariffs)
   - Query DataWeb for specific commodities
   - Export data
   - Process into time series

4. **Alternative: Use Academic Datasets**
   - Global Trade Alert
   - Peterson Institute datasets
   - Research papers with tariff/sanction data

## Warning

⚠️ **Do NOT use the current proxy data for:**
- Real trading decisions
- Financial predictions
- Production systems
- Regulatory compliance

✅ **Use proxy data only for:**
- Testing pipeline structure
- Development
- Learning/prototyping

## Example: Real Data Integration

See `tariff_sanction_scraper_real.py` (to be created) for actual implementation with real APIs.

