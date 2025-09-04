"""
Data processing utilities for crypto prediction.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import warnings
import zipfile
import io

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

# Removed download_binance_klines_data - using only Binance Vision

def create_minimal_features(df, lag_period=3):
    """Create minimal feature set for memory efficiency."""
    data = df.copy()
    
    # Essential time features
    data['Hour'] = data['Open time'].dt.hour
    data['Minute'] = data['Open time'].dt.minute
    data['Interval_5min'] = data['Hour'] * 12 + data['Minute'] // 5
    data['Interval_sin'] = np.sin(2 * np.pi * data['Interval_5min'] / 288)
    data['Interval_cos'] = np.cos(2 * np.pi * data['Interval_5min'] / 288)
    
    # Essential price features
    data['Price_Range'] = data['High'] - data['Low']
    data['Price_Change'] = data['Close'] - data['Open']
    
    # Minimal lag features
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ohlcv_cols:
        for lag in range(1, lag_period + 1):
            data[f'{col}_Lag_{lag}'] = data[col].shift(lag)
    
    return data

def create_sliding_windows(data, sequence_length=5, target_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
    """Create sliding windows for time series data."""
    # Select feature columns (exclude time and target columns)
    feature_cols = [col for col in data.columns if col not in ['Open time', 'Close time'] + target_cols]
    
    # Remove rows with NaN values
    data_clean = data.dropna()
    
    if len(data_clean) < sequence_length:
        return np.array([]), np.array([]), feature_cols
    
    X, y = [], []
    for i in range(sequence_length, len(data_clean)):
        # Input: sequence_length previous candles
        X.append(data_clean[feature_cols].iloc[i-sequence_length:i].values)
        
        # Target: next candle's OHLCV values
        y.append(data_clean[target_cols].iloc[i].values)
    
    return np.array(X), np.array(y), feature_cols

def download_binance_vision_data(symbol, interval, year, month):
    """Download data from Binance Vision (alternative method)."""
    print(f"📥 Downloading {symbol} {interval} data for {year}-{month} from Binance Vision...")
    
    download_url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month}.zip"
    
    try:
        response = requests.get(download_url)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            csv_filename = f"{symbol}-{interval}-{year}-{month}.csv"
            with zip_file.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file, names=[
                    'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close time', 'Quote asset volume', 'Number of trades',
                    'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
                ])
        
        # Convert data types
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✅ Downloaded {len(df)} rows for {year}-{month}")
        return df
        
    except Exception as e:
        print(f"❌ Error downloading from Binance Vision: {e}")
        return None

def load_multiple_months_data(symbol, interval, year, months, max_rows=50000):
    """Load data for multiple months with memory limits using Binance Vision only."""
    print(f"🔄 Loading data with memory efficiency (max {max_rows:,} rows)...")
    
    all_data = []
    total_rows = 0
    
    for month in months:
        if total_rows >= max_rows:
            print(f"⚠️  Reached row limit at {total_rows:,} rows")
            break
        
        # Use Binance Vision only
        df = download_binance_vision_data(symbol, interval, year, month)
        
        if df is not None:
            # Limit rows if approaching limit
            remaining_rows = max_rows - total_rows
            if len(df) > remaining_rows:
                df = df.head(remaining_rows)
                print(f"   Truncated to {len(df)} rows to stay within memory limit")
            
            all_data.append(df)
            total_rows += len(df)
            
            print(f"   Total rows: {total_rows:,}")
        else:
            print(f"   Failed to download {year}-{month}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ Data loading completed!")
        print(f"   Final shape: {combined_df.shape}")
        return combined_df
    else:
        print("❌ No data was successfully downloaded.")
        return None

def parse_date_range(data_from, data_to):
    """
    Parse date range configuration and return year and months list.
    
    Args:
        data_from (str): Start date in format "YYYY MM" (e.g., "2021 01")
        data_to (str): End date in format "YYYY MM" (e.g., "2021 1")
    
    Returns:
        tuple: (year, months_list)
    """
    # Parse start date
    start_parts = data_from.strip().split()
    start_year = start_parts[0]
    start_month = int(start_parts[1])
    
    # Parse end date
    end_parts = data_to.strip().split()
    end_year = end_parts[0]
    end_month = int(end_parts[1])
    
    # Validate year consistency
    if start_year != end_year:
        print(f"⚠️  Warning: Different years in date range ({start_year} to {end_year})")
        print(f"   Using start year: {start_year}")
    
    # Generate months list
    months = []
    for month in range(start_month, end_month + 1):
        months.append(f"{month:02d}")
    
    print(f"📅 Date range: {start_year} months {start_month} to {end_month}")
    print(f"📊 Will download months: {months}")
    
    return start_year, months

def download_crypto_data(symbol, interval, data_from, data_to, max_rows=50000):
    """
    Common function to download cryptocurrency data using configuration variables.
    Uses Binance Vision only for reliable data downloading.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Time interval (e.g., '5m', '1h', '1d')
        data_from (str): Start date in format "YYYY MM" (e.g., "2021 01")
        data_to (str): End date in format "YYYY MM" (e.g., "2021 1")
        max_rows (int): Maximum rows to download (default: 50000)
    
    Returns:
        pd.DataFrame: Combined data or None if download fails
    """
    print(f"📥 DOWNLOADING CRYPTOCURRENCY DATA")
    print(f"   Symbol: {symbol}")
    print(f"   Interval: {interval}")
    print(f"   Date range: {data_from} to {data_to}")
    print(f"   Max rows: {max_rows:,}")
    print("=" * 50)
    
    # Parse date range
    year, months = parse_date_range(data_from, data_to)
    
    # Download data
    data = load_multiple_months_data(symbol, interval, year, months, max_rows)
    
    if data is not None:
        print(f"\\n✅ Download completed successfully!")
        print(f"📊 Final data shape: {data.shape}")
        print(f"📅 Date range: {data['Open time'].min()} to {data['Open time'].max()}")
        print(f"💾 Memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    else:
        print(f"\\n❌ Download failed!")
    
    return data
