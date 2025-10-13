"""
Data downloading and caching module for cryptocurrency data from Binance.
"""
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
from datetime import datetime
from typing import Optional
from .utils import setup_logger

logger = setup_logger(__name__)


def create_price_sequences(price_data: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Create sliding window sequences from price data.
    
    Args:
        price_data: Array of price values
        seq_len: Length of each sequence
        
    Returns:
        Array of sequences with shape (n_sequences, seq_len)
    """
    sequences = []
    for i in range(len(price_data) - seq_len + 1):
        sequence = price_data[i:i + seq_len]
        sequences.append(sequence)
    return np.array(sequences)


def _download_single_month(
    symbol: str, 
    interval: str, 
    year: str, 
    month: str,
    cache_dir: str
) -> Optional[pd.DataFrame]:
    """
    Download data for a single month from Binance with caching.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1m', '5m', '1h')
        year: Year as string
        month: Month as string (will be zero-padded)
        cache_dir: Directory to cache downloaded ZIP files
        
    Returns:
        DataFrame with OHLCV data or None if download fails
    """
    month_padded = month.zfill(2)
    
    os.makedirs(cache_dir, exist_ok=True)
    
    filename = f"{symbol}-{interval}-{year}-{month_padded}.zip"
    zip_path = os.path.join(cache_dir, filename)
    
    try:
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{filename}"
        
        if os.path.exists(zip_path):
            logger.info(f"Using cached file: {filename}")
            with open(zip_path, 'rb') as f:
                zip_content = f.read()
        else:
            logger.info(f"Downloading: {filename}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            zip_content = response.content
            
            with open(zip_path, 'wb') as f:
                f.write(zip_content)
        
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                return None
            
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, header=None)
        
        df.columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]
        
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        
        numeric_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 
            'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) > 0:
            return df
        else:
            return None
            
    except Exception as e:
        logger.warning(f"Error downloading {year}-{month_padded}: {e}")
        return None


def download_crypto_data(
    symbol: str,
    interval: str,
    start_date_str: str,
    end_date_str: str,
    cache_dir: str,
    columns: str = 'Close'
) -> pd.DataFrame:
    """
    Download cryptocurrency data from Binance for a date range.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1m', '5m', '1h')
        start_date_str: Start date in 'YYYY-MM' format
        end_date_str: End date in 'YYYY-MM' format
        cache_dir: Directory to cache downloaded files
        columns: Which columns to return using character notation (default: 'C'):
                Price: O(Open), H(High), L(Low), C(Close)
                Volume: V(Volume), Q(Quote volume)
                Trading: N(Number of trades), B(Taker buy base vol), R(Taker buy quote vol)
                Time: T(Open time), E(Close time)
                Common: OHLC (prices), OHLCV (prices+volume)
                Any combination: 'OC', 'HLV', 'OHLCVN', etc.
                Case-insensitive, separators (spaces/commas) ignored
        
    Returns:
        DataFrame with selected price columns
        
    Raises:
        ValueError: If no data could be downloaded or invalid columns specified
    """
    logger.info(f"Downloading {symbol} data from {start_date_str} to {end_date_str}")
    
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m')
        end_date = datetime.strptime(end_date_str, '%Y-%m')
        
        all_data = []
        current = start_date.replace(day=1)
        end = end_date.replace(day=1)
        
        while current <= end:
            year = current.strftime('%Y')
            month = current.strftime('%m')
            df = _download_single_month(symbol, interval, year, month, cache_dir)
            if df is not None:
                all_data.append(df)
            
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        if not all_data:
            raise ValueError("No data downloaded")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Handle column selection using character notation
        char_to_col = {
            # Price columns
            'O': 'Open',
            'H': 'High',
            'L': 'Low',
            'C': 'Close',
            # Volume columns
            'V': 'Volume',
            'Q': 'Quote asset volume',
            'N': 'Number of trades',
            'B': 'Taker buy base asset volume',
            'R': 'Taker buy quote asset volume',
            # Time columns
            'T': 'Open time',
            'E': 'Close time'
        }

        # Parse the columns string
        selected_cols = []
        for char in columns.upper():
            if char in char_to_col:
                col_name = char_to_col[char]
                if col_name not in selected_cols:  # Avoid duplicates
                    selected_cols.append(col_name)
            elif char not in [' ', ',', '-', '_']:  # Ignore separators
                raise ValueError(
                    f"Invalid column character '{char}'. "
                    f"Valid: O(Open), H(High), L(Low), C(Close), V(Volume), "
                    f"Q(Quote volume), N(Trades), B(Buy base vol), R(Buy quote vol), "
                    f"T(Open time), E(Close time)"
                )

        if not selected_cols:
            raise ValueError("No valid columns specified. Use: C, OHLC, OHLCV, etc.")

        # Maintain standard column order
        standard_order = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume'
        ]
        selected_cols = [col for col in standard_order if col in selected_cols]

        # Select the requested columns
        if len(selected_cols) == 1:
            result = combined_df[selected_cols[0]]
        else:
            result = combined_df[selected_cols]

        logger.info(f"Downloaded {len(result)} data points with columns: {selected_cols}")
        return result
        
    except Exception as e:
        raise ValueError(f"Download failed: {e}")


