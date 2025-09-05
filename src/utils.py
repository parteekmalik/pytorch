import pandas as pd
import numpy as np
import requests
import zipfile
import io
from datetime import datetime
from typing import Optional, List, Tuple

def _download_single_month(symbol: str, interval: str, year: str, month: str) -> Optional[pd.DataFrame]:
    """Download and process data for a single month from Binance Vision."""
    try:
        month_padded = month.zfill(2)
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month_padded}.zip"
        
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                return None
            
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, header=None)
        
        # Set column names
        df.columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]
        
        # Convert timestamps
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        
        # Convert numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 
                       'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) > 0:
            return df
        else:
            return None
            
    except Exception as e:
        raise ValueError(f"Error downloading {year}-{month_padded}: {e}")



def download_binance_data(symbol: str, interval: str, data_from: str, data_to: str) -> Optional[pd.DataFrame]:
    
    try:
        # Parse date range (format: yyyy-mm)
        start_date = datetime.strptime(data_from, '%Y-%m')
        end_date = datetime.strptime(data_to, '%Y-%m')
        
        # Generate months to download
        months = []
        current = start_date.replace(day=1)
        end = end_date.replace(day=1)
        
        while current <= end:
            months.append(current.strftime('%m'))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        year = start_date.strftime('%Y')
        
        # Download data from Binance Vision
        all_data = []
        
        for month in months:
            df = _download_single_month(symbol, interval, year, month)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No data downloaded")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Apply essential columns logic here - keep only OHLCV data
        essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        combined_df = combined_df[essential_cols]
        
        
        return combined_df
        
    except Exception as e:
        raise ValueError(f"Download failed: {e}")





def create_sequences(data: pd.DataFrame, sequence_length: int, prediction_length: int, target_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create sequences from OHLCV data for time series prediction.
    
    Args:
        data: DataFrame with OHLCV columns
        sequence_length: Number of timesteps to look back
        prediction_length: Number of future timesteps to predict
        target_cols: Columns to use as targets (default: OHLCV)
    
    Returns:
        X: Input sequences (samples, timesteps, features)
        y: Target sequences (samples, prediction_values)
        feature_cols: List of feature column names
    """
    if target_cols is None:
        target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Validate input data
    if data is None or data.empty:
        raise ValueError("Input data is empty or None")
    
    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Feature columns are OHLCV (same as targets for this use case)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Clean data - remove any rows with NaN values
    data_clean = data.dropna()
    
    # Validate data length
    min_required = sequence_length + prediction_length
    if len(data_clean) < min_required:
        raise ValueError(f"Not enough data after cleaning. Need at least {min_required} rows, got {len(data_clean)}")
    
    # Convert to numpy for better performance
    data_values = data_clean[feature_cols].values
    
    # Calculate number of sequences
    num_sequences = len(data_clean) - sequence_length - prediction_length + 1
    
    # Pre-allocate arrays for better performance
    X = np.zeros((num_sequences, sequence_length, len(feature_cols)))
    y = np.zeros((num_sequences, prediction_length * len(target_cols)))
    
    # Create sequences
    for i in range(num_sequences):
        # Input sequence: [i:i+sequence_length]
        X[i] = data_values[i:i+sequence_length]
        
        # Target sequence: [i+sequence_length:i+sequence_length+prediction_length]
        target_start = i + sequence_length
        target_end = target_start + prediction_length
        target_values = data_values[target_start:target_end]
        
        # Get only the target columns from the data
        target_indices = [feature_cols.index(col) for col in target_cols if col in feature_cols]
        target_subset = target_values[:, target_indices]
        
        # Flatten target values for prediction
        y[i] = target_subset.flatten()
    
    return X, y, feature_cols
