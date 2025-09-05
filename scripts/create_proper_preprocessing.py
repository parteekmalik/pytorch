#!/usr/bin/env python3
"""
Proper preprocessing functions for time series prediction with sliding windows.
This creates the correct data structure for training LSTM to predict from last n candles.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_time_series_features(df, lag_period=5):
    """
    Create time series features for each candle.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data
        lag_period (int): Number of previous candles to use for lag features
    
    Returns:
        pd.DataFrame: DataFrame with time series features
    """
    print("🔧 Creating time series features...")
    data = df.copy()
    
    # Time-based features for 5-minute intervals
    # 5-minute intervals = 288 intervals per day (24 * 60 / 5)
    data['Hour'] = data['Open time'].dt.hour
    data['Minute'] = data['Open time'].dt.minute
    
    # Calculate the 5-minute interval number within the day (0-287)
    data['Interval_5min'] = data['Hour'] * 12 + data['Minute'] // 5
    
    # Cyclical encoding for 5-minute intervals
    data['Interval_sin'] = np.sin(2 * np.pi * data['Interval_5min'] / 288)
    data['Interval_cos'] = np.cos(2 * np.pi * data['Interval_5min'] / 288)
    
    # Also keep hour-based features for broader patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    
    # Price-based features
    data['Price_Range'] = data['High'] - data['Low']
    data['Price_Change'] = data['Close'] - data['Open']
    data['Price_Change_Pct'] = (data['Close'] - data['Open']) / data['Open']
    
    # Volume features
    data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
    data['Volume_MA_10'] = data['Volume'].rolling(window=10).mean()
    
    # Lag features for OHLCV
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ohlcv_cols:
        for lag in range(1, lag_period + 1):
            data[f'{col}_Lag_{lag}'] = data[col].shift(lag)
    
    print(f"✅ Time series features created! Shape: {data.shape}")
    
    # Show time feature examples
    print(f"\n⏰ Time Feature Examples:")
    print(f"   Hour range: {data['Hour'].min()}-{data['Hour'].max()}")
    print(f"   5-min interval range: {data['Interval_5min'].min()}-{data['Interval_5min'].max()}")
    print(f"   Interval_sin range: {data['Interval_sin'].min():.3f} to {data['Interval_sin'].max():.3f}")
    print(f"   Interval_cos range: {data['Interval_cos'].min():.3f} to {data['Interval_cos'].max():.3f}")
    
    return data

def create_sliding_windows(data, sequence_length=10, target_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
    """
    Create sliding windows for time series prediction.
    
    Args:
        data (pd.DataFrame): DataFrame with features
        sequence_length (int): Number of previous candles to use for prediction
        target_cols (list): Columns to predict
    
    Returns:
        tuple: (X, y) - Features and targets for LSTM training
    """
    print(f"🔄 Creating sliding windows (sequence length: {sequence_length})...")
    
    # Select feature columns (exclude time and target columns)
    feature_cols = [col for col in data.columns if col not in ['Open time', 'Close time'] + target_cols]
    
    # Remove rows with NaN values
    data_clean = data.dropna()
    
    X, y = [], []
    
    for i in range(sequence_length, len(data_clean)):
        # Input: sequence_length previous candles
        X.append(data_clean[feature_cols].iloc[i-sequence_length:i].values)
        
        # Target: next candle's OHLCV values
        y.append(data_clean[target_cols].iloc[i].values)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Sliding windows created!")
    print(f"📊 X shape: {X.shape} (samples, sequence_length, features)")
    print(f"📊 y shape: {y.shape} (samples, targets)")
    
    return X, y, feature_cols

def preprocess_for_lstm_prediction(df, sequence_length=10, test_size=0.2, lag_period=5):
    """
    Complete preprocessing pipeline for LSTM time series prediction.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data
        sequence_length (int): Number of previous candles to use for prediction
        test_size (float): Proportion of data to use for testing
        lag_period (int): Number of lag features to create
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_cols, scaler_X, scaler_y)
    """
    print("🚀 Starting LSTM prediction preprocessing...")
    
    # Step 1: Create time series features
    data_with_features = create_time_series_features(df, lag_period)
    
    # Step 2: Create sliding windows
    X, y, feature_cols = create_sliding_windows(data_with_features, sequence_length)
    
    # Step 3: Split data
    split_index = int(len(X) * (1 - test_size))
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    print(f"✂️ Data split completed!")
    print(f"📊 Training samples: {X_train.shape[0]}")
    print(f"📊 Test samples: {X_test.shape[0]}")
    print(f"📊 Sequence length: {X_train.shape[1]}")
    print(f"📊 Features per timestep: {X_train.shape[2]}")
    print(f"📊 Target variables: {y_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def scale_time_series_data(X_train, X_test, y_train, y_test, feature_cols):
    """
    Scale the time series data for LSTM training.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        feature_cols: List of feature column names
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y)
    """
    print("🔢 Scaling time series data...")
    
    # Reshape for scaling (samples * timesteps, features)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    # Scale features
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
    X_test_scaled = scaler_X.transform(X_test_reshaped)
    
    # Reshape back to (samples, timesteps, features)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # Scale targets
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    print("✅ Scaling completed!")
    print(f"📊 Scaled X_train shape: {X_train_scaled.shape}")
    print(f"📊 Scaled y_train shape: {y_train_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

def demonstrate_data_structure(X, y, feature_cols, target_cols=['Open', 'High', 'Low', 'Close', 'Volume'], n_samples=3):
    """
    Demonstrate the data structure for understanding.
    
    Args:
        X: Feature data
        y: Target data
        feature_cols: Feature column names
        target_cols: Target column names
        n_samples: Number of samples to show
    """
    print("\n📊 DATA STRUCTURE DEMONSTRATION:")
    print("=" * 60)
    
    for i in range(min(n_samples, len(X))):
        print(f"\n🔍 Sample {i+1}:")
        print(f"   Input: {X.shape[1]} previous candles → Predict next candle")
        print(f"   Features per candle: {X.shape[2]}")
        print(f"   Target: {len(target_cols)} OHLCV values")
        
        # Show the last candle's features
        last_candle_features = X[i, -1, :]  # Last timestep of this sample
        print(f"\n   Last candle features (scaled):")
        for j, feature in enumerate(feature_cols[:10]):  # Show first 10 features
            print(f"     {feature}: {last_candle_features[j]:.4f}")
        if len(feature_cols) > 10:
            print(f"     ... and {len(feature_cols) - 10} more features")
        
        # Show target
        print(f"\n   Target (next candle OHLCV):")
        for j, target in enumerate(target_cols):
            print(f"     {target}: {y[i, j]:.4f}")
    
    print(f"\n📈 This means:")
    print(f"   - Each training sample uses {X.shape[1]} consecutive candles")
    print(f"   - Each candle has {X.shape[2]} features")
    print(f"   - Model predicts the next candle's {len(target_cols)} values")
    print(f"   - Total training samples: {X.shape[0]}")

if __name__ == "__main__":
    # Test the functions
    print("Testing proper preprocessing functions...")
    
    # Create sample data
    dates = pd.date_range('2021-01-01', periods=100, freq='5min')
    sample_data = pd.DataFrame({
        'Open time': dates,
        'Open': np.random.uniform(30000, 50000, 100),
        'High': np.random.uniform(30000, 50000, 100),
        'Low': np.random.uniform(30000, 50000, 100),
        'Close': np.random.uniform(30000, 50000, 100),
        'Volume': np.random.uniform(100, 1000, 100)
    })
    
    # Test preprocessing
    X_train, X_test, y_train, y_test, feature_cols = preprocess_for_lstm_prediction(
        sample_data, sequence_length=10, test_size=0.2
    )
    
    # Test scaling
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = scale_time_series_data(
        X_train, X_test, y_train, y_test, feature_cols
    )
    
    # Demonstrate structure
    demonstrate_data_structure(X_train_scaled, y_train_scaled, feature_cols)
    
    print("\n🎉 All preprocessing functions working correctly!")
