# Data Utilities

## Overview

The data utilities module provides functions for downloading, processing, and preparing cryptocurrency data for machine learning models.

## Core Functions

### Data Download

#### `download_binance_data(symbol, interval, data_from, data_to, max_rows)`

Downloads cryptocurrency data from Binance Vision API.

**Parameters:**

- `symbol`: Trading pair (e.g., 'BTCUSDT')
- `interval`: Time interval (e.g., '5m', '1h', '1d')
- `data_from`: Start date (YYYY-MM-DD or YYYY MM format)
- `data_to`: End date (YYYY-MM-DD or YYYY MM format)
- `max_rows`: Maximum rows to download (memory limit)

**Returns:** DataFrame with OHLCV data or None if failed

**Example:**

```python
from utils import download_binance_data

# Download BTC data for January 2021
data = download_binance_data(
    symbol='BTCUSDT',
    interval='5m',
    data_from='2021-01-01',
    data_to='2021-01-31',
    max_rows=10000
)
```

#### `download_crypto_data(symbol, interval, data_from, data_to, max_rows)`

Legacy function - same as `download_binance_data()`.

### Feature Engineering

#### `create_features(df)`

Creates comprehensive time series features from raw OHLCV data.

**Parameters:**

- `df`: DataFrame with OHLCV data

**Returns:** DataFrame with additional features

**Features Created:**

- `Minutes_of_day`: Minute of day (0-1439)
- `Price_Range`: High - Low
- `Price_Change`: Close - Open
- `Price_Change_Pct`: Percentage price change
- `Volume_MA_5`: 5-period volume moving average
- `Volume_MA_10`: 10-period volume moving average

**Example:**

```python
from utils import create_features

# Create features from raw data
features_df = create_features(raw_data)
print(f"Features created: {features_df.shape}")
```

#### `create_minimal_features(df, lag_period)`

Legacy function - same as `create_features()`.

### Sequence Creation

#### `create_sequences(data, sequence_length, prediction_length, target_cols)`

Creates sliding windows for time series data.

**Parameters:**

- `data`: DataFrame with features
- `sequence_length`: Number of timesteps for input
- `prediction_length`: Number of future timesteps to predict
- `target_cols`: List of target columns (default: OHLCV)

**Returns:** Tuple of (X, y, feature_columns)

**Example:**

```python
from utils import create_sequences

# Create sequences for LSTM training
X, y, feature_cols = create_sequences(
    data=features_df,
    sequence_length=10,
    prediction_length=1
)

print(f"X shape: {X.shape}")  # (samples, timesteps, features)
print(f"y shape: {y.shape}")  # (samples, targets)
```

#### `create_sliding_windows(data, sequence_length, target_cols)`

Legacy function - same as `create_sequences()` with prediction_length=1.

### Data Scaling

#### `scale_data(X_train, X_test, y_train, y_test, feature_cols)`

Scales time series data using grouped normalization.

**Parameters:**

- `X_train`: Training input sequences
- `X_test`: Test input sequences
- `y_train`: Training target sequences
- `y_test`: Test target sequences
- `feature_cols`: List of feature column names

**Returns:** Tuple of scaled data and scalers

**Example:**

```python
from utils import scale_data

# Scale data for training
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = scale_data(
    X_train, X_test, y_train, y_test, feature_cols
)
```

#### `scale_time_series_data_grouped(X_train, X_test, y_train, y_test, feature_cols)`

Legacy function - same as `scale_data()`.

## Data Processing Pipeline

### Complete Workflow

```python
from utils import (
    download_binance_data,
    create_features,
    create_sequences,
    scale_data
)

# 1. Download data
raw_data = download_binance_data(
    symbol='BTCUSDT',
    interval='5m',
    data_from='2021-01-01',
    data_to='2021-01-31',
    max_rows=10000
)

# 2. Create features
features_df = create_features(raw_data)

# 3. Create sequences
X, y, feature_cols = create_sequences(
    data=features_df,
    sequence_length=10,
    prediction_length=1
)

# 4. Split data
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 5. Scale data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = scale_data(
    X_train, X_test, y_train, y_test, feature_cols
)
```

### Using BinanceDataOrganizer (Recommended)

```python
from utils import BinanceDataOrganizer, DataConfig

# Configure data parameters
config = DataConfig(
    symbol='BTCUSDT',
    timeframe='5m',
    start_time='2021-01-01',
    end_time='2021-01-31',
    sequence_length=10,
    prediction_length=1,
    max_rows=10000,
    train_split=0.8
)

# Create organizer and process data
organizer = BinanceDataOrganizer(config)
if organizer.process_all():
    # Get scaled data
    data = organizer.get_scaled_data('all')
    X_train = data['X_train_scaled']
    y_train = data['y_train_scaled']
    X_test = data['X_test_scaled']
    y_test = data['y_test_scaled']
```

## Data Formats

### Input Data Format

Raw data from Binance Vision includes:

| Column                       | Description                  |
| ---------------------------- | ---------------------------- |
| Open time                    | Opening timestamp            |
| Open                         | Opening price                |
| High                         | Highest price                |
| Low                          | Lowest price                 |
| Close                        | Closing price                |
| Volume                       | Trading volume               |
| Close time                   | Closing timestamp            |
| Quote asset volume           | Quote asset volume           |
| Number of trades             | Number of trades             |
| Taker buy base asset volume  | Taker buy base asset volume  |
| Taker buy quote asset volume | Taker buy quote asset volume |

### Feature Data Format

After feature engineering:

| Column           | Type   | Description            |
| ---------------- | ------ | ---------------------- |
| Minutes_of_day   | Time   | Minute of day (0-1439) |
| Price_Range      | Price  | High - Low             |
| Price_Change     | Price  | Close - Open           |
| Price_Change_Pct | Price  | Percentage change      |
| Volume_MA_5      | Volume | 5-period volume MA     |
| Volume_MA_10     | Volume | 10-period volume MA    |

### Sequence Data Format

For LSTM training:

- **X**: (samples, timesteps, features) - Input sequences
- **y**: (samples, targets) - Target values

## Memory Management

### Row Limits

Set appropriate `max_rows` based on available memory:

```python
# Conservative (2GB RAM)
max_rows = 10000

# Moderate (8GB RAM)
max_rows = 50000

# Aggressive (16GB+ RAM)
max_rows = 100000
```

### Memory Monitoring

Use memory utilities to monitor usage:

```python
from utils import get_memory_usage, print_memory_stats

# Check memory usage
memory_usage = get_memory_usage()
print(f"Memory usage: {memory_usage:.2f}%")

# Print detailed stats
print_memory_stats()
```

## Error Handling

### Common Errors

**Download Failures:**

```python
# Check internet connection and symbol validity
data = download_binance_data('INVALID', '5m', '2021-01-01', '2021-01-01')
if data is None:
    print("Download failed - check symbol and connection")
```

**Insufficient Data:**

```python
# Ensure enough data for sequences
if len(data) < sequence_length:
    raise ValueError(f"Need at least {sequence_length} rows")
```

**Memory Errors:**

```python
# Reduce max_rows if memory error occurs
try:
    data = download_binance_data(..., max_rows=100000)
except MemoryError:
    data = download_binance_data(..., max_rows=10000)
```

## Performance Tips

### Optimization Strategies

1. **Use Appropriate Timeframes**: Higher timeframes = less data
2. **Limit Date Ranges**: Shorter periods = faster processing
3. **Set Row Limits**: Prevent memory overflow
4. **Use BinanceDataOrganizer**: Optimized for common workflows
5. **Monitor Memory**: Use memory utilities

### Benchmarking

```python
import time

# Time data download
start = time.time()
data = download_binance_data('BTCUSDT', '5m', '2021-01-01', '2021-01-31', 10000)
download_time = time.time() - start

# Time feature creation
start = time.time()
features = create_features(data)
feature_time = time.time() - start

print(f"Download: {download_time:.2f}s")
print(f"Features: {feature_time:.2f}s")
```

## Integration Examples

### With LSTM Models

```python
from utils import BinanceDataOrganizer, DataConfig
from utils import create_lstm_model, train_model_memory_efficient

# Get data
config = DataConfig('BTCUSDT', '5m', '2021-01-01', '2021-01-31', 10, 1)
organizer = BinanceDataOrganizer(config)
organizer.process_all()

data = organizer.get_scaled_data('all')
X_train = data['X_train_scaled']
y_train = data['y_train_scaled']

# Create and train model
model = create_lstm_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    lstm_units=50,
    dropout_rate=0.2
)

history = train_model_memory_efficient(
    model, X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)
```

### With Production Systems

```python
# Save processed data
import pickle

# Save organizer with fitted scalers
with open('organizer.pkl', 'wb') as f:
    pickle.dump(organizer, f)

# Load in production
with open('organizer.pkl', 'rb') as f:
    organizer = pickle.load(f)

# Make predictions
new_data = organizer.get_data_in_range('2021-02-01', '2021-02-01', scaled=True)
if new_data:
    predictions = model.predict(new_data['X_scaled'])
```
