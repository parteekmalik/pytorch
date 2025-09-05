# Binance Data Organizer

## Overview
The `BinanceDataOrganizer` class provides a unified interface for managing Binance cryptocurrency data, including loading, feature engineering, normalization, and on-demand data generation for LSTM training.

## Core Components

### DataConfig
Configuration class for data loading and processing parameters.

```python
@dataclass
class DataConfig:
    symbol: str                    # Trading pair (e.g., 'BTCUSDT')
    timeframe: str                 # Time interval (e.g., '5m', '1h')
    start_time: str               # Start date (YYYY-MM-DD format)
    end_time: str                 # End date (YYYY-MM-DD format)
    sequence_length: int          # LSTM input sequence length
    prediction_length: int        # Number of future candles to predict
    max_rows: int = 50000         # Maximum rows to load (memory limit)
    train_split: float = 0.8      # Train/test split ratio
```

### BinanceDataOrganizer
Main class for data management.

#### Key Methods

**Data Loading:**
- `load_data()` - Downloads raw data from Binance Vision
- `create_features()` - Creates time series features
- `process_all()` - Complete pipeline (load + features)

**Data Access:**
- `get_unscaled_data(data_type)` - Returns raw sequences
- `get_scaled_data(data_type)` - Returns normalized sequences
- `get_data_in_range(start, end, scaled)` - Get data for specific time range

**Utilities:**
- `get_scalers()` - Access fitted scalers
- `get_feature_info()` - Get feature information

#### Usage Example

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

# Create organizer
organizer = BinanceDataOrganizer(config)

# Process data
if organizer.process_all():
    # Get scaled training data
    train_data = organizer.get_scaled_data('train')
    X_train = train_data['X_train_scaled']
    y_train = train_data['y_train_scaled']
    
    # Get scaled test data
    test_data = organizer.get_scaled_data('test')
    X_test = test_data['X_test_scaled']
    y_test = test_data['y_test_scaled']
```

## Data Processing Pipeline

### 1. Data Download
- Downloads from Binance Vision API
- Supports multiple months automatically
- Memory-efficient loading with row limits
- Handles date range parsing

### 2. Feature Engineering
Creates comprehensive time series features:

**Time Features:**
- `Minutes_of_day` - Minute of day (0-1439)

**Price Features:**
- `Price_Range` - High - Low
- `Price_Change` - Close - Open
- `Price_Change_Pct` - Percentage change

**Volume Features:**
- `Volume_MA_5` - 5-period volume moving average
- `Volume_MA_10` - 10-period volume moving average

### 3. Sequence Creation
- Creates sliding windows for LSTM training
- Configurable sequence and prediction lengths
- Handles missing data automatically

### 4. Normalization
Uses `GroupedScaler` for intelligent feature scaling:

**Feature Groups:**
- **Price Group**: Open, High, Low, Close, Price_Range, Price_Change, Price_Change_Pct
- **Volume Group**: Volume, Volume_MA_5, Volume_MA_10, Quote asset volume, etc.
- **Time Group**: Minutes_of_day (fixed range 0-1439)
- **Other Group**: Any remaining features

## Advanced Features

### On-Demand Data Generation
Generate data for specific time ranges without reprocessing:

```python
# Get data for specific time range
data = organizer.get_data_in_range(
    start_time='2021-01-15',
    end_time='2021-01-20',
    scaled=True
)

if data:
    X_scaled = data['X_scaled']
    y_scaled = data['y_scaled']
```

### Memory Management
- Configurable row limits to prevent memory issues
- Lazy loading of data
- Efficient data structures

### Error Handling
- Graceful handling of download failures
- Data validation and cleaning
- Informative error messages

## Performance Considerations

### Memory Usage
- Set appropriate `max_rows` based on available memory
- Use `get_data_in_range()` for specific time periods
- Monitor memory usage with utility functions

### Data Quality
- Automatic handling of missing data
- Data validation during download
- Feature engineering optimizations

### Scalability
- Efficient data structures
- Optimized algorithms
- Configurable parameters

## Integration with LSTM Models

The organizer is designed to work seamlessly with LSTM models:

```python
# Get data for training
data = organizer.get_scaled_data('all')
X_train = data['X_train_scaled']
y_train = data['y_train_scaled']
X_test = data['X_test_scaled']
y_test = data['y_test_scaled']

# Train LSTM model
model = create_lstm_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    lstm_units=50,
    dropout_rate=0.2
)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions
scaler_y = organizer.get_scalers()['y']
predictions_original = scaler_y.inverse_transform(predictions)
```

## Troubleshooting

### Common Issues

**Memory Errors:**
- Reduce `max_rows` parameter
- Use `get_data_in_range()` for smaller datasets
- Monitor memory usage

**Download Failures:**
- Check internet connection
- Verify symbol and timeframe validity
- Check date range format

**Data Quality Issues:**
- Ensure sufficient data for sequences
- Check for missing values
- Validate feature engineering

### Debug Information
The organizer provides detailed logging for troubleshooting:
- Download progress
- Feature creation status
- Scaling information
- Memory usage statistics
