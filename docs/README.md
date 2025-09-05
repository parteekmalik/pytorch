# Cryptocurrency Prediction Documentation

## Overview
This documentation covers the cryptocurrency prediction system built with LSTM models and Binance data. The system is designed for memory efficiency and provides comprehensive data management, feature engineering, and model training capabilities.

## Quick Start

```python
from utils import BinanceDataOrganizer, DataConfig, create_lstm_model, train_model_memory_efficient

# Configure data
config = DataConfig('BTCUSDT', '5m', '2021-01-01', '2021-01-31', 10, 1)
organizer = BinanceDataOrganizer(config)

# Process data
organizer.process_all()
data = organizer.get_scaled_data('all')

# Create and train model
model = create_lstm_model((data['X_train_scaled'].shape[1], data['X_train_scaled'].shape[2]), data['y_train_scaled'].shape[1])
history = train_model_memory_efficient(model, data['X_train_scaled'], data['y_train_scaled'], data['X_test_scaled'], data['y_test_scaled'])
```

## Documentation Structure

### Core Components
- **[Binance Data Organizer](BINANCE_DATA_ORGANIZER.md)** - Main data management class
- **[Grouped Scaler](GROUPED_SCALER.md)** - Intelligent normalization system
- **[Data Utilities](DATA_UTILITIES.md)** - Data download and processing functions
- **[Model Utilities](MODEL_UTILITIES.md)** - LSTM model creation and training

### Key Features
- **Memory Efficient**: Designed for 16GB M4 MacBook
- **On-Demand Data**: Generate data for specific time ranges
- **Grouped Normalization**: Intelligent feature scaling
- **Comprehensive Features**: Time, price, and volume features
- **Production Ready**: Save/load models and scalers

## File Structure

```
utils/
├── binance_data_organizer.py  # Main data management
├── model_utils.py            # LSTM model functions
├── memory_utils.py           # Memory management
└── __init__.py              # Package imports

docs/
├── README.md                 # This file
├── BINANCE_DATA_ORGANIZER.md # Data organizer documentation
├── GROUPED_SCALER.md         # Normalization documentation
├── DATA_UTILITIES.md         # Data processing documentation
└── MODEL_UTILITIES.md        # Model training documentation
```

## Architecture

### Data Flow
1. **Download**: Binance Vision API → Raw OHLCV data
2. **Features**: Time, price, volume features
3. **Sequences**: Sliding windows for LSTM
4. **Normalization**: Grouped scaling by feature type
5. **Training**: Memory-efficient LSTM training
6. **Prediction**: Real-time candle prediction

### Memory Management
- Configurable row limits
- Lazy data loading
- Garbage collection
- Memory monitoring

## Examples

### Basic Usage
```python
from utils import BinanceDataOrganizer, DataConfig

config = DataConfig('BTCUSDT', '5m', '2021-01-01', '2021-01-31', 10, 1)
organizer = BinanceDataOrganizer(config)
organizer.process_all()
```

### Advanced Usage
```python
# Get data for specific time range
data = organizer.get_data_in_range('2021-01-15', '2021-01-20', scaled=True)

# Make predictions
predictions = model.predict(data['X_scaled'])
```

### Production Deployment
```python
# Save model and scalers
model.save('crypto_model.h5')
import pickle
with open('scalers.pkl', 'wb') as f:
    pickle.dump(organizer.get_scalers(), f)
```

## Performance

### Memory Usage
- **Conservative**: 10,000 rows (~2GB)
- **Moderate**: 50,000 rows (~8GB)
- **Aggressive**: 100,000+ rows (16GB+)

### Training Time
- **Small dataset**: 5-10 minutes
- **Medium dataset**: 15-30 minutes
- **Large dataset**: 1+ hours

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce `max_rows`
2. **Download failures**: Check internet connection
3. **Poor performance**: Tune hyperparameters
4. **Overfitting**: Increase dropout rate

### Debug Tools
```python
from utils import get_memory_usage, print_memory_stats

# Monitor memory
print_memory_stats()

# Check memory usage
usage = get_memory_usage()
print(f"Memory usage: {usage:.1f} MB")
```

## Contributing

### Code Style
- Minimal comments in code
- Comprehensive documentation in .md files
- Type hints for all functions
- Clean, readable code structure

### Testing
```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test
python tests/test_data_utilities.py
```

## License
This project is for educational and research purposes.
