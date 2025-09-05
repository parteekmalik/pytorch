# Cryptocurrency Prediction System

A streamlined LSTM-based cryptocurrency prediction system using Binance data. Features simplified OHLCV data structure, dynamic model output, and automatic data download.

## Quick Start

```python
from src import BinanceDataOrganizer, DataConfig, create_lstm_model, evaluate_model

# Configure data
config = DataConfig(
    symbol='BTCUSDT',
    timeframe='5m',
    start_time='2021-01',
    end_time='2021-01',
    sequence_length=5,
    prediction_length=1,
    train_split=0.8
)

# Create organizer (auto-downloads data)
organizer = BinanceDataOrganizer(config)

# Process data
if organizer.process_all():
    # Get scaled data
    data = organizer.get_scaled_data('all')

    # Create model
    model = create_lstm_model(
        input_shape=(data['X_train_scaled'].shape[1], data['X_train_scaled'].shape[2]),
        prediction_length=config.prediction_length
    )

    # Train model
    history = model.fit(
        data['X_train_scaled'], data['y_train_scaled'],
        validation_data=(data['X_test_scaled'], data['y_test_scaled']),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
        ]
    )

    # Evaluate model
    scalers = organizer.get_scalers()
    results = evaluate_model(model, data['X_test_scaled'], data['y_test_scaled'], scalers['y'])
```

## Key Features

- **Simplified Data**: OHLCV-only data structure
- **Dynamic Output**: Models adapt to prediction length
- **Auto-Download**: Data downloaded automatically
- **Early Stopping**: Prevents overfitting
- **Multi-Step Prediction**: Predict multiple future candles

## Multi-Step Prediction

```python
# Configure for 5-step prediction
config = DataConfig(
    symbol='BTCUSDT',
    timeframe='5m',
    start_time='2021-01',
    end_time='2021-01',
    sequence_length=10,
    prediction_length=5  # Predict 5 future candles
)

organizer = BinanceDataOrganizer(config)
# Model automatically adjusts output size to 25 (5 candles × 5 OHLCV values)
```

## File Structure

```
src/
├── binance_data_organizer.py  # Main data management class
├── model_utils.py            # LSTM model functions
├── utils.py                  # Data download and sequence creation
└── __init__.py              # Package imports

tests/
├── test_binance_organizer.py # Comprehensive organizer tests
├── test_model_utils.py       # Model function tests
└── test_utils.py             # Utility function tests

docs/
├── README.md                 # This file
└── MODEL_UTILITIES.md        # Model training documentation

scripts/
└── run_notebook.py           # Notebook execution script
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test notebook execution
python scripts/run_notebook.py crypto_prediction.ipynb
```

## Documentation

- **[Model Utilities](MODEL_UTILITIES.md)** - LSTM model creation and training

## Troubleshooting

### Common Issues

1. **Import errors**: Use `from src import` instead of `from utils import`
2. **Dimension errors**: Ensure prediction_length matches model output
3. **Download failures**: Check internet connection and date format (yyyy-mm)
4. **Poor performance**: Tune hyperparameters or increase sequence_length
5. **Overfitting**: Early stopping should handle this automatically

### Debug Tools

```python
# Test imports
python -c "from src import BinanceDataOrganizer, DataConfig; print('✅ Imports working')"
```
