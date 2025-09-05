# Model Utilities

## Overview

Functions for creating, training, and evaluating LSTM models for cryptocurrency prediction.

## Core Functions

### `create_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate, prediction_length)`

Creates a LSTM model with dynamic output size.

**Parameters:**

- `input_shape`: Tuple of (timesteps, features)
- `lstm_units`: Number of LSTM units (default: 50)
- `dropout_rate`: Dropout rate (default: 0.2)
- `learning_rate`: Learning rate (default: 0.001)
- `prediction_length`: Number of future candles to predict (default: 1)

**Output Size:** `prediction_length * 5` (5 OHLCV values per candle)

**Example:**

```python
from src import create_lstm_model

# Single-step prediction
model1 = create_lstm_model(
    input_shape=(10, 5),
    prediction_length=1
)
print(f"Output shape: {model1.output_shape}")  # (None, 5)

# Multi-step prediction
model5 = create_lstm_model(
    input_shape=(10, 5),
    prediction_length=5
)
print(f"Output shape: {model5.output_shape}")  # (None, 25)
```

### `evaluate_model(model, X_test, y_test, scaler_y)`

Evaluates model performance on test data.

**Returns:** Dictionary with evaluation metrics

**Example:**

```python
from src import evaluate_model

metrics = evaluate_model(model, X_test, y_test, scaler_y)
print(f"Test Loss: {metrics['test_loss']:.4f}")
print(f"Test MAE: {metrics['test_mae']:.4f}")
print(f"Test MAPE: {metrics['test_mape']:.2f}%")
print(f"RMSE: {metrics['rmse']:.4f}")
```

## Complete Training Pipeline

```python
import tensorflow as tf
from src import BinanceDataOrganizer, DataConfig, create_lstm_model, evaluate_model

# 1. Configure and get data
config = DataConfig(
    symbol='BTCUSDT',
    timeframe='5m',
    start_time='2021-01',
    end_time='2021-01',
    sequence_length=10,
    prediction_length=1
)
organizer = BinanceDataOrganizer(config)
organizer.process_all()

data = organizer.get_scaled_data('all')

# 2. Create model
model = create_lstm_model(
    input_shape=(data['X_train_scaled'].shape[1], data['X_train_scaled'].shape[2]),
    prediction_length=config.prediction_length
)

# 3. Train model
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

# 4. Evaluate model
scalers = organizer.get_scalers()
metrics = evaluate_model(
    model=model,
    X_test=data['X_test_scaled'],
    y_test=data['y_test_scaled'],
    scaler_y=scalers['y']
)
```

## Model Architecture

1. **Input Layer**: Accepts (timesteps, features) input
2. **LSTM Layer 1**: LSTM with return_sequences=True
3. **Dropout Layer 1**: Regularization
4. **LSTM Layer 2**: LSTM with return_sequences=False
5. **Dropout Layer 2**: Additional regularization
6. **Dense Layer 1**: Hidden layer with 25 units and ReLU activation
7. **Dense Layer 2**: Output layer with dynamic size (prediction_length × 5)

## Multi-Step Prediction

```python
# Configure for multi-step prediction
config = DataConfig(
    symbol='BTCUSDT',
    timeframe='5m',
    start_time='2021-01',
    end_time='2021-01',
    sequence_length=10,
    prediction_length=5  # Predict 5 future candles
)

organizer = BinanceDataOrganizer(config)
organizer.process_all()
data = organizer.get_scaled_data('all')

# Create model for multi-step prediction
model = create_lstm_model(
    input_shape=(data['X_train_scaled'].shape[1], data['X_train_scaled'].shape[2]),
    prediction_length=5  # 5 candles × 5 OHLCV = 25 output values
)

# Train and use model
history = model.fit(data['X_train_scaled'], data['y_train_scaled'], ...)

# Make multi-step prediction
last_sequence = data['X_test_scaled'][-1:]  # Shape: (1, 10, 5)
predictions = model.predict(last_sequence)  # Shape: (1, 25)

# Reshape predictions to (5 candles, 5 OHLCV values)
predictions_reshaped = predictions.reshape(5, 5)
print(f"Predicted 5 future candles: {predictions_reshaped}")
```
