# Model Utilities

## Overview
The model utilities module provides functions for creating, training, and evaluating LSTM models for cryptocurrency prediction.

## Core Functions

### Model Creation

#### `create_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate)`
Creates a LSTM model for time series prediction.

**Parameters:**
- `input_shape`: Tuple of (timesteps, features)
- `lstm_units`: Number of LSTM units
- `dropout_rate`: Dropout rate for regularization
- `learning_rate`: Learning rate for optimizer

**Returns:** Compiled Keras model

**Example:**
```python
from utils import create_lstm_model

# Create LSTM model
model = create_lstm_model(
    input_shape=(10, 15),  # 10 timesteps, 15 features
    lstm_units=50,
    dropout_rate=0.2,
    learning_rate=0.001
)

print(f"Model created with {model.count_params()} parameters")
```

### Model Training

#### `train_model_memory_efficient(model, X_train, y_train, **kwargs)`
Trains a model with memory-efficient techniques.

**Parameters:**
- `model`: Keras model to train
- `X_train`: Training input data
- `y_train`: Training target data
- `**kwargs`: Additional training parameters

**Returns:** Training history

**Example:**
```python
from utils import train_model_memory_efficient

# Train model with memory efficiency
history = train_model_memory_efficient(
    model=model,
    X_train=X_train,
    y_train=y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

### Model Evaluation

#### `evaluate_model(model, X_test, y_test, scaler_y)`
Evaluates model performance on test data.

**Parameters:**
- `model`: Trained Keras model
- `X_test`: Test input data
- `y_test`: Test target data
- `scaler_y`: Target scaler for inverse transformation

**Returns:** Dictionary with evaluation metrics

**Example:**
```python
from utils import evaluate_model

# Evaluate model
metrics = evaluate_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    scaler_y=scaler_y
)

print(f"Test Loss: {metrics['loss']:.4f}")
print(f"Test MAE: {metrics['mae']:.4f}")
print(f"Test MAPE: {metrics['mape']:.2f}%")
```

### Prediction Utilities

#### `predict_next_candle(model, last_sequence, scaler_X, scaler_y)`
Makes prediction for the next candle.

**Parameters:**
- `model`: Trained Keras model
- `last_sequence`: Last sequence of data
- `scaler_X`: Input scaler
- `scaler_y`: Target scaler

**Returns:** Predicted OHLCV values

**Example:**
```python
from utils import predict_next_candle

# Get last sequence
last_sequence = X_test[-1:]  # Shape: (1, timesteps, features)

# Make prediction
prediction = predict_next_candle(
    model=model,
    last_sequence=last_sequence,
    scaler_X=scaler_X,
    scaler_y=scaler_y
)

print(f"Predicted OHLCV: {prediction}")
```

## Complete Training Pipeline

### Basic Training

```python
from utils import (
    BinanceDataOrganizer, DataConfig,
    create_lstm_model, train_model_memory_efficient, evaluate_model
)

# 1. Get data
config = DataConfig('BTCUSDT', '5m', '2021-01-01', '2021-01-31', 10, 1)
organizer = BinanceDataOrganizer(config)
organizer.process_all()

data = organizer.get_scaled_data('all')
X_train = data['X_train_scaled']
y_train = data['y_train_scaled']
X_test = data['X_test_scaled']
y_test = data['y_test_scaled']

# 2. Create model
model = create_lstm_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    lstm_units=50,
    dropout_rate=0.2
)

# 3. Train model
history = train_model_memory_efficient(
    model=model,
    X_train=X_train,
    y_train=y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

# 4. Evaluate model
scalers = organizer.get_scalers()
metrics = evaluate_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    scaler_y=scalers['y']
)
```

### Advanced Training with Callbacks

```python
import tensorflow as tf
from utils import create_lstm_model, train_model_memory_efficient

# Create model
model = create_lstm_model(
    input_shape=(10, 15),
    lstm_units=100,
    dropout_rate=0.3,
    learning_rate=0.0005
)

# Train with callbacks
history = train_model_memory_efficient(
    model=model,
    X_train=X_train,
    y_train=y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
    ]
)
```

## Model Architecture

### LSTM Architecture
The default LSTM model includes:

1. **Input Layer**: Accepts (timesteps, features) input
2. **LSTM Layer**: Bidirectional LSTM with specified units
3. **Dropout Layer**: Regularization to prevent overfitting
4. **Dense Layer**: Output layer with 5 units (OHLCV)

### Custom Architecture
You can create custom models:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_custom_lstm(input_shape, lstm_units, dropout_rate):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(5)  # OHLCV output
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

## Training Strategies

### Memory-Efficient Training
The `train_model_memory_efficient` function includes:

- **Gradient Accumulation**: Reduces memory usage
- **Memory Monitoring**: Tracks memory usage during training
- **Garbage Collection**: Cleans up memory between epochs
- **Batch Size Optimization**: Adjusts batch size based on available memory

### Hyperparameter Tuning

```python
# Test different configurations
configs = [
    {'lstm_units': 50, 'dropout_rate': 0.2, 'learning_rate': 0.001},
    {'lstm_units': 100, 'dropout_rate': 0.3, 'learning_rate': 0.0005},
    {'lstm_units': 200, 'dropout_rate': 0.4, 'learning_rate': 0.0001}
]

best_model = None
best_score = float('inf')

for config in configs:
    model = create_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        **config
    )
    
    history = train_model_memory_efficient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs=50,
        validation_split=0.2
    )
    
    val_loss = min(history.history['val_loss'])
    if val_loss < best_score:
        best_score = val_loss
        best_model = model
```

## Evaluation Metrics

### Available Metrics
The `evaluate_model` function calculates:

- **Loss**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Squared Error

### Custom Evaluation

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def custom_evaluate(model, X_test, y_test, scaler_y):
    # Make predictions
    predictions_scaled = model.predict(X_test)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, predictions)
    mae = mean_absolute_error(y_test_original, predictions)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test_original - predictions) / y_test_original)) * 100
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }
```

## Production Deployment

### Model Saving

```python
# Save model
model.save('crypto_prediction_model.h5')

# Save with scalers
import pickle
with open('model_scalers.pkl', 'wb') as f:
    pickle.dump({
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }, f)
```

### Model Loading

```python
# Load model
model = tf.keras.models.load_model('crypto_prediction_model.h5')

# Load scalers
with open('model_scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']
```

### Real-time Prediction

```python
def predict_real_time(model, organizer, symbol, timeframe):
    # Get latest data
    latest_data = organizer.get_data_in_range(
        start_time='2021-02-01',
        end_time='2021-02-01',
        scaled=True
    )
    
    if latest_data:
        # Make prediction
        prediction = predict_next_candle(
            model=model,
            last_sequence=latest_data['X_scaled'][-1:],
            scaler_X=organizer.get_scalers()['X'],
            scaler_y=organizer.get_scalers()['y']
        )
        
        return prediction
    
    return None
```

## Troubleshooting

### Common Issues

**Memory Errors:**
```python
# Reduce batch size
history = train_model_memory_efficient(
    model=model,
    X_train=X_train,
    y_train=y_train,
    batch_size=16  # Reduced from 32
)
```

**Overfitting:**
```python
# Increase dropout rate
model = create_lstm_model(
    input_shape=(10, 15),
    lstm_units=50,
    dropout_rate=0.5  # Increased from 0.2
)
```

**Poor Performance:**
```python
# Try different architectures
model = create_lstm_model(
    input_shape=(10, 15),
    lstm_units=200,  # Increased units
    dropout_rate=0.3,
    learning_rate=0.0001  # Lower learning rate
)
```

### Debug Information

```python
# Print model summary
model.summary()

# Print training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.legend()
plt.show()
```

## Best Practices

1. **Start Simple**: Begin with basic architecture
2. **Monitor Training**: Watch for overfitting
3. **Use Validation**: Always use validation split
4. **Save Models**: Persist trained models
5. **Test Thoroughly**: Evaluate on unseen data
6. **Monitor Memory**: Use memory-efficient training
7. **Tune Hyperparameters**: Experiment with different configurations
