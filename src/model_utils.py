import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Tuple, Dict, Union

def create_lstm_model(input_shape: Tuple[int, int], lstm_units: int = 50, dropout_rate: float = 0.2, learning_rate: float = 0.001, prediction_length: int = 1) -> Sequential:
    """
    Create LSTM model for cryptocurrency price prediction.
    
    UPDATED FOR NEW APPROACH:
    - Input: OHLC data (4 features per timestep)
    - Output: HLC data (3 features per timestep) - Open is derived from previous Close
    - Uses expanded range scaling (no separate scalers needed)
    - All data is already in 0-1 range
    """
    # Calculate output size: prediction_length * 3 (HLC values - Open is derived from previous Close)
    output_size = prediction_length * 3
    
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(output_size, activation='sigmoid')  # Constrain output to 0-1 range
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def evaluate_model(model: Sequential, input_test: np.ndarray, output_test: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate model performance on test data.
    
    UPDATED FOR NEW APPROACH:
    - No scalers needed (data already in 0-1 range)
    - Direct evaluation on scaled data
    - Metrics calculated on scaled values
    """
    predictions_scaled = model.predict(input_test, verbose=0)
    
    # Calculate metrics on scaled data (already in 0-1 range)
    mse = np.mean((output_test - predictions_scaled) ** 2)
    mae = np.mean(np.abs(output_test - predictions_scaled))
    mape = np.mean(np.abs((output_test - predictions_scaled) / (output_test + 1e-8))) * 100  # Add small epsilon to avoid division by zero
    rmse = np.sqrt(mse)
    
    metrics = {
        'test_loss': mse,
        'test_mae': mae,
        'test_mape': mape,
        'rmse': rmse,
        'predictions': predictions_scaled,
        'output_true': output_test
    }
    
    return metrics

def predict_next_candle(model: Sequential, last_sequence: np.ndarray) -> np.ndarray:
    """
    Predict next candle using the model.
    
    UPDATED FOR NEW APPROACH:
    - No scalers needed (data already in 0-1 range)
    - Direct prediction on scaled data
    - Returns scaled predictions
    """
    if last_sequence.ndim == 2:
        last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
    
    prediction_scaled = model.predict(last_sequence, verbose=0)
    
    return prediction_scaled

def create_sequences_for_prediction(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sequences for prediction from scaled data.
    
    INPUT:
    - data: Scaled OHLC data (already in 0-1 range)
    - sequence_length: Length of input sequences
    
    OUTPUT:
    - Sequences ready for model prediction
    """
    sequences = []
    
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    
    return np.array(sequences)

def add_open_to_predictions(predictions: np.ndarray, last_close: float) -> np.ndarray:
    """
    Add Open column to predictions for proper OHLC format.
    
    LOGIC:
    - Predictions contain HLC values (3 features per timestep)
    - Add Open column derived from previous Close
    - First Open = last_close from input
    - Subsequent Opens = previous Close from predictions
    
    INPUT:
    - predictions: 2D array (num_predictions, prediction_length * 3)
    - last_close: Last Close value from input sequence
    
    OUTPUT:
    - 2D array (num_predictions, prediction_length * 4) with complete OHLC
    """
    num_predictions = predictions.shape[0]
    prediction_length = predictions.shape[1] // 3
    
    # Reshape predictions to (num_predictions, prediction_length, 3)
    predictions_reshaped = predictions.reshape(num_predictions, prediction_length, 3)
    
    # Create output with Open column
    predictions_with_open = np.zeros((num_predictions, prediction_length, 4))
    
    for i in range(num_predictions):
        pred = predictions_reshaped[i]
        
        # Set first Open = last_close, subsequent Opens = previous Close
        predictions_with_open[i, 0, 0] = last_close  # First Open
        predictions_with_open[i, :, 1] = pred[:, 0]  # High
        predictions_with_open[i, :, 2] = pred[:, 1]  # Low
        predictions_with_open[i, :, 3] = pred[:, 2]  # Close
        
        # Set remaining Opens = previous Close
        for j in range(1, prediction_length):
            predictions_with_open[i, j, 0] = predictions_with_open[i, j-1, 3]
    
    # Flatten back to original format
    return predictions_with_open.reshape(num_predictions, prediction_length * 4)

def calculate_prediction_metrics(predictions: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """
    Calculate detailed prediction metrics.
    
    INPUT:
    - predictions: Predicted values (scaled)
    - actual: Actual values (scaled)
    
    OUTPUT:
    - Dictionary with various metrics
    """
    # Reshape for per-timestep analysis
    prediction_length = predictions.shape[1] // 3
    pred_reshaped = predictions.reshape(-1, prediction_length, 3)
    actual_reshaped = actual.reshape(-1, prediction_length, 3)
    
    # Calculate metrics for each feature (H, L, C)
    feature_names = ['High', 'Low', 'Close']
    metrics = {}
    
    for i, feature in enumerate(feature_names):
        pred_feature = pred_reshaped[:, :, i]
        actual_feature = actual_reshaped[:, :, i]
        
        mse = np.mean((actual_feature - pred_feature) ** 2)
        mae = np.mean(np.abs(actual_feature - pred_feature))
        mape = np.mean(np.abs((actual_feature - pred_feature) / (actual_feature + 1e-8))) * 100
        
        metrics[f'{feature}_mse'] = mse
        metrics[f'{feature}_mae'] = mae
        metrics[f'{feature}_mape'] = mape
    
    # Overall metrics
    metrics['overall_mse'] = np.mean((actual - predictions) ** 2)
    metrics['overall_mae'] = np.mean(np.abs(actual - predictions))
    metrics['overall_mape'] = np.mean(np.abs((actual - predictions) / (actual + 1e-8))) * 100
    
    return metrics