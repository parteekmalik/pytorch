import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Union

def create_lstm_model(input_shape: Tuple[int, int], lstm_units: int = 50, dropout_rate: float = 0.2, learning_rate: float = 0.001, prediction_length: int = 1) -> Sequential:
    # Calculate output size: prediction_length * 4 (HLCV values - Open is derived from previous Close)
    output_size = prediction_length * 4
    
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(output_size)  # Dynamic output based on prediction_length
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def evaluate_model(model: Sequential, input_test: np.ndarray, output_test: np.ndarray, scaler_output: MinMaxScaler) -> Dict[str, Union[float, np.ndarray]]:
    
    predictions_scaled = model.predict(input_test, verbose=0)
    predictions = scaler_output.inverse_transform(predictions_scaled)
    output_test_original = scaler_output.inverse_transform(output_test)
    
    mse = np.mean((output_test_original - predictions) ** 2)
    mae = np.mean(np.abs(output_test_original - predictions))
    mape = np.mean(np.abs((output_test_original - predictions) / output_test_original)) * 100
    rmse = np.sqrt(mse)
    
    metrics = {
        'test_loss': mse,
        'test_mae': mae,
        'test_mape': mape,
        'rmse': rmse,
        'predictions': predictions,
        'output_true_original': output_test_original
    }
    
    
    return metrics

def predict_next_candle(model: Sequential, last_sequence: np.ndarray, scaler_input: MinMaxScaler, scaler_output: MinMaxScaler) -> np.ndarray:
    if last_sequence.ndim == 2:
        last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
    
    prediction_scaled = model.predict(last_sequence, verbose=0)
    prediction = scaler_output.inverse_transform(prediction_scaled)
    
    return prediction