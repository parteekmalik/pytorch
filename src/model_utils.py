import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Union

def create_lstm_model(input_shape: Tuple[int, int], lstm_units: int = 50, dropout_rate: float = 0.2, learning_rate: float = 0.001) -> Sequential:
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(5)  # OHLCV output
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray, scaler_y: MinMaxScaler) -> Dict[str, Union[float, np.ndarray]]:
    print("📊 Evaluating model performance...")
    
    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    mse = np.mean((y_test_original - predictions) ** 2)
    mae = np.mean(np.abs(y_test_original - predictions))
    mape = np.mean(np.abs((y_test_original - predictions) / y_test_original)) * 100
    rmse = np.sqrt(mse)
    
    metrics = {
        'test_loss': mse,
        'test_mae': mae,
        'test_mape': mape,
        'rmse': rmse,
        'predictions': predictions,
        'y_true_original': y_test_original
    }
    
    print(f"📈 Test Loss: {mse:.4f}")
    print(f"📈 Test MAE: {mae:.4f}")
    print(f"📈 Test MAPE: {mape:.2f}%")
    print(f"📈 Test RMSE: {rmse:.4f}")
    
    return metrics

def predict_next_candle(model: Sequential, last_sequence: np.ndarray, scaler_X: MinMaxScaler, scaler_y: MinMaxScaler) -> np.ndarray:
    if last_sequence.ndim == 2:
        last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
    
    prediction_scaled = model.predict(last_sequence, verbose=0)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    return prediction