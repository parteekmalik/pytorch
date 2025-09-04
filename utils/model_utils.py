"""
Model utilities for crypto prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

def create_lstm_model(input_shape, output_dim, lstm_units=50, dropout_rate=0.2):
    """Create LSTM model for crypto prediction."""
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(output_dim)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_lightweight_lstm_model(input_shape, output_dim, lstm_units=25, dropout_rate=0.1):
    """Create lightweight LSTM model for memory efficiency."""
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(12, activation='relu'),
        Dense(output_dim)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def scale_data(X, y, scaler_X=None, scaler_y=None, fit_scalers=True):
    """Scale data for LSTM training."""
    if fit_scalers:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        # Reshape for scaling
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler_X.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        y_scaled = scaler_y.fit_transform(y)
        
        return X_scaled, y_scaled, scaler_X, scaler_y
    else:
        # Use existing scalers
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler_X.transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        y_scaled = scaler_y.transform(y)
        
        return X_scaled, y_scaled

def train_model_memory_efficient(model, X_train, y_train, X_test, y_test, 
                                epochs=50, batch_size=32, verbose=1):
    """Train model with memory efficiency."""
    print("🚀 Starting memory-efficient training...")
    
    # Callbacks for memory efficiency
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    print("✅ Training completed!")
    return history

def evaluate_model(model, X_test, y_test, scaler_y):
    """Evaluate model performance."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    # Calculate metrics
    mse = np.mean((y_pred_original - y_test_original) ** 2)
    mae = np.mean(np.abs(y_pred_original - y_test_original))
    rmse = np.sqrt(mse)
    
    print(f"📊 Model Performance:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'y_pred': y_pred_original,
        'y_test': y_test_original
    }

def predict_next_candle(model, last_sequence, scaler_X, scaler_y):
    """Predict the next candle using the last sequence."""
    # Scale the sequence
    sequence_reshaped = last_sequence.reshape(1, -1)
    sequence_scaled = scaler_X.transform(sequence_reshaped)
    sequence_scaled = sequence_scaled.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
    
    # Make prediction
    prediction_scaled = model.predict(sequence_scaled, verbose=0)
    
    # Inverse transform
    prediction_original = scaler_y.inverse_transform(prediction_scaled)
    
    return prediction_original[0]
