import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_utils import create_lstm_model, evaluate_model
from config import test_config


class TestCreateLSTMModel:
    """Test create_lstm_model function"""
    
    def test_create_lstm_model_basic(self):
        """Test basic LSTM model creation"""
        model = create_lstm_model(
            input_shape=(test_config.sequence_length, 5),  # test sequence length, 5 features
            lstm_units=test_config.lstm_units,
            dropout_rate=test_config.dropout_rate
        )
        
        assert isinstance(model, tf.keras.Model)
        assert len(model.layers) == 6  # LSTM, Dropout, LSTM, Dropout, Dense, Dense
        
        # Check input shape
        assert model.input_shape == (None, test_config.sequence_length, 5)
        
        # Check output shape
        assert model.output_shape == (None, 5)  # 5 OHLCV values
    
    def test_create_lstm_model_different_units(self):
        """Test LSTM model with different unit counts"""
        model = create_lstm_model(
            input_shape=(20, 5),
            lstm_units=100,
            dropout_rate=0.3
        )
        
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 20, 5)
        assert model.output_shape == (None, 5)
    
    def test_create_lstm_model_no_dropout(self):
        """Test LSTM model without dropout"""
        model = create_lstm_model(
            input_shape=(15, 5),
            lstm_units=75,
            dropout_rate=0.0
        )
        
        assert isinstance(model, tf.keras.Model)
        assert len(model.layers) == 6  # LSTM, Dropout, LSTM, Dropout, Dense, Dense (dropout still added even with 0.0 rate)
    
    def test_create_lstm_model_compilation(self):
        """Test that model can be compiled"""
        model = create_lstm_model(
            input_shape=(10, 5),
            lstm_units=50,
            dropout_rate=0.2
        )
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Model should compile without errors
        assert model.optimizer is not None
        assert model.loss == 'mse'
        # Check that metrics are available (newer TensorFlow versions handle this differently)
        assert hasattr(model, 'metrics')
    
    def test_create_lstm_model_forward_pass(self):
        """Test model forward pass with sample data"""
        model = create_lstm_model(
            input_shape=(5, 5),
            lstm_units=32,
            dropout_rate=0.1
        )
        
        # Create sample input data
        sample_input = np.random.rand(10, 5, 5)  # 10 samples, 5 timesteps, 5 features
        
        # Test forward pass
        output = model(sample_input)
        
        assert output.shape == (10, 5)  # 10 samples, 5 outputs
        assert isinstance(output, tf.Tensor)
    
    def test_create_lstm_model_parameters(self):
        """Test model parameter count"""
        model = create_lstm_model(
            input_shape=(10, 5),
            lstm_units=64,
            dropout_rate=0.2
        )
        
        param_count = model.count_params()
        assert param_count > 0
        
        # LSTM with 64 units should have significant parameters
        # Rough estimate: 4 * (input_size * units + units^2 + units) + dense layers
        expected_min_params = 4 * (5 * 64 + 64 * 64 + 64)  # LSTM parameters
        assert param_count >= expected_min_params


class TestEvaluateModel:
    """Test evaluate_model function"""
    
    def test_evaluate_model_basic(self):
        """Test basic model evaluation"""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(5)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Create sample data
        X_test = np.random.rand(100, 5)
        y_test = np.random.rand(100, 5)
        
        # Train model briefly
        model.fit(X_test, y_test, epochs=1, verbose=0)
        
        # Create a mock scaler for testing
        from sklearn.preprocessing import MinMaxScaler
        scaler_y = MinMaxScaler()
        scaler_y.fit(y_test)
        
        # Evaluate model
        results = evaluate_model(model, X_test, y_test, scaler_y)
        
        assert isinstance(results, dict)
        assert 'test_loss' in results
        assert 'test_mae' in results
        assert 'test_mape' in results
        assert 'rmse' in results
        
        # Check that values are reasonable
        assert results['test_loss'] >= 0
        assert results['test_mae'] >= 0
        assert results['test_mape'] >= 0
        assert results['rmse'] >= 0
    
    def test_evaluate_model_with_mape_calculation(self):
        """Test model evaluation with MAPE calculation"""
        # Create a model that predicts values close to actual
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(5)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Create data where model can learn
        X_test = np.random.rand(50, 5)
        y_test = X_test + np.random.normal(0, 0.1, X_test.shape)  # Close to input
        
        # Train model
        model.fit(X_test, y_test, epochs=5, verbose=0)
        
        # Create a mock scaler for testing
        from sklearn.preprocessing import MinMaxScaler
        scaler_y = MinMaxScaler()
        scaler_y.fit(y_test)
        
        # Evaluate
        results = evaluate_model(model, X_test, y_test, scaler_y)
        
        # MAPE should be reasonable (not too high)
        assert 0 <= results['test_mape'] <= 1000  # Allow for some error
    
    def test_evaluate_model_zero_values(self):
        """Test model evaluation with zero values (edge case for MAPE)"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, input_shape=(5,))
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Create data with some zero values
        X_test = np.random.rand(20, 5)
        y_test = np.random.rand(20, 5)
        y_test[0, 0] = 0  # Add a zero value
        
        # Train and evaluate
        model.fit(X_test, y_test, epochs=1, verbose=0)
        
        # Create a mock scaler for testing
        from sklearn.preprocessing import MinMaxScaler
        scaler_y = MinMaxScaler()
        scaler_y.fit(y_test)
        
        results = evaluate_model(model, X_test, y_test, scaler_y)
        
        # Should handle zero values gracefully
        assert isinstance(results['test_mape'], (int, float))
        assert not np.isnan(results['test_mape'])
    
    def test_evaluate_model_different_shapes(self):
        """Test model evaluation with different data shapes"""
        # Test with 2D input (no timesteps)
        model_2d = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(5)
        ])
        model_2d.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        X_test_2d = np.random.rand(30, 5)
        y_test_2d = np.random.rand(30, 5)
        
        model_2d.fit(X_test_2d, y_test_2d, epochs=1, verbose=0)
        
        # Create a mock scaler for testing
        from sklearn.preprocessing import MinMaxScaler
        scaler_y_2d = MinMaxScaler()
        scaler_y_2d.fit(y_test_2d)
        
        results_2d = evaluate_model(model_2d, X_test_2d, y_test_2d, scaler_y_2d)
        
        assert 'test_loss' in results_2d
        assert 'test_mae' in results_2d
        
        # Test with 3D input (with timesteps)
        model_3d = tf.keras.Sequential([
            tf.keras.layers.LSTM(10, input_shape=(5, 5)),
            tf.keras.layers.Dense(5)
        ])
        model_3d.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        X_test_3d = np.random.rand(20, 5, 5)
        y_test_3d = np.random.rand(20, 5)
        
        model_3d.fit(X_test_3d, y_test_3d, epochs=1, verbose=0)
        
        # Create a mock scaler for testing
        from sklearn.preprocessing import MinMaxScaler
        scaler_y_3d = MinMaxScaler()
        scaler_y_3d.fit(y_test_3d)
        
        results_3d = evaluate_model(model_3d, X_test_3d, y_test_3d, scaler_y_3d)
        
        assert 'test_loss' in results_3d
        assert 'test_mae' in results_3d
    
    def test_evaluate_model_return_format(self):
        """Test that evaluate_model returns the expected format"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, input_shape=(5,))
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        X_test = np.random.rand(10, 5)
        y_test = np.random.rand(10, 5)
        
        model.fit(X_test, y_test, epochs=1, verbose=0)
        
        # Create a mock scaler for testing
        from sklearn.preprocessing import MinMaxScaler
        scaler_y = MinMaxScaler()
        scaler_y.fit(y_test)
        
        results = evaluate_model(model, X_test, y_test, scaler_y)
        
        # Check all required keys are present
        required_keys = ['test_loss', 'test_mae', 'test_mape', 'rmse']
        for key in required_keys:
            assert key in results
            assert isinstance(results[key], (int, float))
            assert not np.isnan(results[key])
    
    def test_evaluate_model_mape_edge_cases(self):
        """Test MAPE calculation with edge cases"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, input_shape=(5,))
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Test with very small values
        X_test = np.random.rand(10, 5) * 0.001
        y_test = np.random.rand(10, 5) * 0.001
        
        model.fit(X_test, y_test, epochs=1, verbose=0)
        
        # Create a mock scaler for testing
        from sklearn.preprocessing import MinMaxScaler
        scaler_y = MinMaxScaler()
        scaler_y.fit(y_test)
        
        results = evaluate_model(model, X_test, y_test, scaler_y)
        
        # MAPE should be calculated without errors
        assert isinstance(results['test_mape'], (int, float))
        assert not np.isnan(results['test_mape'])
        assert not np.isinf(results['test_mape'])


if __name__ == '__main__':
    pytest.main([__file__])
