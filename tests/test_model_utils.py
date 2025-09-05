#!/usr/bin/env python3
"""
Test suite for model utility functions.
Tests LSTM model creation, training, and evaluation.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_creation():
    """Test LSTM model creation."""
    print("🧪 Testing model creation...")
    
    try:
        from utils import create_lstm_model
        
        # Test model creation
        model = create_lstm_model(
            input_shape=(10, 20),
            output_dim=5,
            lstm_units=50,
            dropout_rate=0.2
        )
        
        # Validate model
        assert model is not None, "Model should be created"
        assert model.count_params() > 0, "Model should have parameters"
        print(f"   ✅ Model created with {model.count_params():,} parameters")
        
        # Test model architecture
        assert len(model.layers) > 0, "Model should have layers"
        print(f"   ✅ Model has {len(model.layers)} layers")
        
        # Test compilation
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("   ✅ Model compiled successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_model_training():
    """Test model training functionality."""
    print("🧪 Testing model training...")
    
    try:
        from utils import create_lstm_model, train_model_memory_efficient
        
        # Create test data
        np.random.seed(42)
        X_train = np.random.rand(100, 10, 20)
        y_train = np.random.rand(100, 5)
        X_test = np.random.rand(20, 10, 20)
        y_test = np.random.rand(20, 5)
        
        # Create model
        model = create_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_dim=y_train.shape[1],
            lstm_units=25,
            dropout_rate=0.1
        )
        
        # Test training
        print("   Testing memory efficient training...")
        history = train_model_memory_efficient(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
        # Validate training
        assert history is not None, "Training should return history"
        assert 'loss' in history.history, "History should contain loss"
        assert 'val_loss' in history.history, "History should contain validation loss"
        print(f"   ✅ Training completed, final loss: {history.history['loss'][-1]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def test_model_evaluation():
    """Test model evaluation functionality."""
    print("🧪 Testing model evaluation...")
    
    try:
        from utils import create_lstm_model, evaluate_model
        from sklearn.preprocessing import MinMaxScaler
        
        # Create test data
        np.random.seed(42)
        X_test = np.random.rand(20, 10, 20)
        y_test = np.random.rand(20, 5)
        
        # Create and train model
        model = create_lstm_model(
            input_shape=(X_test.shape[1], X_test.shape[2]),
            output_dim=y_test.shape[1],
            lstm_units=25,
            dropout_rate=0.1
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train briefly
        model.fit(X_test, y_test, epochs=2, verbose=0)
        
        # Create scaler
        scaler_y = MinMaxScaler()
        scaler_y.fit(y_test)
        
        # Test evaluation
        evaluation_results = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            scaler_y=scaler_y
        )
        
        # Validate results
        assert 'test_loss' in evaluation_results, "Should contain test loss"
        assert 'test_mae' in evaluation_results, "Should contain test MAE"
        assert 'predictions' in evaluation_results, "Should contain predictions"
        assert 'y_true_original' in evaluation_results, "Should contain true values"
        
        print(f"   ✅ Test loss: {evaluation_results['test_loss']:.6f}")
        print(f"   ✅ Test MAE: {evaluation_results['test_mae']:.6f}")
        print(f"   ✅ Predictions shape: {evaluation_results['predictions'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model evaluation error: {e}")
        return False

def test_predict_next_candle():
    """Test next candle prediction functionality."""
    print("🧪 Testing next candle prediction...")
    
    try:
        from utils import create_lstm_model, predict_next_candle
        from sklearn.preprocessing import MinMaxScaler
        
        # Create test data
        np.random.seed(42)
        last_sequence = np.random.rand(1, 10, 20)
        
        # Create model
        model = create_lstm_model(
            input_shape=(last_sequence.shape[1], last_sequence.shape[2]),
            output_dim=5,
            lstm_units=25,
            dropout_rate=0.1
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Create scalers
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        # Fit scalers on dummy data
        X_dummy = np.random.rand(100, 20)
        y_dummy = np.random.rand(100, 5)
        scaler_X.fit(X_dummy)
        scaler_y.fit(y_dummy)
        
        # Test prediction
        prediction = predict_next_candle(
            model=model,
            last_sequence=last_sequence,
            scaler_X=scaler_X,
            scaler_y=scaler_y
        )
        
        # Validate prediction
        assert prediction is not None, "Prediction should not be None"
        assert prediction.shape[1] == 5, "Prediction should have 5 outputs"
        print(f"   ✅ Prediction shape: {prediction.shape}")
        print(f"   ✅ Prediction range: {prediction.min():.4f} to {prediction.max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Next candle prediction error: {e}")
        return False

def test_model_integration():
    """Test model integration with BinanceDataOrganizer."""
    print("🧪 Testing model integration...")
    
    try:
        from utils import BinanceDataOrganizer, DataConfig, create_lstm_model, train_model_memory_efficient
        
        # Create organizer
        config = DataConfig(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time="2021-01-01",
            end_time="2021-01-01",
            sequence_length=5,
            prediction_length=1,
            max_rows=200,
            train_split=0.8
        )
        
        organizer = BinanceDataOrganizer(config)
        if not organizer.process_all():
            print("   ❌ Data processing failed")
            return False
        
        # Get scaled data
        scaled_data = organizer.get_scaled_data('all')
        X_train = scaled_data['X_train_scaled']
        y_train = scaled_data['y_train_scaled']
        X_test = scaled_data['X_test_scaled']
        y_test = scaled_data['y_test_scaled']
        
        # Create model
        model = create_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_dim=y_train.shape[1],
            lstm_units=25,
            dropout_rate=0.1
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        history = train_model_memory_efficient(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=3,
            batch_size=32,
            verbose=0
        )
        
        # Test prediction
        sample_input = X_test[:1]
        prediction = model.predict(sample_input, verbose=0)
        
        print(f"   ✅ Integration successful:")
        print(f"   ✅ Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"   ✅ Model parameters: {model.count_params():,}")
        print(f"   ✅ Prediction shape: {prediction.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model integration error: {e}")
        return False

def main():
    """Run all model utility tests."""
    print("🚀 MODEL UTILITIES TEST SUITE")
    print("=" * 40)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Model Training", test_model_training),
        ("Model Evaluation", test_model_evaluation),
        ("Next Candle Prediction", test_predict_next_candle),
        ("Model Integration", test_model_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📊 {test_name}")
        print("-" * 20)
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n📊 TEST RESULTS")
    print("=" * 40)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
