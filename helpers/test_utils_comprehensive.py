#!/usr/bin/env python3
"""
Comprehensive test suite for all utils modules.
Tests data_utils, memory_utils, and model_utils functionality.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append('/Users/parteekmalik/github/pytorch')

def test_imports():
    """Test all required imports."""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error
        import matplotlib.pyplot as plt
        import requests
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_utils_imports():
    """Test utils module imports."""
    print("🧪 Testing utils imports...")
    
    try:
        from utils import (
            # Memory utilities
            get_memory_usage, check_memory_limit, force_garbage_collection,
            get_memory_stats, print_memory_stats,
            
            # Data utilities
            download_binance_vision_data, create_minimal_features, 
            create_sliding_windows, load_multiple_months_data,
            parse_date_range, download_crypto_data,
            
            # Model utilities
            create_lstm_model, create_lightweight_lstm_model, scale_data,
            train_model_memory_efficient, evaluate_model, predict_next_candle
        )
        print("✅ Utils imports successful!")
        return True
    except Exception as e:
        print(f"❌ Utils import error: {e}")
        return False

def test_memory_utils():
    """Test memory utility functions."""
    print("🧪 Testing memory utilities...")
    
    try:
        from utils import get_memory_usage, check_memory_limit, force_garbage_collection
        
        # Test memory usage
        memory = get_memory_usage()
        print(f"   Current memory usage: {memory:.1f} MB")
        
        # Test memory limit check
        within_limit = check_memory_limit(max_memory_mb=8000)
        print(f"   Within 8GB limit: {within_limit}")
        
        # Test garbage collection
        force_garbage_collection()
        print("   Garbage collection completed")
        
        print("✅ Memory utilities working!")
        return True
    except Exception as e:
        print(f"❌ Memory utils error: {e}")
        return False

def test_data_utils():
    """Test data utility functions."""
    print("🧪 Testing data utilities...")
    
    try:
        from utils import download_crypto_data, create_minimal_features, create_sliding_windows
        
        # Test 1: Download data
        print("   Testing data download...")
        data = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 01",
            max_rows=1000  # Small test
        )
        
        if data is not None and not data.empty:
            print(f"   ✅ Downloaded {len(data)} rows")
            
            # Test 2: Create features
            print("   Testing feature creation...")
            features = create_minimal_features(data, lag_period=3)
            print(f"   ✅ Created features: {features.shape}")
            
            # Test 3: Create sliding windows
            print("   Testing sliding windows...")
            X, y, feature_cols = create_sliding_windows(features, sequence_length=5)
            print(f"   ✅ Created windows: X={X.shape}, y={y.shape}")
            
            print("✅ Data utilities working!")
            return True
        else:
            print("   ❌ No data downloaded")
            return False
            
    except Exception as e:
        print(f"❌ Data utils error: {e}")
        return False

def test_model_utils():
    """Test model utility functions."""
    print("🧪 Testing model utilities...")
    
    try:
        from utils import create_lstm_model, create_lightweight_lstm_model, scale_data
        
        # Test 1: Create models
        print("   Testing model creation...")
        model1 = create_lstm_model(input_shape=(10, 20), output_dim=5)
        model2 = create_lightweight_lstm_model(input_shape=(10, 20), output_dim=5)
        print(f"   ✅ Created models: {model1.count_params()} and {model2.count_params()} parameters")
        
        # Test 2: Scale data
        print("   Testing data scaling...")
        X_train = np.random.rand(100, 10, 20)
        X_test = np.random.rand(20, 10, 20)
        y_train = np.random.rand(100, 5)
        y_test = np.random.rand(20, 5)
        
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = scale_data(
            X_train, X_test, y_train, y_test
        )
        print(f"   ✅ Scaled data: X_train={X_train_scaled.shape}, y_train={y_train_scaled.shape}")
        
        print("✅ Model utilities working!")
        return True
    except Exception as e:
        print(f"❌ Model utils error: {e}")
        return False

def test_notebook_configurations():
    """Test notebook configurations."""
    print("🧪 Testing notebook configurations...")
    
    try:
        # Test original notebook config
        print("   Testing original notebook config...")
        from utils import download_crypto_data
        
        data = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 01",
            max_rows=5000
        )
        
        if data is not None:
            print(f"   ✅ Original config: Downloaded {len(data)} rows")
        else:
            print("   ❌ Original config: No data")
            return False
        
        # Test on-demand notebook config
        print("   Testing on-demand notebook config...")
        data2 = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 01",
            max_rows=3000
        )
        
        if data2 is not None:
            print(f"   ✅ On-demand config: Downloaded {len(data2)} rows")
        else:
            print("   ❌ On-demand config: No data")
            return False
        
        print("✅ Notebook configurations working!")
        return True
    except Exception as e:
        print(f"❌ Notebook config error: {e}")
        return False

def test_complete_pipeline():
    """Test complete data processing pipeline."""
    print("🧪 Testing complete pipeline...")
    
    try:
        from utils import download_crypto_data, create_minimal_features, create_sliding_windows, scale_data, create_lstm_model
        
        # Step 1: Download data
        print("   Step 1: Downloading data...")
        data = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 01",
            max_rows=2000
        )
        
        if data is None or data.empty:
            print("   ❌ No data available")
            return False
        
        # Step 2: Create features
        print("   Step 2: Creating features...")
        features = create_minimal_features(data, lag_period=3)
        
        # Step 3: Create sliding windows
        print("   Step 3: Creating sliding windows...")
        X, y, feature_cols = create_sliding_windows(features, sequence_length=5)
        
        # Step 4: Split data
        print("   Step 4: Splitting data...")
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Step 5: Scale data
        print("   Step 5: Scaling data...")
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = scale_data(
            X_train, X_test, y_train, y_test
        )
        
        # Step 6: Create model
        print("   Step 6: Creating model...")
        model = create_lstm_model(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            output_dim=y_train_scaled.shape[1]
        )
        
        print(f"   ✅ Complete pipeline: X_train={X_train_scaled.shape}, y_train={y_train_scaled.shape}")
        print(f"   ✅ Model created with {model.count_params()} parameters")
        
        print("✅ Complete pipeline working!")
        return True
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 COMPREHENSIVE UTILS TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Utils Imports", test_utils_imports),
        ("Memory Utils", test_memory_utils),
        ("Data Utils", test_data_utils),
        ("Model Utils", test_model_utils),
        ("Notebook Configs", test_notebook_configurations),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📊 {test_name}")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n📊 TEST RESULTS")
    print("=" * 50)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Utils are working correctly.")
        return True
    else:
        print(f"\n💥 {total - passed} TESTS FAILED! Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
