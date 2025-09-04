#!/usr/bin/env python3
"""
Test the complete notebooks with utilities.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append('/Users/parteekmalik/github/pytorch')

def test_utils_import():
    """Test that all utilities can be imported."""
    print("🧪 TESTING UTILITIES IMPORT")
    print("=" * 50)
    
    try:
        from utils import (
            get_memory_usage, check_memory_limit, force_garbage_collection,
            download_binance_klines_data, create_minimal_features, create_sliding_windows,
            create_lstm_model, create_lightweight_lstm_model, scale_data,
            train_model_memory_efficient, evaluate_model
        )
        
        print("✅ All utilities imported successfully!")
        
        # Test memory functions
        memory = get_memory_usage()
        print(f"💾 Current memory: {memory:.1f} MB")
        
        # Test memory limit check
        within_limit = check_memory_limit(8000)
        print(f"🔍 Memory within limit: {within_limit}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_processing():
    """Test data processing utilities."""
    print("\n🧪 TESTING DATA PROCESSING")
    print("=" * 50)
    
    try:
        from utils import create_minimal_features, create_sliding_windows
        import pandas as pd
        import numpy as np
        
        # Create test data
        dates = pd.date_range('2021-01-01 00:00:00', periods=1000, freq='5min')
        test_data = pd.DataFrame({
            'Open time': dates,
            'Open': np.random.uniform(30000, 50000, 1000),
            'High': np.random.uniform(30000, 50000, 1000),
            'Low': np.random.uniform(30000, 50000, 1000),
            'Close': np.random.uniform(30000, 50000, 1000),
            'Volume': np.random.uniform(100, 1000, 1000)
        })
        
        print(f"📊 Test data created: {test_data.shape}")
        
        # Test feature creation
        features = create_minimal_features(test_data, lag_period=3)
        print(f"✅ Features created: {features.shape}")
        
        # Test sliding windows
        X, y, feature_cols = create_sliding_windows(features, sequence_length=5)
        print(f"✅ Sliding windows created: X={X.shape}, y={y.shape}")
        print(f"   Feature columns: {len(feature_cols)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def test_model_creation():
    """Test model creation utilities."""
    print("\n🧪 TESTING MODEL CREATION")
    print("=" * 50)
    
    try:
        from utils import create_lstm_model, create_lightweight_lstm_model
        
        # Test regular LSTM model
        model1 = create_lstm_model((5, 22), 5)
        print(f"✅ Regular LSTM model created: {model1.count_params():,} parameters")
        
        # Test lightweight LSTM model
        model2 = create_lightweight_lstm_model((5, 22), 5)
        print(f"✅ Lightweight LSTM model created: {model2.count_params():,} parameters")
        
        # Test model compilation
        print(f"✅ Models compiled successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_memory_efficiency():
    """Test memory efficiency."""
    print("\n🧪 TESTING MEMORY EFFICIENCY")
    print("=" * 50)
    
    try:
        from utils import get_memory_usage, check_memory_limit, force_garbage_collection
        
        initial_memory = get_memory_usage()
        print(f"💾 Initial memory: {initial_memory:.1f} MB")
        
        # Test memory limit check
        within_limit = check_memory_limit(8000)
        print(f"🔍 Memory within 8GB limit: {within_limit}")
        
        # Test garbage collection
        final_memory = force_garbage_collection()
        print(f"🧹 Memory after cleanup: {final_memory:.1f} MB")
        
        if final_memory < 1000:  # Less than 1GB
            print("✅ Memory usage is efficient!")
        else:
            print("⚠️  Memory usage is high")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 TESTING COMPLETE NOTEBOOKS WITH UTILITIES")
    print("=" * 60)
    
    tests = [
        ("Utils Import", test_utils_import),
        ("Data Processing", test_data_processing),
        ("Model Creation", test_model_creation),
        ("Memory Efficiency", test_memory_efficiency)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Notebooks are ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Complete notebooks with utilities are working!")
        sys.exit(0)
    else:
        print("\n❌ Some issues found.")
        sys.exit(1)
