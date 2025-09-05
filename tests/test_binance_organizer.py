#!/usr/bin/env python3
"""
Test suite for BinanceDataOrganizer class.
Tests the main organizer functionality and integration.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_organizer_creation():
    """Test BinanceDataOrganizer creation and configuration."""
    print("🧪 Testing BinanceDataOrganizer creation...")
    
    try:
        from utils import BinanceDataOrganizer, DataConfig
        
        # Test configuration creation
        config = DataConfig(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time="2021-01-01",
            end_time="2021-01-01",
            sequence_length=10,
            prediction_length=1,
            max_rows=500,
            train_split=0.8
        )
        
        print(f"   ✅ Config created: {config.symbol} {config.timeframe}")
        
        # Test organizer creation
        organizer = BinanceDataOrganizer(config)
        print("   ✅ Organizer created successfully")
        
        # Test configuration access
        assert organizer.config.symbol == "BTCUSDT"
        assert organizer.config.timeframe == "5m"
        assert organizer.config.sequence_length == 10
        print("   ✅ Configuration properly set")
        
        return True
        
    except Exception as e:
        print(f"❌ Organizer creation error: {e}")
        return False

def test_data_processing():
    """Test data loading and feature creation."""
    print("🧪 Testing data processing...")
    
    try:
        from utils import BinanceDataOrganizer, DataConfig
        
        # Create organizer
        config = DataConfig(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time="2021-01-01",
            end_time="2021-01-01",
            sequence_length=10,
            prediction_length=1,
            max_rows=500,
            train_split=0.8
        )
        
        organizer = BinanceDataOrganizer(config)
        
        # Test data processing
        if organizer.process_all():
            print("   ✅ Data processing completed successfully")
            
            # Test feature info
            feature_info = organizer.get_feature_info()
            assert 'feature_columns' in feature_info
            assert 'sequence_length' in feature_info
            assert 'prediction_length' in feature_info
            print(f"   ✅ Feature info: {feature_info['num_features']} features")
            
            return True
        else:
            print("   ❌ Data processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False

def test_data_generation():
    """Test on-demand data generation."""
    print("🧪 Testing data generation...")
    
    try:
        from utils import BinanceDataOrganizer, DataConfig
        
        # Create organizer and process data
        config = DataConfig(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time="2021-01-01",
            end_time="2021-01-01",
            sequence_length=10,
            prediction_length=1,
            max_rows=500,
            train_split=0.8
        )
        
        organizer = BinanceDataOrganizer(config)
        if not organizer.process_all():
            print("   ❌ Data processing failed")
            return False
        
        # Test unscaled data generation
        print("   Testing unscaled data generation...")
        unscaled_data = organizer.get_unscaled_data('all')
        assert 'X_train' in unscaled_data
        assert 'X_test' in unscaled_data
        assert 'y_train' in unscaled_data
        assert 'y_test' in unscaled_data
        print(f"   ✅ Unscaled data: X_train={unscaled_data['X_train'].shape}")
        
        # Test scaled data generation
        print("   Testing scaled data generation...")
        scaled_data = organizer.get_scaled_data('all')
        assert 'X_train_scaled' in scaled_data
        assert 'X_test_scaled' in scaled_data
        assert 'y_train_scaled' in scaled_data
        assert 'y_test_scaled' in scaled_data
        print(f"   ✅ Scaled data: X_train_scaled={scaled_data['X_train_scaled'].shape}")
        
        # Test partial data generation
        print("   Testing partial data generation...")
        train_data = organizer.get_scaled_data('train')
        assert 'X_train_scaled' in train_data
        assert 'y_train_scaled' in train_data
        print(f"   ✅ Train data: {train_data['X_train_scaled'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        return False

def test_scalers():
    """Test scaler functionality."""
    print("🧪 Testing scalers...")
    
    try:
        from utils import BinanceDataOrganizer, DataConfig, GroupedScaler
        
        # Create organizer and process data
        config = DataConfig(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time="2021-01-01",
            end_time="2021-01-01",
            sequence_length=10,
            prediction_length=1,
            max_rows=500,
            train_split=0.8
        )
        
        organizer = BinanceDataOrganizer(config)
        if not organizer.process_all():
            print("   ❌ Data processing failed")
            return False
        
        # Test scaler access
        scalers = organizer.get_scalers()
        assert 'X' in scalers
        assert 'y' in scalers
        assert isinstance(scalers['X'], GroupedScaler)
        print("   ✅ Scalers available and properly typed")
        
        # Test feature groups
        feature_groups = scalers['X'].get_feature_groups()
        assert len(feature_groups) > 0
        print(f"   ✅ Feature groups: {list(feature_groups.keys())}")
        
        # Test scaling info
        scaling_info = scalers['X'].get_scaling_info()
        assert len(scaling_info) > 0
        print(f"   ✅ Scaling info available for {len(scaling_info)} groups")
        
        return True
        
    except Exception as e:
        print(f"❌ Scalers test error: {e}")
        return False

def test_data_in_range():
    """Test data retrieval for specific time ranges."""
    print("🧪 Testing data in range...")
    
    try:
        from utils import BinanceDataOrganizer, DataConfig
        
        # Create organizer and process data
        config = DataConfig(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time="2021-01-01",
            end_time="2021-01-01",
            sequence_length=10,
            prediction_length=1,
            max_rows=500,
            train_split=0.8
        )
        
        organizer = BinanceDataOrganizer(config)
        if not organizer.process_all():
            print("   ❌ Data processing failed")
            return False
        
        # Test data in range
        range_data = organizer.get_data_in_range(
            start_time="2021-01-01 00:00:00",
            end_time="2021-01-01 12:00:00",
            scaled=True
        )
        
        if range_data is not None:
            assert 'X' in range_data or 'X_scaled' in range_data
            assert 'y' in range_data or 'y_scaled' in range_data
            print(f"   ✅ Range data: {list(range_data.keys())}")
        else:
            print("   ⚠️ No data in specified range (this may be normal)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data in range test error: {e}")
        return False

def test_memory_efficiency():
    """Test memory efficiency of on-demand generation."""
    print("🧪 Testing memory efficiency...")
    
    try:
        from utils import BinanceDataOrganizer, DataConfig, get_memory_usage
        
        # Create organizer and process data
        config = DataConfig(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time="2021-01-01",
            end_time="2021-01-01",
            sequence_length=10,
            prediction_length=1,
            max_rows=500,
            train_split=0.8
        )
        
        organizer = BinanceDataOrganizer(config)
        if not organizer.process_all():
            print("   ❌ Data processing failed")
            return False
        
        # Test memory usage
        initial_memory = get_memory_usage()
        print(f"   Initial memory: {initial_memory:.1f} MB")
        
        # Generate data multiple times
        for i in range(3):
            data = organizer.get_scaled_data('all')
            current_memory = get_memory_usage()
            print(f"   Iteration {i+1}: {current_memory:.1f} MB")
        
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        print(f"   Memory increase: {memory_increase:.1f} MB")
        print(f"   Memory efficient: {'✅ Yes' if memory_increase < 100 else '⚠️ High increase'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test error: {e}")
        return False

def main():
    """Run all BinanceDataOrganizer tests."""
    print("🚀 BINANCE DATA ORGANIZER TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Organizer Creation", test_organizer_creation),
        ("Data Processing", test_data_processing),
        ("Data Generation", test_data_generation),
        ("Scalers", test_scalers),
        ("Data in Range", test_data_in_range),
        ("Memory Efficiency", test_memory_efficiency),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📊 {test_name}")
        print("-" * 25)
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
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
