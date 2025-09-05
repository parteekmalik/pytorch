#!/usr/bin/env python3
"""
Test suite for data utility functions.
Tests data downloading, feature creation, and preprocessing.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_download():
    """Test data downloading functionality."""
    print("🧪 Testing data download...")
    
    try:
        from utils import download_crypto_data
        
        # Test 1: Download small dataset
        print("   Testing BTCUSDT 5m data download...")
        data = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 01",
            max_rows=100
        )
        
        if data is not None and not data.empty:
            print(f"   ✅ Downloaded {len(data)} rows")
            print(f"   ✅ Data shape: {data.shape}")
            print(f"   ✅ Columns: {list(data.columns)}")
            return True
        else:
            print("   ❌ No data downloaded")
            return False
            
    except Exception as e:
        print(f"❌ Data download error: {e}")
        return False

def test_feature_creation():
    """Test feature creation functionality."""
    print("🧪 Testing feature creation...")
    
    try:
        from utils import create_minimal_features, download_crypto_data
        
        # Get test data
        data = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 01",
            max_rows=100
        )
        
        if data is None or data.empty:
            print("   ❌ No data available for testing")
            return False
        
        # Test feature creation
        print("   Testing minimal features...")
        features = create_minimal_features(data, lag_period=3)
        
        print(f"   ✅ Original data shape: {data.shape}")
        print(f"   ✅ Features shape: {features.shape}")
        print(f"   ✅ Feature columns: {len(features.columns)}")
        
        # Check for expected features
        expected_features = ['Hour', 'Minute', 'Price_Range', 'Price_Change']
        for feature in expected_features:
            if feature in features.columns:
                print(f"   ✅ Found expected feature: {feature}")
            else:
                print(f"   ⚠️ Missing expected feature: {feature}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature creation error: {e}")
        return False

def test_sliding_windows():
    """Test sliding window creation."""
    print("🧪 Testing sliding windows...")
    
    try:
        from utils import create_sliding_windows, create_minimal_features, download_crypto_data
        
        # Get test data
        data = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 01",
            max_rows=100
        )
        
        if data is None or data.empty:
            print("   ❌ No data available for testing")
            return False
        
        # Create features
        features = create_minimal_features(data, lag_period=3)
        
        # Test sliding windows
        print("   Testing sliding window creation...")
        X, y, feature_cols = create_sliding_windows(features, sequence_length=5)
        
        print(f"   ✅ X shape: {X.shape}")
        print(f"   ✅ y shape: {y.shape}")
        print(f"   ✅ Feature columns: {len(feature_cols)}")
        
        # Validate shapes
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert X.shape[1] == 5, "X should have sequence length of 5"
        assert y.shape[1] == 5, "y should have 5 target columns"
        
        print("   ✅ Sliding windows created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Sliding windows error: {e}")
        return False

def test_date_parsing():
    """Test date range parsing functionality."""
    print("🧪 Testing date parsing...")
    
    try:
        from utils import parse_date_range
        
        # Test various date formats
        test_cases = [
            ("2021 01", "2021 01", "2021", ["01"]),
            ("2021-01-01", "2021-01-31", "2021", ["01"]),
            ("2021 01", "2021 03", "2021", ["01", "02", "03"]),
        ]
        
        for data_from, data_to, expected_year, expected_months in test_cases:
            year, months = parse_date_range(data_from, data_to)
            assert year == expected_year, f"Expected year {expected_year}, got {year}"
            assert months == expected_months, f"Expected months {expected_months}, got {months}"
            print(f"   ✅ {data_from} to {data_to} -> {year} {months}")
        
        print("   ✅ Date parsing working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Date parsing error: {e}")
        return False

def main():
    """Run all data utility tests."""
    print("🚀 DATA UTILITIES TEST SUITE")
    print("=" * 40)
    
    tests = [
        ("Data Download", test_data_download),
        ("Feature Creation", test_feature_creation),
        ("Sliding Windows", test_sliding_windows),
        ("Date Parsing", test_date_parsing),
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
