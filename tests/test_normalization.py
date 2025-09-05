#!/usr/bin/env python3
"""
Test suite for normalization and scaling functionality.
Tests GroupedScaler and related normalization utilities.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_grouped_scaler_basic():
    """Test basic GroupedScaler functionality."""
    print("🧪 Testing GroupedScaler basic functionality...")
    
    try:
        from utils import GroupedScaler
        
        # Create test data
        np.random.seed(42)
        X = np.random.rand(100, 5)
        feature_names = ['Price_Range', 'Price_Change', 'Volume', 'Taker_buy_volume', 'Minutes_of_day']
        
        # Test fit and transform
        scaler = GroupedScaler()
        X_scaled = scaler.fit_transform(X, feature_names)
        
        # Validate results
        assert X_scaled.shape == X.shape, "Shape should remain the same"
        assert not np.isnan(X_scaled).any(), "No NaN values should be present"
        print(f"   ✅ Scaled data shape: {X_scaled.shape}")
        print(f"   ✅ Scaled data range: {X_scaled.min():.4f} to {X_scaled.max():.4f}")
        
        # Test inverse transform
        X_original = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X_original, X, decimal=6)
        print("   ✅ Inverse transform working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ GroupedScaler basic test error: {e}")
        return False

def test_grouped_scaler_3d():
    """Test GroupedScaler with 3D data."""
    print("🧪 Testing GroupedScaler with 3D data...")
    
    try:
        from utils import GroupedScaler
        
        # Create 3D test data (samples, timesteps, features)
        np.random.seed(42)
        X_3d = np.random.rand(50, 10, 5)
        feature_names = ['Price_Range', 'Price_Change', 'Volume', 'Taker_buy_volume', 'Minutes_of_day']
        
        # Test fit and transform
        scaler = GroupedScaler()
        X_scaled = scaler.fit_transform(X_3d, feature_names)
        
        # Validate results
        assert X_scaled.shape == X_3d.shape, "3D shape should remain the same"
        assert not np.isnan(X_scaled).any(), "No NaN values should be present"
        print(f"   ✅ 3D scaled data shape: {X_scaled.shape}")
        
        # Test inverse transform
        X_original = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X_original, X_3d, decimal=6)
        print("   ✅ 3D inverse transform working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ GroupedScaler 3D test error: {e}")
        return False

def test_feature_groups():
    """Test feature grouping functionality."""
    print("🧪 Testing feature grouping...")
    
    try:
        from utils import GroupedScaler
        
        # Create test data with different feature types
        feature_names = [
            'Price_Range', 'Price_Change', 'Open', 'High', 'Low', 'Close',  # Price group
            'Volume', 'Taker_buy_volume', 'Number_of_trades',  # Volume group
            'Minutes_of_day', 'Hour', 'Minute'  # Time group
        ]
        
        np.random.seed(42)
        X = np.random.rand(100, len(feature_names))
        
        # Test feature grouping
        scaler = GroupedScaler()
        scaler.fit(X, feature_names)
        
        feature_groups = scaler.get_feature_groups()
        print(f"   ✅ Feature groups found: {list(feature_groups.keys())}")
        
        # Check that features are properly grouped
        for group_name, features in feature_groups.items():
            print(f"   ✅ {group_name} group: {features}")
            assert len(features) > 0, f"{group_name} group should not be empty"
        
        # Test scaling info
        scaling_info = scaler.get_scaling_info()
        assert len(scaling_info) == len(feature_groups), "Scaling info should match feature groups"
        print(f"   ✅ Scaling info available for {len(scaling_info)} groups")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature grouping test error: {e}")
        return False

def test_scale_time_series_data_grouped():
    """Test the scale_time_series_data_grouped function."""
    print("🧪 Testing scale_time_series_data_grouped...")
    
    try:
        from utils import scale_time_series_data_grouped
        
        # Create test data
        np.random.seed(42)
        X_train = np.random.rand(100, 10, 5)
        X_test = np.random.rand(20, 10, 5)
        y_train = np.random.rand(100, 3)
        y_test = np.random.rand(20, 3)
        feature_cols = ['Price_Range', 'Price_Change', 'Volume', 'Taker_buy_volume', 'Minutes_of_day']
        
        # Test scaling
        result = scale_time_series_data_grouped(X_train, X_test, y_train, y_test, feature_cols)
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = result
        
        # Validate results
        assert X_train_scaled.shape == X_train.shape, "X_train shape should remain the same"
        assert X_test_scaled.shape == X_test.shape, "X_test shape should remain the same"
        assert y_train_scaled.shape == y_train.shape, "y_train shape should remain the same"
        assert y_test_scaled.shape == y_test.shape, "y_test shape should remain the same"
        
        print(f"   ✅ Scaled shapes: X_train={X_train_scaled.shape}, y_train={y_train_scaled.shape}")
        print(f"   ✅ X_train range: {X_train_scaled.min():.4f} to {X_train_scaled.max():.4f}")
        print(f"   ✅ y_train range: {y_train_scaled.min():.4f} to {y_train_scaled.max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ scale_time_series_data_grouped test error: {e}")
        return False

def test_predict_with_grouped_scaler():
    """Test the predict_with_grouped_scaler function."""
    print("🧪 Testing predict_with_grouped_scaler...")
    
    try:
        from utils import predict_with_grouped_scaler, GroupedScaler
        from sklearn.preprocessing import MinMaxScaler
        
        # Create mock data
        np.random.seed(42)
        data = pd.DataFrame({
            'Open time': pd.date_range('2021-01-01', periods=50, freq='5min'),
            'Open': np.random.rand(50) * 50000 + 30000,
            'High': np.random.rand(50) * 50000 + 30000,
            'Low': np.random.rand(50) * 50000 + 30000,
            'Close': np.random.rand(50) * 50000 + 30000,
            'Volume': np.random.rand(50) * 1000,
            'Number of trades': np.random.randint(100, 1000, 50),
            'Taker buy base asset volume': np.random.rand(50) * 500,
            'Taker buy quote asset volume': np.random.rand(50) * 1000000
        })
        
        # Create mock scalers
        scaler_X = GroupedScaler()
        scaler_y = MinMaxScaler()
        
        # Mock feature columns
        feature_cols = ['Number of trades', 'Taker buy base asset volume', 'Minutes_of_day', 
                       'Price_Range', 'Price_Change', 'Price_Change_Pct', 'Volume_MA_5', 'Volume_MA_10']
        
        # Test prediction
        prediction = predict_with_grouped_scaler(
            model=None,  # We'll skip actual model prediction
            new_data_df=data,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            feature_cols=feature_cols,
            timesteps=5
        )
        
        if prediction is not None:
            print(f"   ✅ Prediction shape: {prediction.shape}")
            print(f"   ✅ Prediction columns: {list(prediction.columns)}")
        else:
            print("   ⚠️ Prediction returned None (expected without model)")
        
        return True
        
    except Exception as e:
        print(f"❌ predict_with_grouped_scaler test error: {e}")
        return False

def main():
    """Run all normalization tests."""
    print("🚀 NORMALIZATION TEST SUITE")
    print("=" * 40)
    
    tests = [
        ("GroupedScaler Basic", test_grouped_scaler_basic),
        ("GroupedScaler 3D", test_grouped_scaler_3d),
        ("Feature Groups", test_feature_groups),
        ("Scale Time Series Data", test_scale_time_series_data_grouped),
        ("Predict with Grouped Scaler", test_predict_with_grouped_scaler),
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
