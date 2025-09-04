#!/usr/bin/env python3
"""
Test all notebooks with the common download function.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append('/Users/parteekmalik/github/pytorch')

def test_original_notebook():
    """Test original notebook configuration."""
    print("🧪 TESTING ORIGINAL NOTEBOOK CONFIGURATION")
    print("=" * 50)
    
    try:
        from utils import download_crypto_data
        
        # Test with original notebook configuration
        print("📥 Testing original notebook config (1 month)...")
        data = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 1",
            max_rows=10000,
            # Uses Binance Vision only
        )
        
        if data is not None:
            print(f"✅ Original notebook config: {len(data)} rows")
            print(f"   Date range: {data['Open time'].min()} to {data['Open time'].max()}")
            return True
        else:
            print("❌ Original notebook config failed")
            return False
            
    except Exception as e:
        print(f"❌ Original notebook test failed: {e}")
        return False

def test_streaming_notebook():
    """Test streaming notebook configuration."""
    print("\\n🧪 TESTING STREAMING NOTEBOOK CONFIGURATION")
    print("=" * 50)
    
    try:
        from utils import download_crypto_data
        
        # Test with streaming notebook configuration
        print("📥 Testing streaming notebook config (6 months)...")
        data = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 6",
            max_rows=20000,
            # Uses Binance Vision only
        )
        
        if data is not None:
            print(f"✅ Streaming notebook config: {len(data)} rows")
            print(f"   Date range: {data['Open time'].min()} to {data['Open time'].max()}")
            return True
        else:
            print("❌ Streaming notebook config failed")
            return False
            
    except Exception as e:
        print(f"❌ Streaming notebook test failed: {e}")
        return False

def test_ondemand_notebook():
    """Test on-demand notebook configuration."""
    print("\\n🧪 TESTING ON-DEMAND NOTEBOOK CONFIGURATION")
    print("=" * 50)
    
    try:
        from utils import download_crypto_data
        
        # Test with on-demand notebook configuration
        print("📥 Testing on-demand notebook config (6 months)...")
        data = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 6",
            max_rows=15000,
            # Uses Binance Vision only
        )
        
        if data is not None:
            print(f"✅ On-demand notebook config: {len(data)} rows")
            print(f"   Date range: {data['Open time'].min()} to {data['Open time'].max()}")
            return True
        else:
            print("❌ On-demand notebook config failed")
            return False
            
    except Exception as e:
        print(f"❌ On-demand notebook test failed: {e}")
        return False

def main():
    """Run all notebook tests."""
    print("🧪 TESTING ALL NOTEBOOKS WITH COMMON DOWNLOAD FUNCTION")
    print("=" * 70)
    
    # Test all notebooks
    original_success = test_original_notebook()
    streaming_success = test_streaming_notebook()
    ondemand_success = test_ondemand_notebook()
    
    # Summary
    print(f"\\n📊 ALL NOTEBOOKS TEST SUMMARY:")
    print(f"   Original notebook: {'✅' if original_success else '❌'}")
    print(f"   Streaming notebook: {'✅' if streaming_success else '❌'}")
    print(f"   On-demand notebook: {'✅' if ondemand_success else '❌'}")
    
    success_count = sum([original_success, streaming_success, ondemand_success])
    print(f"   Success rate: {success_count}/3")
    
    if success_count == 3:
        print(f"\\n🎉 All notebooks are working with common download function!")
        print(f"   Ready for production use")
        return True
    else:
        print(f"\\n❌ Some notebooks have issues")
        print(f"   Check the implementations")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎉 All notebook tests passed!")
        sys.exit(0)
    else:
        print("\\n❌ Some notebook tests failed.")
        sys.exit(1)
