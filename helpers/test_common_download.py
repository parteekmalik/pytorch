#!/usr/bin/env python3
"""
Test the common download function.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append('/Users/parteekmalik/github/pytorch')

def test_common_download():
    """Test the common download function."""
    print("🧪 TESTING COMMON DOWNLOAD FUNCTION")
    print("=" * 50)
    
    try:
        from utils import download_crypto_data
        
        # Test 1: Download 1 month of data
        print("📥 Test 1: Download 1 month (January 2021)...")
        data_1month = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 1",
            max_rows=10000,
            # Uses Binance Vision only
        )
        
        if data_1month is not None:
            print(f"✅ 1 month download: {len(data_1month)} rows")
            print(f"   Date range: {data_1month['Open time'].min()} to {data_1month['Open time'].max()}")
        else:
            print("❌ 1 month download failed")
        
        # Test 2: Download 3 months of data
        print(f"\\n📥 Test 2: Download 3 months (Jan-Mar 2021)...")
        data_3months = download_crypto_data(
            symbol="BTCUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 3",
            max_rows=20000,
            # Uses Binance Vision only
        )
        
        if data_3months is not None:
            print(f"✅ 3 months download: {len(data_3months)} rows")
            print(f"   Date range: {data_3months['Open time'].min()} to {data_3months['Open time'].max()}")
        else:
            print("❌ 3 months download failed")
        
        # Test 3: Test with different symbol
        print(f"\\n📥 Test 3: Download ETH data...")
        data_eth = download_crypto_data(
            symbol="ETHUSDT",
            interval="5m",
            data_from="2021 01",
            data_to="2021 1",
            max_rows=5000,
            # Uses Binance Vision only
        )
        
        if data_eth is not None:
            print(f"✅ ETH download: {len(data_eth)} rows")
            print(f"   Date range: {data_eth['Open time'].min()} to {data_eth['Open time'].max()}")
        else:
            print("❌ ETH download failed")
        
        # Summary
        print(f"\\n📊 Common Download Test Summary:")
        print(f"   1 month BTC: {'✅' if data_1month is not None else '❌'}")
        print(f"   3 months BTC: {'✅' if data_3months is not None else '❌'}")
        print(f"   1 month ETH: {'✅' if data_eth is not None else '❌'}")
        
        success_count = sum([data_1month is not None, data_3months is not None, data_eth is not None])
        print(f"   Success rate: {success_count}/3")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    """Run common download tests."""
    print("🧪 TESTING COMMON DOWNLOAD FUNCTIONALITY")
    print("=" * 60)
    
    success = test_common_download()
    
    if success:
        print(f"\\n🎉 Common download function is working!")
        print(f"   Ready to use in all notebooks")
        return True
    else:
        print(f"\\n❌ Common download function has issues")
        print(f"   Check the implementation")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎉 Common download function test passed!")
        sys.exit(0)
    else:
        print("\\n❌ Common download function test failed.")
        sys.exit(1)
